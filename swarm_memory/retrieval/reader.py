"""
swarm_memory/retrieval/reader.py

Hybrid retrieval engine for SwarmMemory.

Implements the full read path that powers every memory_search tool call:

  Agent query
    ├─ 1. resolve_scope_tiers      → [exact scope, parent, repo root]
    ├─ 2. embed_query              → float32 bytes
    ├─ 3. _dense_search            → top-20 via sqlite-vec KNN
    ├─ 4. preprocess_for_fts5      → FTS5 expression (or None → dense-only)
    ├─ 5. _bm25_search             → top-20 via FTS5 BM25
    ├─ 6. reciprocal_rank_fusion   → merged + fused scores
    ├─ 7. _apply_temporal_filter   → keep only currently-valid facts
    ├─ 8. _apply_confidence_decay  → recency bias (1%/day, floor 0.5)
    ├─ 9. search_with_dedup        → strip already-seen IDs for this run_id
    ├─ 10. apply_gotcha_priority   → ⚠ facts always surface first
    ├─ 11. trim to top_k
    └─ 12. format_results_for_agent → LLM-readable context block string

See implementation_plan.md §3, §14 for the full design rationale.
"""

from __future__ import annotations

import contextlib
import logging
import re
import sqlite3
import time
from collections import defaultdict
from collections.abc import Generator
from contextlib import contextmanager
from datetime import UTC, datetime

from swarm_memory.core import config
from swarm_memory.core.models import Fact, FactType, SearchResult
from swarm_memory.retrieval.embeddings import EmbeddingModel

logger = logging.getLogger(__name__)

# ── Stop words for FTS5 NL preprocessing (§3.6) ────────────────────────────

_STOP_WORDS: frozenset[str] = frozenset(
    {
        "what", "is", "the", "how", "do", "does", "a", "an", "to", "for",
        "in", "of", "and", "or", "should", "i", "we", "it", "are", "be",
        "with", "this", "that", "can", "right", "way", "when", "where",
        "why", "which", "there", "was", "its",
    }
)

# ── Session deduplication state (§14.3) ─────────────────────────────────────
# Module-level dict so it is shared across all FactReader instances within the
# same server process — the key property for a centralized SSE server.
#
# Structure: run_id → (set of fact IDs already shown, last-access unix timestamp)
_session_seen: dict[str, tuple[set[str], float]] = {}


# ═══════════════════════════════════════════════════════════════════════════
# Timing utility
# ═══════════════════════════════════════════════════════════════════════════

@contextmanager
def timed(label: str) -> Generator[None, None, None]:
    """
    Context manager that logs the wall-clock duration of a code block.

    Shared utility — also importable by Person A's ingestion/writer.py.

    Example:
        with timed("dense_search"):
            results = db.execute(...)
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.debug("⏱  %s: %.1f ms", label, elapsed_ms)


# ═══════════════════════════════════════════════════════════════════════════
# Pure helper functions (no DB access, fully unit-testable)
# ═══════════════════════════════════════════════════════════════════════════

def resolve_scope_tiers(scope: str) -> list[tuple[str, int]]:
    """
    Break a scope path into a hierarchy of increasingly broad scopes.

    When an agent queries scope "repo/src/auth/sessions", it should also
    receive facts scoped to parent modules — because a gotcha in "repo/src/auth"
    is just as relevant. Sibling scopes ("repo/src/payments") are excluded.

    Args:
        scope: A UNIX-path-style scope string (e.g., "myrepo/src/auth/sessions").

    Returns:
        List of (scope_string, tier_int) tuples from most-specific to most-general.
        Tier 0 = exact match (highest priority), tier N = repo root (broadest).

    Example:
        resolve_scope_tiers("repo/src/auth")
        → [("repo/src/auth", 0), ("repo/src", 1), ("repo", 2)]
    """
    parts = scope.split("/")
    tiers = []
    for i in range(len(parts), 0, -1):
        ancestor = "/".join(parts[:i])
        tier = len(parts) - i  # 0 = most specific
        tiers.append((ancestor, tier))
    return tiers


def reciprocal_rank_fusion(
    *result_lists: list[tuple[str, float]],
    k: int = config.RRF_K,
    weights: list[float] | None = None,
) -> list[tuple[str, float]]:
    """
    Fuse multiple ranked result lists using Reciprocal Rank Fusion (RRF).

    RRF is a parameter-free rank aggregation method. For each document in each
    list, its RRF score contribution is:  weight / (k + rank)

    Documents that appear in multiple lists accumulate contributions from each,
    so cross-list agreement is rewarded.

    Args:
        *result_lists: Variable number of ranked lists. Each list is
                       [(doc_id, raw_score), ...] sorted best-first.
                       raw_score is only used for ordering within each list;
                       the RRF formula ignores the actual score values.
        k:            Smoothing constant. Default 60 (standard literature value).
                      Higher k → more uniform weight across ranks.
        weights:      Per-list multipliers. Defaults to [1.0, 1.0, ...].
                      Dense search is weighted 1.0, BM25 at 0.8 per the plan.

    Returns:
        [(doc_id, rrf_score), ...] sorted descending by fused score.

    Example:
        dense =  [("fact_A", 0.9), ("fact_C", 0.7), ("fact_D", 0.5)]
        bm25  =  [("fact_B", 12.), ("fact_A", 8.),  ("fact_E", 6.)]
        fused = reciprocal_rank_fusion(dense, bm25, weights=[1.0, 0.8])
        # fact_A wins — found by both systems
    """
    if weights is None:
        weights = [1.0] * len(result_lists)

    scores: dict[str, float] = defaultdict(float)
    for weight, results in zip(weights, result_lists, strict=False):
        for rank, (doc_id, _raw_score) in enumerate(results, start=1):
            scores[doc_id] += weight / (k + rank)

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def preprocess_for_fts5(query: str) -> str | None:
    """
    Convert a natural language query into a valid FTS5 MATCH expression.

    FTS5's MATCH operator is a query language, not a plain text search.
    Raw NL queries with characters like ?, ', !, (, ) will raise syntax errors.
    This function sanitizes the query and strips stop words.

    Args:
        query: Raw natural language query from an agent.

    Returns:
        A valid FTS5 expression string, or None if no meaningful terms remain
        (caller should fall back to dense-only retrieval in that case).

    Examples:
        "How do I connect to the database?"  → "connect OR database"
        "What's the gotcha with auth logout?" → "gotcha OR auth OR logout"
        "authentication"                      → "authentication"
        "What is it?"                         → None  (all stop words)
    """
    # Strip FTS5 special characters to prevent syntax errors
    clean = re.sub(r"[^\w\s]", " ", query.lower())

    # Keep only non-stop, non-trivial tokens (len > 2 avoids noise like "in", "at")
    terms = [
        word
        for word in clean.split()
        if word not in _STOP_WORDS and len(word) > 2
    ]

    if not terms:
        return None  # signal to caller: skip BM25, use dense-only

    # OR-join: partial matches surface results (AND would be too restrictive)
    return " OR ".join(terms)


def apply_gotcha_priority(results: list[SearchResult]) -> list[SearchResult]:
    """
    Re-rank results so gotcha facts always surface first.

    The most expensive failure mode is an agent making a mistake that a prior
    agent already documented as a gotcha. This final re-rank step ensures
    gotchas are always visible regardless of their RRF score.

    Each sub-group (gotchas, others) preserves its internal RRF ordering.

    Args:
        results: SearchResult list, typically after RRF + temporal filter.

    Returns:
        New list with gotchas first, then all other fact types.
    """
    gotchas = [r for r in results if r.fact.fact_type == FactType.GOTCHA]
    others = [r for r in results if r.fact.fact_type != FactType.GOTCHA]
    return gotchas + others


def format_results_for_agent(results: list[SearchResult], scope: str) -> str:
    """
    Format search results as a human/LLM-readable context block.

    Agents are LLMs. Raw JSON with RRF scores is noise to them. This function
    returns a structured plain-text block that agents can inject directly into
    their reasoning context.

    Key formatting rules (§14.2):
    - Gotchas get a ⚠ prefix and always appear first (see apply_gotcha_priority).
    - fact_id and known-since date are included so agents can call
      memory_invalidate() without a second lookup.
    - Numeric scores (RRF, confidence, distance) are stripped — meaningless to LLMs.
    - Empty results still return a message (never silently empty).

    Args:
        results: Ordered list of SearchResult objects to format.
        scope:   The scope that was queried (used in the header).

    Returns:
        A formatted multi-line string, ready to inject into agent context.
    """
    if not results:
        return f"[MEMORY HUB — no relevant facts found for scope: {scope}]"

    lines = [f"[MEMORY HUB — {len(results)} fact(s) for {scope}]", ""]

    for result in results:
        fact = result.fact
        prefix = "⚠ " if fact.fact_type == FactType.GOTCHA else ""
        lines.append(f"{prefix}[{fact.fact_type.value}] {fact.scope}")
        lines.append(fact.content)
        # known_since is the valid_from date (when the fact became true)
        known_since = fact.valid_from.strftime("%Y-%m-%d") if fact.valid_from else "unknown"
        lines.append(f"→ id: {fact.id}  |  known since: {known_since}")
        lines.append("")

    lines.append('If any fact above is outdated, call: memory_invalidate(fact_id, reason="...")')
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# Session deduplication (module-level, shared across all FactReader instances)
# ═══════════════════════════════════════════════════════════════════════════

def _evict_stale_sessions() -> None:
    """
    Remove sessions that have been idle for longer than SESSION_TTL_SECONDS.

    This is a safety net for agents that crash without calling memory_end_run().
    Without eviction, orphaned sessions would leak memory in the server process.

    Called at the top of every search_with_dedup() — cheap O(sessions) scan
    since the number of concurrent agent sessions is expected to be small.
    """
    cutoff = time.time() - config.SESSION_TTL_SECONDS
    stale_ids = [
        run_id
        for run_id, (_seen_ids, last_access) in _session_seen.items()
        if last_access < cutoff
    ]
    for run_id in stale_ids:
        del _session_seen[run_id]
        logger.debug("Evicted stale session: %s", run_id)


def end_session(run_id: str) -> None:
    """
    Explicitly evict a session from the deduplication store.

    Called by the MCP server's memory_end_run() tool when an agent finishes
    its run cleanly. After this call, the agent's seen-fact-IDs set is cleared,
    so if the same agent starts a new run, it will receive fresh results.

    Args:
        run_id: The run ID returned by memory_begin_run().
    """
    _session_seen.pop(run_id, None)
    logger.debug("Session ended and evicted: %s", run_id)


# ═══════════════════════════════════════════════════════════════════════════
# FactReader — the main retrieval engine
# ═══════════════════════════════════════════════════════════════════════════

class FactReader:
    """
    Hybrid retrieval engine combining dense vector search and BM25 keyword search.

    Instantiate once per server process and pass the same instance to all
    request handlers. The embedding model is loaded lazily on first use.

    Args:
        conn:          sqlite3.Connection to an initialized SwarmMemory database.
        embedder:      EmbeddingModel instance for query embedding.
        vec_available: Whether sqlite-vec is loaded in the connection.
                       If False, falls back to BM25-only retrieval.

    Example:
        conn = get_initialized_db()
        reader = FactReader(conn, EmbeddingModel(), vec_available=True)
        results = reader.search("how does auth work?", scope="myrepo/src/auth")
        print(format_results_for_agent(results, scope="myrepo/src/auth"))
    """

    def __init__(
        self,
        conn: sqlite3.Connection,
        embedder: EmbeddingModel,
        vec_available: bool = True,
    ) -> None:
        self._conn = conn
        self._embedder = embedder
        self._vec_available = vec_available

    # ── Private retrieval methods ───────────────────────────────────────────

    def _dense_search(
        self,
        query_vec: bytes,
        scope_tiers: list[tuple[str, int]],
        top_n: int = 20,
    ) -> list[tuple[str, float]]:
        """
        KNN vector search via sqlite-vec.

        We JOIN facts_vec against facts to pre-filter by scope and validity
        before the KNN, preventing cross-scope contamination.

        Args:
            query_vec:   float32 bytes for the query (from embed_query).
            scope_tiers: Output of resolve_scope_tiers(scope).
            top_n:       Number of candidates to return (over-fetch for RRF).

        Returns:
            [(fact_id, distance), ...] sorted ascending by distance (lower = closer).
        """
        if not self._vec_available:
            return []

        scope_paths = [s for s, _ in scope_tiers]
        placeholders = ",".join("?" * len(scope_paths))

        with timed("dense_search"):
            rows = self._conn.execute(
                f"""
                SELECT fv.fact_id, fv.distance
                FROM facts_vec fv
                JOIN facts f ON fv.fact_id = f.id
                WHERE fv.embedding MATCH ?
                  AND f.scope IN ({placeholders})
                  AND f.valid_to IS NULL
                  AND f.superseded_by IS NULL
                ORDER BY fv.distance
                LIMIT ?
                """,
                [query_vec, *scope_paths, top_n],
            ).fetchall()

        # Lower distance = more similar; convert to (id, score) keeping distance
        return [(row["fact_id"], row["distance"]) for row in rows]

    def _bm25_search(
        self,
        fts_expr: str,
        scope_tiers: list[tuple[str, int]],
        top_n: int = 20,
    ) -> list[tuple[str, float]]:
        """
        BM25 keyword search via FTS5.

        FTS5's bm25() function returns negative values (less negative = higher rank).
        We keep the raw values because RRF only uses rank ordering, not magnitudes.

        Args:
            fts_expr:    Pre-processed FTS5 MATCH expression (from preprocess_for_fts5).
            scope_tiers: Output of resolve_scope_tiers(scope).
            top_n:       Number of candidates to return.

        Returns:
            [(fact_id, bm25_score), ...] sorted descending by BM25 (most relevant first).
        """
        scope_paths = [s for s, _ in scope_tiers]
        placeholders = ",".join("?" * len(scope_paths))

        with timed("bm25_search"):
            rows = self._conn.execute(
                f"""
                SELECT ff.fact_id, bm25(facts_fts) AS score
                FROM facts_fts ff
                JOIN facts f ON ff.fact_id = f.id
                WHERE facts_fts MATCH ?
                  AND f.scope IN ({placeholders})
                  AND f.valid_to IS NULL
                  AND f.superseded_by IS NULL
                ORDER BY score          -- bm25() is negative; higher (less negative) = better
                LIMIT ?
                """,
                [fts_expr, *scope_paths, top_n],
            ).fetchall()

        return [(row["fact_id"], row["score"]) for row in rows]

    def _fetch_facts_by_ids(
        self,
        fact_ids: list[str],
        as_of: str | None = None,
    ) -> list[Fact]:
        """
        Fetch full Fact objects for a list of IDs and apply temporal filtering.

        Temporal filter logic:
        - Default (as_of=None): only facts with valid_to IS NULL (currently valid)
        - Point-in-time (as_of=T): facts where valid_from <= T and (valid_to IS NULL
          or valid_to > T). This is the "what was true at time T?" query.

        Args:
            fact_ids: List of fact IDs to hydrate (from RRF fusion output).
            as_of:    Optional ISO-8601 timestamp for point-in-time queries.

        Returns:
            List of Fact Pydantic objects, temporally filtered.
        """
        if not fact_ids:
            return []

        placeholders = ",".join("?" * len(fact_ids))

        if as_of is None:
            # Standard query: currently valid facts only
            rows = self._conn.execute(
                f"""
                SELECT * FROM facts
                WHERE id IN ({placeholders})
                  AND valid_to IS NULL
                  AND superseded_by IS NULL
                """,
                fact_ids,
            ).fetchall()
        else:
            # Point-in-time query: facts that were valid at the given timestamp
            rows = self._conn.execute(
                f"""
                SELECT * FROM facts
                WHERE id IN ({placeholders})
                  AND valid_from <= ?
                  AND (valid_to IS NULL OR valid_to > ?)
                  AND superseded_by IS NULL
                """,
                [*fact_ids, as_of, as_of],
            ).fetchall()

        facts = []
        for row in rows:
            with contextlib.suppress(Exception):
                facts.append(Fact(**dict(row)))
        return facts

    def _apply_confidence_decay(
        self,
        facts: list[Fact],
        rrf_scores: dict[str, float],
        now: datetime,
    ) -> list[SearchResult]:
        """
        Apply recency decay to relevance scores and wrap as SearchResult objects.

        Decay formula: adjusted = rrf_score * max(FLOOR, 1.0 - age_days * RATE)
        At default settings (RATE=0.01, FLOOR=0.5):
          - A 30-day-old fact has 70% of its original score
          - A 100-day-old fact hits the floor at 50%

        Args:
            facts:      Hydrated Fact objects from _fetch_facts_by_ids.
            rrf_scores: Mapping of fact_id → RRF fused score.
            now:        Current datetime for age calculation.

        Returns:
            List of SearchResult objects with decay-adjusted relevance_score.
        """
        results = []
        for fact in facts:
            base_score = rrf_scores.get(fact.id, 0.0)

            # Calculate age in days (created_at is when the fact was registered)
            fact_time = fact.created_at
            if fact_time.tzinfo is None:
                fact_time = fact_time.replace(tzinfo=UTC)
            age_days = max(0.0, (now - fact_time).total_seconds() / 86400)

            # Recency decay: 1% per day, minimum 50% multiplier
            decay = max(
                config.CONFIDENCE_DECAY_FLOOR,
                1.0 - age_days * config.CONFIDENCE_DECAY_RATE,
            )
            adjusted_score = base_score * decay

            results.append(
                SearchResult(
                    fact=fact,
                    relevance_score=adjusted_score,
                    retrieval_method="hybrid" if self._vec_available else "bm25",
                )
            )
        return results

    # ── Public search API ───────────────────────────────────────────────────

    def search(
        self,
        query: str,
        scope: str | None = None,
        as_of: str | None = None,
        top_k: int = 5,
        fact_type: str | None = None,
    ) -> list[SearchResult]:
        """
        Run a full hybrid search query and return ranked SearchResult objects.

        This is the core method called by search_with_dedup() and directly
        by the MCP server's memory_search tool.

        Pipeline steps 1-10 (session dedup is step 9, handled by caller):
          1. Resolve scope tiers
          2. Embed query (dense path)
          3. Dense KNN search (or skip if vec unavailable)
          4. Preprocess query for FTS5 (sparse path)
          5. BM25 search (or skip if query is all stop words)
          6. RRF fusion (dense weighted 1.0, BM25 weighted 0.8)
          7. Temporal filter (valid_to IS NULL or as_of range)
          8. Confidence decay
          9. [Session dedup — handled externally by search_with_dedup]
          10. Gotcha-first re-rank
          11. Trim to top_k

        Args:
            query:     Natural language search query.
            scope:     Scope to search within. Applies hierarchy resolution.
                       None searches across all scopes (useful for session-start mode).
            as_of:     ISO-8601 timestamp for point-in-time query (default: now).
            top_k:     Maximum number of results to return.
            fact_type: Optional filter to restrict results to a specific FactType.

        Returns:
            List of SearchResult objects, ordered by adjusted relevance score,
            with gotchas always first.
        """
        now = datetime.now(UTC)
        OVER_FETCH = top_k * 4  # over-fetch so temporal/type filters have headroom

        # Step 1: Resolve scope hierarchy
        scope_tiers = resolve_scope_tiers(scope) if scope else []

        # Step 2 + 3: Dense search (always runs when vec is available)
        with timed("search.dense"):
            query_vec = self._embedder.embed_query(query)
            dense_results = (
                self._dense_search(query_vec, scope_tiers, top_n=OVER_FETCH)
                if scope_tiers
                else []
            )
            # Without scope, fall back to global dense (no scope filter)
            if not scope_tiers and self._vec_available:
                with timed("search.dense_global"):
                    rows = self._conn.execute(
                        """
                        SELECT fv.fact_id, fv.distance
                        FROM facts_vec fv
                        JOIN facts f ON fv.fact_id = f.id
                        WHERE fv.embedding MATCH ?
                          AND f.valid_to IS NULL
                          AND f.superseded_by IS NULL
                        ORDER BY fv.distance
                        LIMIT ?
                        """,
                        [query_vec, OVER_FETCH],
                    ).fetchall()
                    dense_results = [(r["fact_id"], r["distance"]) for r in rows]

        # Step 4 + 5: BM25 search (skipped if query is too vague)
        fts_expr = preprocess_for_fts5(query)
        bm25_results: list[tuple[str, float]] = []
        if fts_expr:
            with timed("search.bm25"):
                if scope_tiers:
                    bm25_results = self._bm25_search(fts_expr, scope_tiers, top_n=OVER_FETCH)
                else:
                    # Global BM25 (no scope filter)
                    rows = self._conn.execute(
                        """
                        SELECT ff.fact_id, bm25(facts_fts) AS score
                        FROM facts_fts ff
                        JOIN facts f ON ff.fact_id = f.id
                        WHERE facts_fts MATCH ?
                          AND f.valid_to IS NULL
                          AND f.superseded_by IS NULL
                        ORDER BY score
                        LIMIT ?
                        """,
                        [fts_expr, OVER_FETCH],
                    ).fetchall()
                    bm25_results = [(r["fact_id"], r["score"]) for r in rows]
        else:
            logger.debug("Query '%s' is all stop words — using dense-only retrieval", query)

        # Step 6: RRF fusion
        # dense weighted 1.0 (primary signal for semantic queries)
        # BM25 weighted 0.8 (booster for technical term matches)
        with timed("search.rrf"):
            if dense_results and bm25_results:
                fused = reciprocal_rank_fusion(
                    dense_results, bm25_results, weights=[1.0, 0.8]
                )
            elif dense_results:
                fused = [(fid, score) for fid, score in dense_results]
            elif bm25_results:
                fused = [(fid, score) for fid, score in bm25_results]
            else:
                return []  # nothing found

        # Build a score lookup for decay calculation later
        rrf_scores = dict(fused)
        candidate_ids = [fid for fid, _ in fused[: OVER_FETCH]]

        # Step 7: Temporal filter — hydrate facts and validate temporal window
        with timed("search.temporal_filter"):
            facts = self._fetch_facts_by_ids(candidate_ids, as_of=as_of)

        # Optional fact_type filter
        if fact_type:
            facts = [f for f in facts if f.fact_type.value == fact_type]

        # Step 8: Confidence decay → SearchResult objects
        results = self._apply_confidence_decay(facts, rrf_scores, now)

        # Preserve RRF order after decay (sort by adjusted score descending)
        results.sort(key=lambda r: r.relevance_score, reverse=True)

        # Step 10: Gotcha-first re-rank
        results = apply_gotcha_priority(results)

        # Step 11: Trim
        return results[:top_k]

    def search_with_dedup(
        self,
        query: str,
        scope: str | None,
        run_id: str,
        top_k: int = 5,
        as_of: str | None = None,
        fact_type: str | None = None,
    ) -> list[SearchResult]:
        """
        Search with in-session deduplication.

        Agents query multiple times per run. Without dedup, the same gotcha
        would appear in every response, wasting tokens and LLM attention.
        This method tracks which fact IDs have been shown for a given run_id
        and filters them out of subsequent queries.

        Over-fetches by len(seen) to compensate for filtered-out results,
        ensuring top_k fresh results are always returned when available.

        Args:
            query:     Natural language query.
            scope:     Scope filter (with hierarchy resolution).
            run_id:    The agent's current run ID (from memory_begin_run).
            top_k:     Number of fresh (not-yet-seen) results to return.
            as_of:     Optional point-in-time filter.
            fact_type: Optional fact type filter.

        Returns:
            List of SearchResult objects the agent has NOT yet seen this session.
        """
        # Evict stale sessions first (safety net for crashed agents)
        _evict_stale_sessions()

        # Get or create the session state for this run_id
        seen_ids, _ = _session_seen.get(run_id, (set(), 0.0))

        # Over-fetch: request extra results to cover already-seen IDs
        fetch_k = top_k + len(seen_ids)

        candidates = self.search(
            query=query,
            scope=scope,
            as_of=as_of,
            top_k=fetch_k,
            fact_type=fact_type,
        )

        # Filter to only fresh (not-yet-seen) results
        fresh = [r for r in candidates if r.fact.id not in seen_ids][:top_k]

        # Update session state with newly shown fact IDs
        new_seen = seen_ids | {r.fact.id for r in fresh}
        _session_seen[run_id] = (new_seen, time.time())

        return fresh
