"""
tests/retrieval/test_reader.py

Unit tests for Phase 2: Embedding + Retrieval.

Test strategy:
- EmbeddingModel tests: mock sentence-transformers entirely so tests run
  without downloading the 500MB model. We only test the wrapper logic
  (prefix handling, truncation, bytes output).
- Reader tests: use an in-memory SQLite DB (no sqlite-vec) so tests run
  without the native extension. BM25 path is fully testable; dense path
  is verified to gracefully degrade.
- Helper function tests: pure functions with no external dependencies —
  fully unit testable with no mocking needed.
"""

from __future__ import annotations

import sqlite3
import time
from datetime import UTC, datetime
from unittest.mock import MagicMock

import numpy as np
import pytest

from swarm_memory.core.models import Fact, FactType, SearchResult
from swarm_memory.retrieval.embeddings import PREFIX_DOCUMENT, PREFIX_QUERY, EmbeddingModel
from swarm_memory.retrieval.reader import (
    FactReader,
    _evict_stale_sessions,
    _session_seen,
    apply_gotcha_priority,
    end_session,
    format_results_for_agent,
    preprocess_for_fts5,
    reciprocal_rank_fusion,
    resolve_scope_tiers,
    timed,
)
from swarm_memory.store.db import get_initialized_db

# ═══════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════


@pytest.fixture
def in_memory_db() -> sqlite3.Connection:
    """
    Fresh in-memory SQLite DB with schema applied.
    No sqlite-vec (dense search disabled) — safe for any environment.
    """
    conn = get_initialized_db(":memory:")
    yield conn
    conn.close()


@pytest.fixture
def mock_embedder() -> EmbeddingModel:
    """
    EmbeddingModel with sentence-transformers mocked out.
    Returns deterministic 768-dim float32 vectors without downloading the model.
    """
    embedder = EmbeddingModel(model_name="mock-model", dim=768)
    # Inject a mock model that returns deterministic vectors
    mock_model = MagicMock()
    mock_model.encode = lambda text, **_kwargs: np.ones(768, dtype=np.float32)
    embedder._model = mock_model
    return embedder


@pytest.fixture
def reader(in_memory_db, mock_embedder) -> FactReader:
    """FactReader with in-memory DB and mocked embedder (no vec search)."""
    return FactReader(in_memory_db, mock_embedder, vec_available=False)


@pytest.fixture
def db_with_facts(in_memory_db) -> tuple[sqlite3.Connection, dict]:
    """
    DB pre-populated with a run + several facts for search testing.

    Returns (conn, ids) where ids maps short names to fact/run IDs.
    """
    conn = in_memory_db
    run_id = "run-test-001"
    now = datetime.now(UTC)

    # Insert a run (required FK for facts)
    conn.execute(
        "INSERT INTO runs (id, agent_id, repo, started_at) VALUES (?, ?, ?, ?)",
        (run_id, "agent-test", "test-repo", now.isoformat()),
    )

    facts_data = [
        # (id, content, fact_type, scope, content_hash)
        (
            "fact-A",
            "auth.logout() does not invalidate server sessions by default. Pass invalidate=True.",
            "gotcha",
            "test-repo/src/auth",
            "hash-A",
        ),
        (
            "fact-B",
            "Session tokens are stored in Redis with a 24h TTL.",
            "architecture",
            "test-repo/src/auth",
            "hash-B",
        ),
        (
            "fact-C",
            "All DB calls must use the connection pool in src/db.py.",
            "convention",
            "test-repo",
            "hash-C",
        ),
        (
            "fact-D",
            "The auth module uses JWT for API authentication.",
            "insight",
            "test-repo/src/auth",
            "hash-D",
        ),
        (
            "fact-E",
            "This fact is superseded and should never appear in current results.",
            "insight",
            "test-repo",
            "hash-E",
        ),
        (
            "fact-F",
            "FastAPI is the web framework used by this project.",
            "dependency",
            "test-repo",
            "hash-F",
        ),
    ]

    for fid, content, ftype, scope, chash in facts_data:
        conn.execute(
            """
            INSERT INTO facts
                (id, content, fact_type, scope, confidence, valid_from, source_run_id, content_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (fid, content, ftype, scope, 1.0, now.isoformat(), run_id, chash),
        )
        conn.execute(
            "INSERT INTO facts_fts (fact_id, content, scope) VALUES (?, ?, ?)",
            (fid, content, scope),
        )

    # Supersede fact-E
    conn.execute(
        "UPDATE facts SET valid_to = ?, superseded_by = 'fact-C' WHERE id = 'fact-E'",
        (now.isoformat(),),
    )

    return conn, {"run_id": run_id, "fact_ids": [row[0] for row in facts_data]}


# ═══════════════════════════════════════════════════════════════════════════
# Tests: EmbeddingModel
# ═══════════════════════════════════════════════════════════════════════════


class TestEmbeddingModel:
    def test_lazy_loading_not_loaded_at_init(self):
        """Model should NOT be loaded at construction time."""
        embedder = EmbeddingModel(model_name="some-model", dim=768)
        assert embedder._model is None
        assert not embedder.is_loaded

    def test_embed_returns_bytes_of_correct_length(self, mock_embedder):
        """embed() must return exactly (dim * 4) bytes (float32 = 4 bytes each)."""
        result = mock_embedder.embed("test text")
        assert isinstance(result, bytes)
        assert len(result) == 768 * 4

    def test_embed_uses_document_prefix_by_default(self, mock_embedder):
        """embed() should prepend PREFIX_DOCUMENT by default."""
        calls = []
        mock_embedder._model.encode = lambda text, **kw: (
            calls.append(text),
            np.ones(768, dtype=np.float32),
        )[1]
        mock_embedder.embed("some content")
        assert calls[0].startswith(PREFIX_DOCUMENT)

    def test_embed_query_uses_query_prefix(self, mock_embedder):
        """embed_query() must use PREFIX_QUERY, not PREFIX_DOCUMENT."""
        calls = []
        mock_embedder._model.encode = lambda text, **kw: (
            calls.append(text),
            np.ones(768, dtype=np.float32),
        )[1]
        mock_embedder.embed_query("search this")
        assert calls[0].startswith(PREFIX_QUERY)
        assert not calls[0].startswith(PREFIX_DOCUMENT)

    def test_dim_truncation(self):
        """Output should be truncated to configured dim even if model outputs more."""
        embedder = EmbeddingModel(dim=256)
        mock_model = MagicMock()
        # Model returns full 768 dims
        mock_model.encode = lambda text, **kw: np.ones(768, dtype=np.float32)
        embedder._model = mock_model

        result = embedder.embed("test")
        # Should be 256 * 4 bytes, not 768 * 4
        assert len(result) == 256 * 4

    def test_embed_batch_returns_correct_count(self, mock_embedder):
        """embed_batch() should return one bytes object per input text."""
        texts = ["fact one", "fact two", "fact three"]
        # Make encode return different arrays per call
        mock_embedder._model.encode = lambda texts_list, **kw: np.ones(
            (len(texts_list), 768), dtype=np.float32
        )
        results = mock_embedder.embed_batch(texts)
        assert len(results) == 3
        assert all(isinstance(r, bytes) for r in results)

    def test_embed_batch_empty_input(self):
        """embed_batch() with empty list should return empty list without calling model."""
        embedder = EmbeddingModel(dim=768)
        # Use a real MagicMock so we can assert_not_called
        mock_encode = MagicMock()
        embedder._model = MagicMock()
        embedder._model.encode = mock_encode

        results = embedder.embed_batch([])
        assert results == []
        mock_encode.assert_not_called()

    def test_dim_property(self, mock_embedder):
        """dim property should return the configured dimension."""
        assert mock_embedder.dim == 768

    def test_is_loaded_after_embed(self, mock_embedder):
        """is_loaded should be True after model has been used."""
        # mock_embedder already has _model set
        assert mock_embedder.is_loaded


# ═══════════════════════════════════════════════════════════════════════════
# Tests: Pure helper functions
# ═══════════════════════════════════════════════════════════════════════════


class TestResolveScopeTiers:
    def test_single_segment(self):
        """A bare repo name has only one tier."""
        result = resolve_scope_tiers("myrepo")
        assert result == [("myrepo", 0)]

    def test_two_segments(self):
        result = resolve_scope_tiers("myrepo/src")
        assert result == [("myrepo/src", 0), ("myrepo", 1)]

    def test_deep_path(self):
        result = resolve_scope_tiers("repo/src/auth/sessions")
        assert result == [
            ("repo/src/auth/sessions", 0),
            ("repo/src/auth", 1),
            ("repo/src", 2),
            ("repo", 3),
        ]

    def test_tiers_ordered_specific_to_general(self):
        """First element must be the most specific (tier 0)."""
        tiers = resolve_scope_tiers("a/b/c")
        assert tiers[0] == ("a/b/c", 0)
        assert tiers[-1] == ("a", 2)


class TestRecipRankFusion:
    def test_document_in_both_lists_ranks_highest(self):
        """A doc found in both lists should score higher than docs in only one."""
        dense = [("fact_A", 0.9), ("fact_C", 0.6)]
        bm25 = [("fact_B", 10.0), ("fact_A", 7.0)]
        fused = reciprocal_rank_fusion(dense, bm25)
        top_id = fused[0][0]
        assert top_id == "fact_A"

    def test_output_sorted_descending(self):
        """Scores in fused output must be strictly non-increasing."""
        dense = [("A", 1.0), ("B", 0.8), ("C", 0.6)]
        bm25 = [("C", 5.0), ("A", 4.0), ("D", 3.0)]
        fused = reciprocal_rank_fusion(dense, bm25)
        scores = [s for _, s in fused]
        assert scores == sorted(scores, reverse=True)

    def test_single_list(self):
        """Works with a single input list (no fusion needed)."""
        results = [("A", 0.9), ("B", 0.5)]
        fused = reciprocal_rank_fusion(results)
        assert fused[0][0] == "A"

    def test_weights_applied(self):
        """A list with weight 0 should contribute 0 to scores."""
        dense = [("A", 1.0)]
        bm25 = [("B", 1.0)]
        # dense has 0 weight: only BM25 contributes
        fused = dict(reciprocal_rank_fusion(dense, bm25, weights=[0.0, 1.0]))
        assert fused.get("A", 0.0) == 0.0
        assert fused["B"] > 0.0

    def test_k_parameter_affects_scores(self):
        """Larger k → lower scores (more uniform weighting)."""
        results = [("A", 1.0)]
        score_k60 = reciprocal_rank_fusion(results, k=60)[0][1]
        score_k10 = reciprocal_rank_fusion(results, k=10)[0][1]
        # k=10 gives 1/11 ≈ 0.0909; k=60 gives 1/61 ≈ 0.0164
        assert score_k10 > score_k60


class TestPreprocessForFts5:
    def test_strips_punctuation(self):
        result = preprocess_for_fts5("How do I connect?")
        assert "?" not in result

    def test_removes_stop_words(self):
        result = preprocess_for_fts5("How do I connect to the database?")
        assert result is not None
        # "connect" and "database" should survive; stop words should not
        terms = set(result.split(" OR "))
        assert "connect" in terms
        assert "database" in terms
        assert "how" not in terms
        assert "the" not in terms

    def test_returns_none_for_all_stop_words(self):
        """Vague queries with only stop words return None → dense-only fallback."""
        assert preprocess_for_fts5("What is it?") is None
        assert preprocess_for_fts5("how do we") is None

    def test_passthrough_single_technical_term(self):
        """A single technical keyword should pass through unchanged."""
        result = preprocess_for_fts5("authentication")
        assert result == "authentication"

    def test_or_joined(self):
        """Multiple terms should be joined with OR."""
        result = preprocess_for_fts5("redis connection pool")
        assert " OR " in result


class TestApplyGotchaPriority:
    def _make_result(self, fact_id: str, fact_type: FactType, score: float) -> SearchResult:
        fact = Fact(
            id=fact_id,
            content="test",
            fact_type=fact_type,
            scope="test",
            valid_from=datetime.now(UTC),
            source_run_id="run-1",
        )
        return SearchResult(fact=fact, relevance_score=score, retrieval_method="hybrid")

    def test_gotchas_are_first(self):
        results = [
            self._make_result("A", FactType.INSIGHT, 0.9),
            self._make_result("B", FactType.GOTCHA, 0.5),  # lower score but gotcha
            self._make_result("C", FactType.ARCHITECTURE, 0.8),
        ]
        ordered = apply_gotcha_priority(results)
        assert ordered[0].fact.id == "B"  # gotcha first
        assert ordered[0].fact.fact_type == FactType.GOTCHA

    def test_non_gotchas_preserve_order(self):
        results = [
            self._make_result("A", FactType.INSIGHT, 0.9),
            self._make_result("B", FactType.CONVENTION, 0.7),
        ]
        ordered = apply_gotcha_priority(results)
        # No gotchas → order unchanged
        assert [r.fact.id for r in ordered] == ["A", "B"]

    def test_multiple_gotchas_preserve_internal_order(self):
        results = [
            self._make_result("G1", FactType.GOTCHA, 0.8),
            self._make_result("A", FactType.INSIGHT, 0.9),
            self._make_result("G2", FactType.GOTCHA, 0.6),
        ]
        ordered = apply_gotcha_priority(results)
        # Both gotchas come first, in their original order relative to each other
        assert ordered[0].fact.id == "G1"
        assert ordered[1].fact.id == "G2"
        assert ordered[2].fact.id == "A"


class TestFormatResultsForAgent:
    def _make_result(self, fact_id: str, fact_type: FactType, content: str) -> SearchResult:
        fact = Fact(
            id=fact_id,
            content=content,
            fact_type=fact_type,
            scope="test-repo/src",
            valid_from=datetime.now(UTC),
            source_run_id="run-1",
        )
        return SearchResult(fact=fact, relevance_score=0.8, retrieval_method="hybrid")

    def test_empty_results_returns_no_facts_message(self):
        output = format_results_for_agent([], scope="myrepo/src")
        assert "no relevant facts found" in output.lower()
        assert "myrepo/src" in output

    def test_header_contains_count_and_scope(self):
        results = [self._make_result("A", FactType.INSIGHT, "test fact")]
        output = format_results_for_agent(results, scope="myrepo")
        assert "1 fact(s)" in output
        assert "myrepo" in output

    def test_gotcha_has_warning_prefix(self):
        results = [self._make_result("G", FactType.GOTCHA, "do not do this")]
        output = format_results_for_agent(results, scope="myrepo")
        assert "⚠" in output

    def test_non_gotcha_has_no_warning_prefix(self):
        results = [self._make_result("A", FactType.INSIGHT, "some insight")]
        output = format_results_for_agent(results, scope="myrepo")
        # Should NOT have ⚠ for non-gotcha
        lines = output.split("\n")
        fact_line = next(line for line in lines if "insight" in line)
        assert "⚠" not in fact_line

    def test_contains_fact_id_and_known_since(self):
        results = [self._make_result("fact-123", FactType.INSIGHT, "test")]
        output = format_results_for_agent(results, scope="myrepo")
        assert "fact-123" in output
        assert "known since" in output

    def test_no_raw_scores_in_output(self):
        results = [self._make_result("A", FactType.INSIGHT, "test")]
        output = format_results_for_agent(results, scope="myrepo")
        # Numeric scores must NOT be in the formatted output
        assert "0.8" not in output
        assert "relevance" not in output.lower()

    def test_invalidate_reminder_at_end(self):
        results = [self._make_result("A", FactType.INSIGHT, "test")]
        output = format_results_for_agent(results, scope="myrepo")
        assert "memory_invalidate" in output


# ═══════════════════════════════════════════════════════════════════════════
# Tests: Session deduplication
# ═══════════════════════════════════════════════════════════════════════════


class TestSessionDedup:
    def setup_method(self):
        """Clear global session state before each test."""
        _session_seen.clear()

    def test_end_session_removes_run_id(self):
        _session_seen["run-abc"] = ({"fact-1"}, time.time())
        end_session("run-abc")
        assert "run-abc" not in _session_seen

    def test_end_session_is_idempotent(self):
        """Calling end_session for unknown run_id should not raise."""
        end_session("run-does-not-exist")  # must not raise

    def test_evict_stale_sessions_removes_old_entries(self):
        # Insert a session that expired long ago
        old_timestamp = time.time() - 99999
        _session_seen["stale-run"] = ({"fact-1"}, old_timestamp)
        _session_seen["fresh-run"] = ({"fact-2"}, time.time())

        _evict_stale_sessions()

        assert "stale-run" not in _session_seen
        assert "fresh-run" in _session_seen

    def test_evict_stale_sessions_keeps_recent_entries(self):
        _session_seen["recent-run"] = ({"fact-1"}, time.time())
        _evict_stale_sessions()
        assert "recent-run" in _session_seen

    def test_search_with_dedup_filters_seen_ids(self, reader, db_with_facts):
        conn, ids = db_with_facts
        reader._conn = conn
        run_id = "test-run-dedup"

        # First search
        first = reader.search_with_dedup(
            "auth sessions", scope="test-repo", run_id=run_id, top_k=10
        )
        first_ids = {r.fact.id for r in first}

        # Second search for same thing — should get 0 overlap (everything already seen)
        second = reader.search_with_dedup(
            "auth sessions", scope="test-repo", run_id=run_id, top_k=10
        )
        second_ids = {r.fact.id for r in second}

        # No ID should appear in both first and second results
        overlap = first_ids & second_ids
        assert len(overlap) == 0

        # Clean up
        end_session(run_id)


# ═══════════════════════════════════════════════════════════════════════════
# Tests: FactReader — BM25 search (vec_available=False)
# ═══════════════════════════════════════════════════════════════════════════


class TestFactReaderBM25:
    """Tests for FactReader using BM25-only mode (no sqlite-vec required)."""

    @pytest.fixture
    def bm25_reader(self, db_with_facts, mock_embedder) -> FactReader:
        """
        FactReader built directly from the pre-populated db_with_facts connection.
        vec_available=False forces BM25-only retrieval.
        """
        conn, _ = db_with_facts
        return FactReader(conn, mock_embedder, vec_available=False)

    def test_search_returns_bm25_results(self, bm25_reader):
        # fact-B ("Redis TTL") is scoped to "test-repo/src/auth"
        # Searching with scope="test-repo/src/auth" resolves tiers:
        #   [("test-repo/src/auth", 0), ("test-repo/src", 1), ("test-repo", 2)]
        # So both fact-B (exact scope) and fact-C (repo root) are in scope.
        results = bm25_reader.search("Redis TTL", scope="test-repo/src/auth")
        assert len(results) > 0
        contents = [r.fact.content for r in results]
        assert any("Redis" in c for c in contents)

    def test_search_excludes_superseded_facts(self, bm25_reader):
        results = bm25_reader.search("superseded", scope="test-repo", top_k=10)
        fact_ids = [r.fact.id for r in results]
        # fact-E was superseded and should never appear
        assert "fact-E" not in fact_ids

    def test_search_scope_hierarchy(self, bm25_reader):
        # Searching from a sub-scope should also return parent-scope facts
        results = bm25_reader.search("connection pool", scope="test-repo/src/auth", top_k=10)
        fact_ids = [r.fact.id for r in results]
        # fact-C is scoped to "test-repo" (repo root) but should appear when
        # querying "test-repo/src/auth" because scope hierarchy includes the root
        assert "fact-C" in fact_ids

    def test_search_gotcha_appears_first(self, bm25_reader):
        results = bm25_reader.search("auth logout sessions", scope="test-repo/src/auth", top_k=5)
        if results:
            # If any gotcha is in the results, it must be ranked first
            gotchas = [r for r in results if r.fact.fact_type == FactType.GOTCHA]
            if gotchas:
                assert results[0].fact.fact_type == FactType.GOTCHA

    def test_search_vague_query_dense_fallback(self, bm25_reader):
        """An all-stop-word query should not crash — BM25 is skipped gracefully."""
        # "What is it?" → None from preprocess_for_fts5 → no BM25
        # With vec_available=False this returns empty (no results) without error
        results = bm25_reader.search("What is it?", scope="test-repo")
        assert isinstance(results, list)

    def test_search_fact_type_filter(self, bm25_reader):
        results = bm25_reader.search("auth", scope="test-repo", top_k=10, fact_type="gotcha")
        assert all(r.fact.fact_type == FactType.GOTCHA for r in results)

    def test_search_returns_at_most_top_k(self, bm25_reader):
        results = bm25_reader.search("auth", scope="test-repo", top_k=2)
        assert len(results) <= 2


# ═══════════════════════════════════════════════════════════════════════════
# Tests: Timed utility
# ═══════════════════════════════════════════════════════════════════════════


class TestTimedUtility:
    def test_timed_does_not_suppress_exceptions(self):
        """Exceptions inside `with timed(...)` should propagate."""
        with pytest.raises(ValueError), timed("test_block"):
            raise ValueError("intentional error")

    def test_timed_yields_control(self):
        """The timed block should execute the inner code."""
        executed = []
        with timed("test"):
            executed.append(True)
        assert executed == [True]
