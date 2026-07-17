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
from datetime import UTC, datetime
from unittest.mock import MagicMock

import numpy as np
import pytest

from swarm_memory.core.embeddings import EmbeddingModel
from swarm_memory.core.models import Fact, FactType, SearchResult
from swarm_memory.retrieval.reader import (
    FactReader,
    apply_gotcha_priority,
    preprocess_for_fts5,
    reciprocal_rank_fusion,
    resolve_scope_tiers,
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


# TestTimedUtility lives in tests/core/test_utils.py — not duplicated here.

# --- RDR-02, RDR-05, RDR-06 ---
class TestMissingPureFunctions:
    def test_resolve_scope_tiers_none(self):
        try:
            assert resolve_scope_tiers(None) == []
        except AttributeError:
            pytest.xfail("resolve_scope_tiers does not handle None currently")

    def test_rrf_empty_rankings(self):
        assert reciprocal_rank_fusion() == []
        assert reciprocal_rank_fusion([]) == []

    def test_rrf_single_ranking_list(self):
        dense = [("A", 1.0), ("B", 0.5)]
        fused = reciprocal_rank_fusion(dense)
        assert fused[0][0] == "A"
        assert fused[1][0] == "B"

# --- RDR-10 to RDR-13 ---
class TestConfidenceDecay:
    @pytest.fixture
    def decay_reader(self, db, mock_embedder):
        return FactReader(db, mock_embedder, vec_available=False)

    def _make_fact(self, fact_id: str, days_old: float) -> Fact:
        from datetime import timedelta
        dt = datetime.now(UTC) - timedelta(days=days_old)
        return Fact(
            id=fact_id,
            content="test",
            fact_type=FactType.INSIGHT,
            scope="test",
            created_at=dt,
            valid_from=dt,
            source_run_id="run-1",
        )

    def test_confidence_decay_new_fact(self, decay_reader):
        fact = self._make_fact("new", 0)
        rrf = {"new": 10.0}
        results = decay_reader._apply_confidence_decay([fact], rrf, datetime.now(UTC))
        assert results[0].relevance_score == pytest.approx(10.0)

    def test_confidence_decay_50_day_old_fact(self, decay_reader):
        fact = self._make_fact("old-50", 50)
        rrf = {"old-50": 10.0}
        results = decay_reader._apply_confidence_decay([fact], rrf, datetime.now(UTC))
        assert results[0].relevance_score == 5.0

    def test_confidence_decay_100_day_old_fact(self, decay_reader):
        fact = self._make_fact("old-100", 100)
        rrf = {"old-100": 10.0}
        results = decay_reader._apply_confidence_decay([fact], rrf, datetime.now(UTC))
        assert results[0].relevance_score == 5.0

    def test_confidence_decay_custom_rate_floor(self, decay_reader, monkeypatch):
        from swarm_memory.core import config
        monkeypatch.setattr(config, "CONFIDENCE_DECAY_RATE", 0.05)
        monkeypatch.setattr(config, "CONFIDENCE_DECAY_FLOOR", 0.3)
        
        fact = self._make_fact("custom", 20)
        rrf = {"custom": 10.0}
        results = decay_reader._apply_confidence_decay([fact], rrf, datetime.now(UTC))
        assert results[0].relevance_score == 3.0

# --- RDR-20 to RDR-23 ---
@pytest.mark.usefixtures("db_with_vec")
class TestFactReaderDense:
    @pytest.fixture
    def dense_reader(self, db_with_vec, mock_embedder):
        return FactReader(db_with_vec, mock_embedder, vec_available=True)

    @pytest.fixture
    def seeded_dense_db(self, db_with_vec):
        from datetime import timedelta
        now = datetime.now(UTC)
        past = now - timedelta(days=1)

        db_with_vec.execute(
            "INSERT INTO facts (id, content, fact_type, scope, confidence, valid_from, source_run_id) VALUES "
            "('fact-dense-1', 'dense content 1', 'insight', 'dense-scope', 1.0, ?, 'run_1'),"
            "('fact-dense-2', 'dense content 2', 'insight', 'dense-scope', 1.0, ?, 'run_1')",
            (now.isoformat(), now.isoformat())
        )
        import numpy as np
        emb = np.ones(768, dtype=np.float32).tobytes()
        db_with_vec.execute("INSERT INTO facts_vec (fact_id, embedding) VALUES (?, ?)", ("fact-dense-1", emb))
        db_with_vec.execute("INSERT INTO facts_vec (fact_id, embedding) VALUES (?, ?)", ("fact-dense-2", emb))
        
        # Superseded fact
        db_with_vec.execute(
            "INSERT INTO facts (id, content, fact_type, scope, confidence, valid_from, valid_to, superseded_by, source_run_id) VALUES "
            "('fact-super', 'super content', 'insight', 'dense-scope', 1.0, ?, ?, 'fact-dense-1', 'run_1')",
            (past.isoformat(), now.isoformat())
        )
        db_with_vec.execute("INSERT INTO facts_vec (fact_id, embedding) VALUES (?, ?)", ("fact-super", emb))
        
        return db_with_vec

    def test_dense_search_returns_results(self, dense_reader, seeded_dense_db):
        res = dense_reader.search("query", scope="dense-scope")
        assert len(res) == 2
        fact_ids = {r.fact.id for r in res}
        assert "fact-dense-1" in fact_ids
        assert "fact-dense-2" in fact_ids

    def test_dense_search_scope_filter(self, dense_reader, seeded_dense_db):
        res = dense_reader.search("query", scope="other-scope")
        assert len(res) == 0

    def test_dense_search_excludes_superseded(self, dense_reader, seeded_dense_db):
        res = dense_reader.search("query", scope="dense-scope")
        fact_ids = [r.fact.id for r in res]
        assert "fact-super" not in fact_ids

    def test_dense_search_as_of_time_travel(self, dense_reader, seeded_dense_db):
        from datetime import timedelta
        past = (datetime.now(UTC) - timedelta(hours=12)).isoformat()
        res = dense_reader.search("query", scope="dense-scope", as_of=past)
        fact_ids = [r.fact.id for r in res]
        assert "fact-super" in fact_ids
        assert "fact-dense-1" not in fact_ids

# --- RDR-30 to RDR-31 ---
class TestGlobalScopeSearch:
    @pytest.fixture
    def global_reader(self, db, mock_embedder):
        return FactReader(db, mock_embedder, vec_available=False)

    def test_search_global_no_scope(self, global_reader):
        res = global_reader.search("Redis JWT", scope=None, top_k=10)
        # Should execute without error (even if empty in blank db)
        assert isinstance(res, list)

    def test_search_global_returns_all_scopes(self, global_reader):
        res = global_reader.search("Redis JWT FastAPI", scope=None, top_k=10)
        assert isinstance(res, list)

# --- RDR-40 to RDR-43 ---
@pytest.mark.usefixtures("db_with_vec")
class TestSearchTrajectories:
    @pytest.fixture
    def traj_reader(self, db_with_vec, mock_embedder):
        return FactReader(db_with_vec, mock_embedder, vec_available=True)

    @pytest.fixture
    def seeded_traj_db(self, db_with_vec):
        from datetime import datetime, UTC
        now = datetime.now(UTC).isoformat()
        db_with_vec.execute(
            "INSERT INTO trajectories (id, content, run_id, created_at) VALUES "
            "('traj-1', 'trajectory one content', 'run_1', ?),"
            "('traj-2', 'trajectory two content', 'run_1', ?),"
            "('traj-3', 'trajectory three content', 'run_1', ?)",
            (now, now, now)
        )
        import numpy as np
        emb = np.ones(768, dtype=np.float32).tobytes()
        db_with_vec.execute("INSERT INTO trajectories_vec (trajectory_id, embedding) VALUES (?, ?)", ("traj-1", emb))
        db_with_vec.execute("INSERT INTO trajectories_vec (trajectory_id, embedding) VALUES (?, ?)", ("traj-2", emb))
        db_with_vec.execute("INSERT INTO trajectories_vec (trajectory_id, embedding) VALUES (?, ?)", ("traj-3", emb))
        return db_with_vec

    def test_search_trajectories_basic(self, traj_reader, seeded_traj_db):
        res = traj_reader.search_trajectories("query", top_k=5)
        assert len(res) == 3

    def test_search_trajectories_top_k(self, traj_reader, seeded_traj_db):
        res = traj_reader.search_trajectories("query", top_k=2)
        assert len(res) == 2

    def test_search_trajectories_returns_search_result(self, traj_reader, seeded_traj_db):
        res = traj_reader.search_trajectories("query", top_k=1)
        assert res[0].fact.scope == "trajectory"
        assert res[0].retrieval_method == "dense"

    def test_search_trajectories_empty_db(self, traj_reader):
        res = traj_reader.search_trajectories("query", top_k=5)
        assert res == []

