import json
import os
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

# Ensure tests use in-memory DB by default
os.environ["SWARM_MEMORY_DB_PATH"] = ":memory:"

import swarm_memory.server.mcp_server as mcp_module
from swarm_memory.core.embeddings import EmbeddingModel
from swarm_memory.ingestion.supersession import ContradictionDetector
from swarm_memory.ingestion.writer import FactWriter
from swarm_memory.retrieval.reader import FactReader
from swarm_memory.server.mcp_server import (
    memory_begin_run,
    memory_end_run,
    memory_invalidate,
    memory_list_runs,
    memory_search,
    memory_write,
)
from swarm_memory.server.session import SessionManager
from swarm_memory.store.db import get_initialized_db

# ==========================================
# Fixtures
# ==========================================

@pytest.fixture(scope="session")
def e2e_embedder():
    """Real EmbeddingModel loaded once for the entire test session."""
    return EmbeddingModel()

@pytest.fixture
def e2e_reset(e2e_embedder):
    """Resets module-level singletons with fresh in-memory DB, real embedder, mock detector."""
    mcp_module.db = get_initialized_db(":memory:")
    mcp_module.embedder = e2e_embedder
    
    # Check if vec is available, if not skip test requiring it
    try:
        mcp_module.db.execute("SELECT vec_version()")
        vec_available = True
    except Exception:
        vec_available = False
        pytest.skip("sqlite-vec is not available in this environment. E2E tests require it.")

    # Mock Detector to return empty relationships by default
    mcp_module.detector = MagicMock(spec=ContradictionDetector)
    mcp_module.detector.detect_contradictions = AsyncMock(side_effect=lambda *args, **kwargs: [])

    mcp_module.writer = FactWriter(
        db=mcp_module.db, embedder=mcp_module.embedder, detector=mcp_module.detector
    )
    mcp_module.reader = FactReader(conn=mcp_module.db, embedder=mcp_module.embedder, vec_available=vec_available)
    mcp_module.session_manager = SessionManager()
    yield

@pytest.fixture
def e2e_reset_with_llm(e2e_embedder):
    """Resets module-level singletons with real embedder AND real ContradictionDetector."""
    mcp_module.db = get_initialized_db(":memory:")
    mcp_module.embedder = e2e_embedder
    mcp_module.detector = ContradictionDetector()
    mcp_module.writer = FactWriter(
        db=mcp_module.db, embedder=mcp_module.embedder, detector=mcp_module.detector
    )
    # Check vec availability
    try:
        mcp_module.db.execute("SELECT vec_version()")
        vec_available = True
    except Exception:
        vec_available = False
        pytest.skip("sqlite-vec is not available in this environment. E2E tests require it.")
    mcp_module.reader = FactReader(conn=mcp_module.db, embedder=mcp_module.embedder, vec_available=vec_available)
    mcp_module.session_manager = SessionManager()
    yield

# Helper marker for tests requiring LLM
requires_llm = pytest.mark.skipif(
    not os.getenv("OPENROUTER_API_KEY"),
    reason="OPENROUTER_API_KEY not set. Requires real LLM."
)


# ==========================================
# Section 1: memory_begin_run
# ==========================================

def test_begin_run_returns_valid_uuid7(e2e_reset):
    # E2E-01
    res = memory_begin_run(repo="e2e_repo", agent_id="agent_1")
    assert "run_id" in res
    assert res["status"] == "started"
    assert isinstance(res["run_id"], str)
    assert len(res["run_id"]) > 30  # UUID string length

def test_begin_run_persists_all_fields(e2e_reset):
    # E2E-02
    res = memory_begin_run(repo="e2e_repo", agent_id="agent_2", branch="feature", model="gpt-4", arm="arm3")
    run_id = res["run_id"]
    row = mcp_module.db.execute("SELECT * FROM runs WHERE id = ?", [run_id]).fetchone()
    assert row["repo"] == "e2e_repo"
    assert row["agent_id"] == "agent_2"
    assert row["branch"] == "feature"
    assert row["model"] == "gpt-4"
    assert row["arm"] == "arm3"
    assert row["started_at"] is not None

def test_begin_run_optional_fields_nullable(e2e_reset):
    # E2E-03
    res = memory_begin_run(repo="e2e_repo", agent_id="agent_3")
    run_id = res["run_id"]
    row = mcp_module.db.execute("SELECT * FROM runs WHERE id = ?", [run_id]).fetchone()
    assert row["branch"] is None
    assert row["model"] is None

def test_begin_run_multiple_runs_unique_ids(e2e_reset):
    # E2E-04
    res1 = memory_begin_run(repo="e2e_repo", agent_id="agent_1")
    res2 = memory_begin_run(repo="e2e_repo", agent_id="agent_2")
    assert res1["run_id"] != res2["run_id"]

def test_begin_run_arm_variants(e2e_reset):
    # E2E-05
    res1 = memory_begin_run(repo="e2e_repo", agent_id="agent_1", arm="arm2")
    row1 = mcp_module.db.execute("SELECT arm FROM runs WHERE id = ?", [res1["run_id"]]).fetchone()
    assert row1["arm"] == "arm2"
    
    res2 = memory_begin_run(repo="e2e_repo", agent_id="agent_1", arm="arm3")
    row2 = mcp_module.db.execute("SELECT arm FROM runs WHERE id = ?", [res2["run_id"]]).fetchone()
    assert row2["arm"] == "arm3"


# ==========================================
# Section 2: memory_write
# ==========================================

@pytest.mark.asyncio
async def test_write_creates_fact_in_db(e2e_reset):
    # E2E-10
    run_id = memory_begin_run(repo="repo", agent_id="a1")["run_id"]
    res = await memory_write(content="Test content", scope="repo", run_id=run_id, confidence=0.8)
    assert res["status"] == "created"
    
    row = mcp_module.db.execute("SELECT * FROM facts WHERE id = ?", [res["fact_id"]]).fetchone()
    assert row["content"] == "Test content"
    assert row["scope"] == "repo"
    assert row["confidence"] == 0.8
    assert row["fact_type"] == "insight"

@pytest.mark.asyncio
async def test_write_creates_embedding_in_vec(e2e_reset):
    # E2E-11
    run_id = memory_begin_run(repo="repo", agent_id="a1")["run_id"]
    res = await memory_write(content="Test content", scope="repo", run_id=run_id)
    
    row = mcp_module.db.execute("SELECT * FROM facts_vec WHERE fact_id = ?", [res["fact_id"]]).fetchone()
    assert row is not None

@pytest.mark.asyncio
async def test_write_creates_fts_entry(e2e_reset):
    # E2E-12
    run_id = memory_begin_run(repo="repo", agent_id="a1")["run_id"]
    res = await memory_write(content="Test content fts", scope="repo", run_id=run_id)
    
    row = mcp_module.db.execute("SELECT * FROM facts_fts WHERE fact_id = ?", [res["fact_id"]]).fetchone()
    assert row["content"] == "Test content fts"

@pytest.mark.asyncio
async def test_write_all_fact_types(e2e_reset):
    # E2E-13
    run_id = memory_begin_run(repo="repo", agent_id="a1")["run_id"]
    types = ["insight", "gotcha", "convention", "architecture", "dependency"]
    
    for ft in types:
        res = await memory_write(content=f"Fact {ft}", scope="repo", run_id=run_id, fact_type=ft)
        row = mcp_module.db.execute("SELECT fact_type FROM facts WHERE id = ?", [res["fact_id"]]).fetchone()
        assert row["fact_type"] == ft

@pytest.mark.asyncio
async def test_write_duplicate_content_returns_duplicate(e2e_reset):
    # E2E-14
    run_id = memory_begin_run(repo="repo", agent_id="a1")["run_id"]
    res1 = await memory_write(content="Same content", scope="repo", run_id=run_id)
    assert res1["status"] == "created"
    
    res2 = await memory_write(content="Same content", scope="repo", run_id=run_id)
    assert res2["status"] == "duplicate"
    assert res1["fact_id"] == res2["fact_id"]

@pytest.mark.asyncio
async def test_write_with_supersedes_hint_invalidates_old(e2e_reset):
    # E2E-15
    run_id = memory_begin_run(repo="repo", agent_id="a1")["run_id"]
    res1 = await memory_write(content="Old fact", scope="repo", run_id=run_id)
    
    res2 = await memory_write(content="New fact", scope="repo", run_id=run_id, supersedes_hint=res1["fact_id"])
    assert res1["fact_id"] in res2["superseded_ids"]
    
    row1 = mcp_module.db.execute("SELECT valid_to, superseded_by FROM facts WHERE id = ?", [res1["fact_id"]]).fetchone()
    assert row1["valid_to"] is not None
    assert row1["superseded_by"] == res2["fact_id"]

@pytest.mark.asyncio
async def test_write_supersedes_hint_invalid_id_ignored(e2e_reset):
    # E2E-16
    run_id = memory_begin_run(repo="repo", agent_id="a1")["run_id"]
    res = await memory_write(content="Fact", scope="repo", run_id=run_id, supersedes_hint="invalid-id")
    assert res["status"] == "created"
    assert "invalid-id" not in res["superseded_ids"]

@pytest.mark.asyncio
async def test_write_custom_valid_from(e2e_reset):
    # E2E-17
    run_id = memory_begin_run(repo="repo", agent_id="a1")["run_id"]
    custom_date = "2024-01-01T00:00:00Z"
    expected_db_date = "2024-01-01T00:00:00+00:00"
    res = await memory_write(content="Fact", scope="repo", run_id=run_id, valid_from=custom_date)
    
    row = mcp_module.db.execute("SELECT valid_from FROM facts WHERE id = ?", [res["fact_id"]]).fetchone()
    assert row["valid_from"] == expected_db_date

@pytest.mark.asyncio
async def test_write_invalid_valid_from_raises(e2e_reset):
    # E2E-18
    run_id = memory_begin_run(repo="repo", agent_id="a1")["run_id"]
    with pytest.raises(ValueError):
        await memory_write(content="Fact", scope="repo", run_id=run_id, valid_from="not-a-date")

@pytest.mark.asyncio
async def test_write_confidence_persists(e2e_reset):
    # E2E-19
    run_id = memory_begin_run(repo="repo", agent_id="a1")["run_id"]
    res = await memory_write(content="Fact", scope="repo", run_id=run_id, confidence=0.42)
    row = mcp_module.db.execute("SELECT confidence FROM facts WHERE id = ?", [res["fact_id"]]).fetchone()
    # Using float comparison with tolerance if needed, or exact if SQLite preserves it well enough
    assert abs(row["confidence"] - 0.42) < 0.001


# ==========================================
# Section 3: memory_search
# ==========================================

@pytest.mark.asyncio
async def test_search_finds_semantically_similar_fact(e2e_reset):
    # E2E-20 (Requires real embedder and vec)
    run_id = memory_begin_run(repo="repo", agent_id="a1")["run_id"]
    await memory_write(content="The auth module uses JWT tokens exclusively", scope="repo/auth", run_id=run_id)
    
    # Query with different phrasing to test dense search
    res = memory_search(query="How does authentication work?", scope="repo", top_k=5)
    assert "JWT tokens" in res

@pytest.mark.asyncio
async def test_search_keyword_match_via_fts(e2e_reset):
    # E2E-21
    run_id = memory_begin_run(repo="repo", agent_id="a1")["run_id"]
    await memory_write(content="The primary database is PostgreSQL configured with WAL mode.", scope="repo", run_id=run_id)
    
    res = memory_search(query="PostgreSQL", scope="repo", top_k=5)
    assert "PostgreSQL" in res

@pytest.mark.asyncio
async def test_search_hybrid_fusion_both_paths(e2e_reset):
    # E2E-22
    run_id = memory_begin_run(repo="repo", agent_id="a1")["run_id"]
    await memory_write(content="Use Redis for caching and PostgreSQL for data.", scope="repo", run_id=run_id)
    await memory_write(content="Caching is done in-memory via Redis.", scope="repo", run_id=run_id)
    
    res = memory_search(query="Redis caching for data", top_k=5)
    assert "Redis" in res

@pytest.mark.asyncio
async def test_search_scope_hierarchy_resolution(e2e_reset):
    # E2E-23
    run_id = memory_begin_run(repo="repo", agent_id="a1")["run_id"]
    await memory_write(content="Auth fact", scope="repo/src/auth", run_id=run_id)
    
    # Search in parent scope
    res = memory_search(query="Auth", scope="repo/src", top_k=5)
    assert "Auth fact" in res

@pytest.mark.asyncio
async def test_search_scope_isolation(e2e_reset):
    # E2E-24
    run_id = memory_begin_run(repo="repo", agent_id="a1")["run_id"]
    await memory_write(content="Auth fact", scope="repo/src/auth", run_id=run_id)
    
    # Search in sibling scope
    res = memory_search(query="Auth", scope="repo/src/payments", top_k=5)
    assert "no relevant facts found" in res

@pytest.mark.asyncio
async def test_search_no_scope_searches_globally(e2e_reset):
    # E2E-25
    run_id = memory_begin_run(repo="repo", agent_id="a1")["run_id"]
    await memory_write(content="Secret global fact", scope="some/weird/path", run_id=run_id)
    
    res = memory_search(query="Secret", top_k=5)
    assert "Secret global fact" in res

@pytest.mark.asyncio
async def test_search_respects_top_k(e2e_reset):
    # E2E-26
    run_id = memory_begin_run(repo="repo", agent_id="a1")["run_id"]
    for i in range(5):
        await memory_write(content=f"Fact number {i}", scope="repo", run_id=run_id)
        
    res = memory_search(query="Fact number", top_k=2)
    # Check lines starting with "[insight]" or similar fact types
    fact_count = res.count("[insight]")
    assert fact_count <= 2

@pytest.mark.asyncio
async def test_search_session_dedup_filters_seen(e2e_reset):
    # E2E-27
    run_id = memory_begin_run(repo="repo", agent_id="a1")["run_id"]
    await memory_write(content="Dedup test fact", scope="repo", run_id=run_id)
    
    res1 = memory_search(query="Dedup", run_id=run_id, top_k=5)
    assert "Dedup test fact" in res1
    
    res2 = memory_search(query="Dedup", run_id=run_id, top_k=5)
    assert "Dedup test fact" not in res2

@pytest.mark.asyncio
async def test_search_without_run_id_no_dedup(e2e_reset):
    # E2E-28
    run_id = memory_begin_run(repo="repo", agent_id="a1")["run_id"]
    await memory_write(content="Dedup test fact", scope="repo", run_id=run_id)
    
    res1 = memory_search(query="Dedup", top_k=5)
    assert "Dedup test fact" in res1
    
    res2 = memory_search(query="Dedup", top_k=5)
    assert "Dedup test fact" in res2

@pytest.mark.asyncio
async def test_search_gotcha_priority(e2e_reset):
    # E2E-29
    run_id = memory_begin_run(repo="repo", agent_id="a1")["run_id"]
    await memory_write(content="Insight: use integers for IDs", scope="repo", run_id=run_id, fact_type="insight")
    await memory_write(content="Gotcha: do NOT use strings for IDs", scope="repo", run_id=run_id, fact_type="gotcha")
    
    res = memory_search(query="IDs", top_k=5)
    # Gotcha should be formatted with '⚠' and appear before the insight
    idx_gotcha = res.find("⚠")
    idx_insight = res.find("[insight]")
    assert idx_gotcha != -1
    assert idx_insight != -1
    assert idx_gotcha < idx_insight

@pytest.mark.asyncio
async def test_search_fact_type_filter(e2e_reset):
    # E2E-30
    run_id = memory_begin_run(repo="repo", agent_id="a1")["run_id"]
    await memory_write(content="Dependency A", scope="repo", run_id=run_id, fact_type="dependency")
    await memory_write(content="Insight B", scope="repo", run_id=run_id, fact_type="insight")
    
    res = memory_search(query="", fact_type="dependency", top_k=5)
    assert "Dependency A" in res
    assert "Insight B" not in res

@pytest.mark.asyncio
async def test_search_returns_formatted_string(e2e_reset):
    # E2E-31
    run_id = memory_begin_run(repo="repo", agent_id="a1")["run_id"]
    await memory_write(content="Format test fact", scope="repo/path", run_id=run_id)
    
    res = memory_search(query="Format", scope="repo/path", top_k=5)
    assert res.startswith("[MEMORY HUB —")
    assert "Format test fact" in res
    assert "id:" in res
    assert "known since:" in res


# ==========================================
# Section 4: memory_invalidate
# ==========================================

@pytest.mark.asyncio
async def test_invalidate_sets_valid_to(e2e_reset):
    # E2E-40
    run_id = memory_begin_run(repo="repo", agent_id="a1")["run_id"]
    res = await memory_write(content="To invalidate", scope="repo", run_id=run_id)
    fact_id = res["fact_id"]
    
    memory_invalidate(fact_id=fact_id, reason="Testing")
    row = mcp_module.db.execute("SELECT valid_to FROM facts WHERE id = ?", [fact_id]).fetchone()
    assert row["valid_to"] is not None

@pytest.mark.asyncio
async def test_invalidate_hides_from_search(e2e_reset):
    # E2E-41
    run_id = memory_begin_run(repo="repo", agent_id="a1")["run_id"]
    res = await memory_write(content="To invalidate hide", scope="repo", run_id=run_id)
    
    search1 = memory_search(query="hide", top_k=5)
    assert "To invalidate hide" in search1
    
    memory_invalidate(fact_id=res["fact_id"], reason="Testing")
    
    search2 = memory_search(query="hide", run_id="new", top_k=5)
    assert "To invalidate hide" not in search2

def test_invalidate_nonexistent_returns_error(e2e_reset):
    # E2E-42
    res = memory_invalidate(fact_id="not-an-id", reason="Testing")
    assert res["status"] == "error"

@pytest.mark.asyncio
async def test_invalidate_already_invalid_returns_error(e2e_reset):
    # E2E-43
    run_id = memory_begin_run(repo="repo", agent_id="a1")["run_id"]
    res = await memory_write(content="Double invalidate", scope="repo", run_id=run_id)
    fact_id = res["fact_id"]
    
    res1 = memory_invalidate(fact_id=fact_id, reason="Testing")
    assert res1["status"] == "invalidated"
    
    res2 = memory_invalidate(fact_id=fact_id, reason="Testing 2")
    assert res2["status"] == "error"

@pytest.mark.asyncio
async def test_invalidate_custom_valid_to(e2e_reset):
    # E2E-44
    run_id = memory_begin_run(repo="repo", agent_id="a1")["run_id"]
    res = await memory_write(content="Custom valid to", scope="repo", run_id=run_id)
    fact_id = res["fact_id"]
    
    custom_time = "2024-05-05T00:00:00Z"
    memory_invalidate(fact_id=fact_id, reason="Testing", valid_to=custom_time)
    row = mcp_module.db.execute("SELECT valid_to FROM facts WHERE id = ?", [fact_id]).fetchone()
    assert row["valid_to"] == custom_time


# ==========================================
# Section 5: memory_list_runs
# ==========================================

def test_list_runs_empty(e2e_reset):
    # E2E-50
    assert len(memory_list_runs(repo="empty")) == 0

def test_list_runs_returns_all_fields(e2e_reset):
    # E2E-51
    memory_begin_run(repo="list_repo", agent_id="a1", branch="main")
    runs = memory_list_runs(repo="list_repo")
    assert len(runs) == 1
    assert "id" in runs[0]
    assert "agent_id" in runs[0]
    assert "repo" in runs[0]
    assert "branch" in runs[0]
    assert "started_at" in runs[0]

def test_list_runs_repo_filter(e2e_reset):
    # E2E-52
    memory_begin_run(repo="r1", agent_id="a1")
    memory_begin_run(repo="r2", agent_id="a1")
    runs = memory_list_runs(repo="r1")
    assert len(runs) == 1
    assert runs[0]["repo"] == "r1"

def test_list_runs_limit(e2e_reset):
    # E2E-53
    memory_begin_run(repo="r1", agent_id="a1")
    memory_begin_run(repo="r1", agent_id="a2")
    memory_begin_run(repo="r1", agent_id="a3")
    runs = memory_list_runs(repo="r1", limit=2)
    assert len(runs) == 2

def test_list_runs_ordered_by_recency(e2e_reset):
    # E2E-54
    import time
    res1 = memory_begin_run(repo="r1", agent_id="a1")
    time.sleep(0.01)
    res2 = memory_begin_run(repo="r1", agent_id="a2")
    
    runs = memory_list_runs(repo="r1")
    assert runs[0]["id"] == res2["run_id"]
    assert runs[1]["id"] == res1["run_id"]


# ==========================================
# Section 6: memory_end_run
# ==========================================

@pytest.mark.asyncio
async def test_end_run_sets_finished_at(e2e_reset):
    # E2E-60
    run_id = memory_begin_run(repo="repo", agent_id="a1")["run_id"]
    await memory_end_run(run_id=run_id, summary="[]")
    row = mcp_module.db.execute("SELECT finished_at FROM runs WHERE id = ?", [run_id]).fetchone()
    assert row["finished_at"] is not None

@pytest.mark.asyncio
async def test_end_run_stores_token_counts(e2e_reset):
    # E2E-61
    run_id = memory_begin_run(repo="repo", agent_id="a1")["run_id"]
    await memory_end_run(run_id=run_id, summary="[]", input_tokens=10, output_tokens=20, total_cost_usd=0.5)
    row = mcp_module.db.execute("SELECT input_tokens, output_tokens, total_cost_usd FROM runs WHERE id = ?", [run_id]).fetchone()
    assert row["input_tokens"] == 10
    assert row["output_tokens"] == 20
    assert row["total_cost_usd"] == 0.5

@pytest.mark.asyncio
async def test_end_run_empty_summary_completes(e2e_reset):
    # E2E-62
    run_id = memory_begin_run(repo="repo", agent_id="a1")["run_id"]
    res = await memory_end_run(run_id=run_id, summary="[]")
    assert res["status"] == "completed"
    assert res["facts_saved"] == 0

@pytest.mark.asyncio
async def test_end_run_extracts_facts_from_summary(e2e_reset):
    # E2E-63
    run_id = memory_begin_run(repo="repo", agent_id="a1")["run_id"]
    summary = json.dumps([{"content": "Summary fact", "scope": "repo", "fact_type": "insight"}])
    res = await memory_end_run(run_id=run_id, summary=summary)
    
    assert res["status"] == "completed"
    assert res["facts_saved"] == 1
    
    search_res = memory_search(query="Summary fact", top_k=5)
    assert "Summary fact" in search_res

@pytest.mark.asyncio
async def test_end_run_malformed_json_returns_error(e2e_reset):
    # E2E-64
    run_id = memory_begin_run(repo="repo", agent_id="a1")["run_id"]
    res = await memory_end_run(run_id=run_id, summary="not json")
    assert res["status"] == "error"

@pytest.mark.asyncio
async def test_end_run_partial_failure(e2e_reset, monkeypatch):
    # E2E-65
    run_id = memory_begin_run(repo="repo", agent_id="a1")["run_id"]
    summary = json.dumps([
        {"content": "Good fact", "scope": "repo", "fact_type": "insight"},
        {"content": "Bad fact", "scope": "repo", "fact_type": "insight"}
    ])
    
    orig_write = mcp_module.writer.write_fact
    async def mock_write(*args, **kwargs):
        if kwargs.get("content") == "Bad fact":
            raise ValueError("Error")
        return await orig_write(*args, **kwargs)
    monkeypatch.setattr(mcp_module.writer, "write_fact", mock_write)
    
    res = await memory_end_run(run_id=run_id, summary=summary)
    assert res["status"] == "partial"
    assert res["facts_saved"] == 1
    assert res["facts_errored"] == 1

@pytest.mark.asyncio
async def test_end_run_cleans_up_session(e2e_reset):
    # E2E-66
    run_id = memory_begin_run(repo="repo", agent_id="a1")["run_id"]
    mcp_module.session_manager.mark_seen(run_id, ["some-id"])
    assert len(mcp_module.session_manager.get_seen_ids(run_id)) > 0
    
    await memory_end_run(run_id=run_id, summary="[]")
    assert len(mcp_module.session_manager.get_seen_ids(run_id)) == 0

@pytest.mark.asyncio
async def test_end_run_markdown_fenced_json(e2e_reset):
    # E2E-67
    run_id = memory_begin_run(repo="repo", agent_id="a1")["run_id"]
    summary = "```json\n[{\"content\": \"Fenced fact\", \"scope\": \"repo\", \"fact_type\": \"insight\"}]\n```"
    res = await memory_end_run(run_id=run_id, summary=summary)
    assert res["status"] == "completed"
    assert res["facts_saved"] == 1


# ==========================================
# Section 7: Cross-Cutting Scenarios
# ==========================================

@pytest.mark.asyncio
async def test_full_agent_lifecycle(e2e_reset):
    # E2E-70
    run_id = memory_begin_run(repo="e2e_lifecycle", agent_id="a1")["run_id"]
    
    res1 = await memory_write(content="F1", scope="repo", run_id=run_id)
    res2 = await memory_write(content="F2", scope="repo", run_id=run_id)
    res3 = await memory_write(content="F3", scope="repo", run_id=run_id)
    
    search_res = memory_search(query="F", top_k=5)
    assert "F1" in search_res
    
    memory_invalidate(fact_id=res1["fact_id"], reason="Del")
    
    search_res2 = memory_search(query="F", run_id="new_run_x", top_k=5)
    assert "F1" not in search_res2
    assert "F2" in search_res2
    
    summary = json.dumps([{"content": "Summary F4", "scope": "repo", "fact_type": "insight"}])
    await memory_end_run(run_id=run_id, summary=summary)
    
    runs = memory_list_runs(repo="e2e_lifecycle")
    assert any(r["id"] == run_id for r in runs)
    
    search_res3 = memory_search(query="F4", top_k=5)
    assert "Summary F4" in search_res3

@pytest.mark.asyncio
async def test_supersession_chain_3_deep(e2e_reset):
    # E2E-71
    run_id = memory_begin_run(repo="repo", agent_id="a1")["run_id"]
    f_a = await memory_write(content="F A", scope="repo", run_id=run_id)
    f_b = await memory_write(content="F B", scope="repo", run_id=run_id, supersedes_hint=f_a["fact_id"])
    f_c = await memory_write(content="F C", scope="repo", run_id=run_id, supersedes_hint=f_b["fact_id"])
    
    res = memory_search(query="F", run_id="new1", top_k=5)
    assert "F A" not in res
    assert "F B" not in res
    assert "F C" in res

@pytest.mark.asyncio
async def test_time_travel_as_of(e2e_reset):
    # E2E-72
    t0 = datetime.now(UTC) - timedelta(days=2)
    t1 = datetime.now(UTC) - timedelta(days=1)
    
    run_id = memory_begin_run(repo="repo", agent_id="a1")["run_id"]
    f1 = await memory_write(content="Old TT", scope="repo", run_id=run_id, valid_from=t0.isoformat())
    f2 = await memory_write(content="New TT", scope="repo", run_id=run_id, supersedes_hint=f1["fact_id"], valid_from=t1.isoformat())
    
    as_of = (t0 + timedelta(hours=12)).isoformat()
    s_past = memory_search(query="TT", as_of=as_of, run_id="new1", top_k=5)
    assert "Old TT" in s_past
    assert "New TT" not in s_past
    
    s_curr = memory_search(query="TT", run_id="new2", top_k=5)
    assert "Old TT" not in s_curr
    assert "New TT" in s_curr

@pytest.mark.asyncio
async def test_multi_agent_concurrent_writes(e2e_reset):
    # E2E-73
    r1 = memory_begin_run(repo="repo", agent_id="a1")["run_id"]
    r2 = memory_begin_run(repo="repo", agent_id="a2")["run_id"]
    
    await memory_write(content="Agent 1 fact", scope="repo", run_id=r1)
    await memory_write(content="Agent 2 fact", scope="repo", run_id=r2)
    
    s = memory_search(query="Agent", top_k=5)
    assert "Agent 1 fact" in s
    assert "Agent 2 fact" in s

@pytest.mark.asyncio
async def test_cross_scope_visibility(e2e_reset):
    # E2E-74
    r1 = memory_begin_run(repo="repo", agent_id="a1")["run_id"]
    await memory_write(content="Auth fact", scope="repo/auth", run_id=r1)
    await memory_write(content="DB fact", scope="repo/db", run_id=r1)
    
    s_repo = memory_search(query="fact", scope="repo", top_k=5)
    assert "Auth fact" in s_repo and "DB fact" in s_repo
    
    s_auth = memory_search(query="fact", scope="repo/auth", top_k=5)
    assert "Auth fact" in s_auth
    assert "DB fact" not in s_auth

@pytest.mark.asyncio
async def test_session_dedup_across_multiple_searches(e2e_reset):
    # E2E-75
    r1 = memory_begin_run(repo="repo", agent_id="a1")["run_id"]
    await memory_write(content="S1", scope="repo", run_id=r1)
    await memory_write(content="S2", scope="repo", run_id=r1)
    
    res1 = memory_search(query="S", run_id=r1, top_k=5)
    assert "S1" in res1 and "S2" in res1
    
    res2 = memory_search(query="S", run_id=r1, top_k=5)
    assert "S1" not in res2 and "S2" not in res2
    
    await memory_write(content="S3", scope="repo", run_id=r1)
    res3 = memory_search(query="S", run_id=r1, top_k=5)
    assert "S3" in res3

@pytest.mark.asyncio
async def test_end_run_facts_searchable_by_new_agent(e2e_reset):
    # E2E-76
    r1 = memory_begin_run(repo="repo", agent_id="a1")["run_id"]
    await memory_end_run(run_id=r1, summary=json.dumps([{"content": "Cross agent fact", "scope": "repo", "fact_type": "insight"}]))
    
    r2 = memory_begin_run(repo="repo", agent_id="a2")["run_id"]
    s = memory_search(query="Cross agent", run_id=r2, top_k=5)
    assert "Cross agent fact" in s

@pytest.mark.asyncio
async def test_duplicate_write_is_idempotent(e2e_reset):
    # E2E-77
    r1 = memory_begin_run(repo="repo", agent_id="a1")["run_id"]
    w1 = await memory_write(content="Idempotent", scope="repo", run_id=r1)
    w2 = await memory_write(content="Idempotent", scope="repo", run_id=r1)
    w3 = await memory_write(content="Idempotent", scope="repo", run_id=r1)
    
    assert w1["fact_id"] == w2["fact_id"] == w3["fact_id"]
    
    c = mcp_module.db.execute("SELECT COUNT(*) as c FROM facts WHERE content = 'Idempotent'").fetchone()
    assert c["c"] == 1


# ==========================================
# Section 8: Real LLM Contradiction
# ==========================================

@requires_llm
@pytest.mark.asyncio
async def test_llm_detects_contradiction(e2e_reset_with_llm):
    # E2E-80
    mcp_module.detector.llm_service.model_name = "openrouter/openai/gpt-4o-mini"
    r1 = memory_begin_run(repo="repo", agent_id="a1")["run_id"]
    f1 = await memory_write(content="The web server runs on port 8080", scope="repo", run_id=r1)
    
    # Write a contradicting fact without supersedes_hint, LLM should catch it
    f2 = await memory_write(content="The web server has been updated to run on port 9000 instead", scope="repo", run_id=r1)
    
    assert f1["fact_id"] in f2["superseded_ids"]
    row = mcp_module.db.execute("SELECT valid_to FROM facts WHERE id = ?", [f1["fact_id"]]).fetchone()
    assert row["valid_to"] is not None

@requires_llm
@pytest.mark.asyncio
async def test_llm_independent_facts_coexist(e2e_reset_with_llm):
    # E2E-81
    mcp_module.detector.llm_service.model_name = "openrouter/openai/gpt-4o-mini"
    r1 = memory_begin_run(repo="repo", agent_id="a1")["run_id"]
    f1 = await memory_write(content="Authentication uses JWT", scope="repo", run_id=r1)
    f2 = await memory_write(content="The database is PostgreSQL", scope="repo", run_id=r1)
    
    assert f1["fact_id"] not in f2["superseded_ids"]
    row = mcp_module.db.execute("SELECT valid_to FROM facts WHERE id = ?", [f1["fact_id"]]).fetchone()
    assert row["valid_to"] is None

@requires_llm
@pytest.mark.asyncio
async def test_llm_refines_fact_keeps_both(e2e_reset_with_llm):
    # E2E-82
    mcp_module.detector.llm_service.model_name = "openrouter/openai/gpt-4o-mini"
    r1 = memory_begin_run(repo="repo", agent_id="a1")["run_id"]
    f1 = await memory_write(content="API uses REST", scope="repo", run_id=r1)
    f2 = await memory_write(content="API uses REST following the JSON:API specification strict guidelines", scope="repo", run_id=r1)
    
    assert f1["fact_id"] not in f2["superseded_ids"]
    row = mcp_module.db.execute("SELECT valid_to FROM facts WHERE id = ?", [f1["fact_id"]]).fetchone()
    assert row["valid_to"] is None


# ==========================================
# Section 9: Edge Cases & Robustness
# ==========================================

@pytest.mark.asyncio
async def test_search_empty_db(e2e_reset):
    # E2E-90
    res = memory_search(query="anything", top_k=5)
    assert "no relevant facts found" in res

@pytest.mark.asyncio
async def test_write_unicode_content(e2e_reset):
    # E2E-91
    r1 = memory_begin_run(repo="repo", agent_id="a1")["run_id"]
    await memory_write(content="Unicode 🔧 数据库 café", scope="repo", run_id=r1)
    
    res = memory_search(query="Unicode café", top_k=5)
    assert "数据库" in res

@pytest.mark.asyncio
async def test_write_very_long_content(e2e_reset):
    # E2E-92
    r1 = memory_begin_run(repo="repo", agent_id="a1")["run_id"]
    long_content = "Word " * 1000
    await memory_write(content=long_content, scope="repo", run_id=r1)
    
    res = memory_search(query="Word", top_k=5)
    assert "Word" in res
    assert len(res) > 4000

@pytest.mark.asyncio
async def test_search_all_stopwords_query(e2e_reset):
    # E2E-93
    r1 = memory_begin_run(repo="repo", agent_id="a1")["run_id"]
    await memory_write(content="Just a standard fact here", scope="repo", run_id=r1)
    
    # "what is the" should strip down to empty for FTS, testing fallback
    res = memory_search(query="what is the", top_k=5)
    # The dense search should still return it if it's the only fact and we search without dedup
    assert "standard fact" in res

@pytest.mark.asyncio
async def test_end_run_summary_with_trailing_comma(e2e_reset):
    # E2E-94
    r1 = memory_begin_run(repo="repo", agent_id="a1")["run_id"]
    # Deliberate malformed JSON trailing comma
    summary = """
    [
      {"content": "Trailing comma fact", "scope": "repo", "fact_type": "insight"},
    ]
    """
    res = await memory_end_run(run_id=r1, summary=summary)
    assert res["status"] == "completed"
    assert res["facts_saved"] == 1
    
    search_res = memory_search(query="Trailing comma", top_k=5)
    assert "Trailing comma fact" in search_res
