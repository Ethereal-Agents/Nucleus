import os
from datetime import UTC, datetime, timedelta

import pytest

# Set up in-memory DB for tests
os.environ["SWARM_MEMORY_DB_PATH"] = ":memory:"

from unittest.mock import AsyncMock, MagicMock

import swarm_memory.server.mcp_server as mcp_module
from swarm_memory.core.embeddings import EmbeddingModel
from swarm_memory.ingestion.supersession import ConsolidationEngine
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


@pytest.fixture(autouse=True)
def reset_mcp_state():
    """Reinitialize all module-level singletons to ensure clean state per test."""
    mcp_module.db = get_initialized_db(":memory:")
    
    dummy_runs = ["dummy_run", "new1", "new2", "new_run", "new_run_x", "new", "run3", "run4", "r1", "r2", "r3"]
    for d in dummy_runs:
        mcp_module.db.execute(
            "INSERT INTO runs (id, repo, agent_id, arm, started_at) VALUES (?, ?, ?, ?, ?)",
            [d, "dummy_repo", "dummy_test", "arm3", datetime.now(UTC).isoformat()]
        )
    mcp_module.db.commit()

    # Mock Embedder to return zeros
    mcp_module.embedder = MagicMock(spec=EmbeddingModel)
    mcp_module.embedder.embed.return_value = b"\x00" * (768 * 4)
    mcp_module.embedder.embed_query.return_value = b"\x00" * (768 * 4)
    mcp_module.embedder.dim = 768

    # Mock consolidation engine: always treat new facts as independent so
    # semantically distinct facts are not accidentally superseded. Tests that
    # need real supersession use monkeypatch to restore the real engine.
    async def mock_consolidate(new_content, existing_facts):
        from swarm_memory.ingestion.supersession import ConsolidationResult
        return ConsolidationResult(
            status="independent",
            superseded_ids=[],
            merged_text=new_content,
        )

    async def mock_split_fact(content):
        import re as _re

        sentences = [s.strip() for s in _re.split(r"(?<=[.!?])\s+", content) if s.strip()]
        return sentences if sentences else [content]

    mcp_module.engine = AsyncMock(spec=ConsolidationEngine)
    mcp_module.engine.consolidate_facts.side_effect = mock_consolidate
    mcp_module.engine.split_fact.side_effect = mock_split_fact

    mcp_module.writer = FactWriter(
        db=mcp_module.db, embedder=mcp_module.embedder, engine=mcp_module.engine
    )
    mcp_module.reader = FactReader(conn=mcp_module.db, embedder=mcp_module.embedder)
    mcp_module.session_manager = SessionManager()
    yield


@pytest.mark.asyncio
async def test_auth_jwt_to_sessions_scenario():
    """
    Integration test: "auth JWT → sessions" scenario.
    (write → supersede → search current → search as_of).
    """
    t0 = datetime.now(UTC) - timedelta(days=2)
    t1 = datetime.now(UTC) - timedelta(days=1)

    # 1. Begin Run for original architecture
    begin_res1 = memory_begin_run(repo="testrepo", agent_id="agent1")
    run1_id = begin_res1["run_id"]

    # Write Fact A (JWT) at t0
    write_res1 = await memory_write(
        content="auth uses JWT",
        scope="auth",
        run_id=run1_id,
        fact_type="architecture",
        valid_from=t0.isoformat(),
    )
    assert write_res1["status"] == "created"
    fact1_id = write_res1["fact_ids"][0]

    # 2. Begin Run for new architecture
    begin_res2 = memory_begin_run(repo="testrepo", agent_id="agent1")
    run2_id = begin_res2["run_id"]

    # Write Fact B (sessions) at t1, superseding Fact A
    write_res2 = await memory_write(
        content="auth uses sessions",
        scope="auth",
        run_id=run2_id,
        fact_type="architecture",
        supersedes_hint=fact1_id,
        valid_from=t1.isoformat(),
    )
    # supersedes_hint goes through LLM flow; since LLM mock returns 'independent',
    # the hint_fact is appended post-LLM, yielding status='consolidated'
    assert write_res2["status"] == "consolidated"
    assert fact1_id in write_res2["superseded_ids"]

    # 3. Search current (should return sessions, not JWT)
    # Using a third run_id to avoid session deduplication filtering
    search_current = memory_search(query="auth", scope="auth", run_id="run3", top_k=5)
    assert "auth uses sessions" in search_current
    assert "auth uses JWT" not in search_current

    # 4. Search AS OF between t0 and t1 (should return JWT, not sessions)
    as_of_time = (t0 + timedelta(hours=12)).isoformat()
    search_as_of = memory_search(
        query="auth", scope="auth", run_id="run4", top_k=5, as_of=as_of_time
    )

    assert "auth uses JWT" in search_as_of
    assert "auth uses sessions" not in search_as_of


# ==========================================
# New Integration Tests (INT-01 to INT-12)
# ==========================================


@pytest.mark.asyncio
async def test_full_lifecycle_arm3():
    # INT-01
    res = memory_begin_run(repo="int_repo", agent_id="agent1", arm="arm3")
    run_id = res["run_id"]

    # Write 3 facts
    f1 = await memory_write(content="Fact 1", scope="repo", run_id=run_id)
    await memory_write(content="Fact 2", scope="repo", run_id=run_id)
    await memory_write(content="Fact 3", scope="repo", run_id=run_id)

    # Search
    s1 = memory_search(run_id="dummy_run", query="Fact", top_k=5)
    assert "Fact 1" in s1 and "Fact 2" in s1 and "Fact 3" in s1

    # Invalidate 1
    memory_invalidate(run_id="dummy_run", fact_id=f1["fact_ids"][0], reason="Invalidated")

    # Search again (with new run to bypass dedup)
    s2 = memory_search(query="Fact", run_id="new_run", top_k=5)
    assert "Fact 1" not in s2
    assert "Fact 2" in s2

    # End run
    await memory_end_run(run_id=run_id, summary="[]")

    # List runs
    runs = memory_list_runs(run_id="dummy_run", repo="int_repo")
    assert any(r["id"] == run_id for r in runs)


@pytest.mark.asyncio
async def test_full_lifecycle_arm2(monkeypatch):
    # INT-02
    import swarm_memory.server.mcp_server as mcp_module

    called_search = False

    def fake_search_traj(query, top_k):
        nonlocal called_search
        called_search = True
        return []

    monkeypatch.setattr(mcp_module.reader, "search_trajectories", fake_search_traj)

    res = memory_begin_run(repo="int_repo2", agent_id="agent1", arm="arm2")
    run_id = res["run_id"]

    called_write = False

    def fake_write_traj(traj, rid):
        nonlocal called_write
        called_write = True
        return [], []

    monkeypatch.setattr(mcp_module.writer, "write_trajectory", fake_write_traj)

    await memory_end_run(run_id=run_id, summary="[]", trajectory='[{"step": 1}]')

    # Pass run_id so _get_arm_for_run resolves to arm2 and routes to search_trajectories
    memory_search(query="test", run_id=run_id, top_k=5)
    assert called_write
    assert called_search


@pytest.mark.asyncio
async def test_supersession_chain():
    # INT-03
    res = memory_begin_run(repo="chain_repo", agent_id="a1")
    run_id = res["run_id"]

    fa = await memory_write(content="Fact A", scope="repo", run_id=run_id)
    fb = await memory_write(
        content="Fact B", scope="repo", run_id=run_id, supersedes_hint=fa["fact_ids"][0]
    )
    await memory_write(
        content="Fact C", scope="repo", run_id=run_id, supersedes_hint=fb["fact_ids"][0]
    )

    s = memory_search(query="Fact", run_id="new", top_k=5)
    assert "Fact A" not in s
    assert "Fact B" not in s
    assert "Fact C" in s


@pytest.mark.asyncio
async def test_multi_scope_hierarchy():
    # INT-04
    res = memory_begin_run(repo="scope_repo", agent_id="a1")
    run_id = res["run_id"]

    await memory_write(content="Auth fact", scope="repo/src/auth", run_id=run_id)
    await memory_write(content="DB fact", scope="repo/src/db", run_id=run_id)

    # Search from parent scope
    s = memory_search(query="fact", scope="repo", run_id="new1", top_k=5)
    assert "Auth fact" in s
    assert "DB fact" in s


@pytest.mark.asyncio
async def test_gotcha_priority_in_search():
    # INT-05
    res = memory_begin_run(repo="gotcha_repo", agent_id="a1")
    run_id = res["run_id"]

    await memory_write(content="Insight 1", scope="repo", run_id=run_id, fact_type="insight")
    await memory_write(content="Gotcha 1", scope="repo", run_id=run_id, fact_type="gotcha")
    await memory_write(content="Insight 2", scope="repo", run_id=run_id, fact_type="insight")

    s = memory_search(query="", run_id="new1", top_k=5)
    # The gotcha should appear first or have special priority text
    assert s.find("Gotcha 1") < s.find("Insight 1") or "⚠" in s


@pytest.mark.asyncio
async def test_time_travel_as_of():
    # INT-06
    t1 = datetime.now(UTC) - timedelta(days=2)
    t2 = datetime.now(UTC) - timedelta(days=1)

    res = memory_begin_run(repo="tt_repo", agent_id="a1")
    run_id = res["run_id"]

    f1 = await memory_write(
        content="TT Fact 1", scope="repo", run_id=run_id, valid_from=t1.isoformat()
    )
    await memory_write(
        content="TT Fact 2",
        scope="repo",
        run_id=run_id,
        supersedes_hint=f1["fact_ids"][0],
        valid_from=t2.isoformat(),
    )

    # As of time between t1 and t2
    as_of = (t1 + timedelta(hours=12)).isoformat()
    s = memory_search(query="TT Fact", as_of=as_of, run_id="new1", top_k=5)

    assert "TT Fact 1" in s
    assert "TT Fact 2" not in s


@pytest.mark.asyncio
async def test_session_dedup_across_searches():
    # INT-07
    res = memory_begin_run(repo="dedup_repo", agent_id="a1")
    run_id = res["run_id"]

    await memory_write(content="Dedup A", scope="repo", run_id=run_id)
    s1 = memory_search(query="Dedup", run_id=run_id, top_k=5)
    assert "Dedup A" in s1

    await memory_write(content="Dedup B", scope="repo", run_id=run_id)
    s2 = memory_search(query="Dedup", run_id=run_id, top_k=5)

    # Should only find new fact
    assert "Dedup B" in s2
    assert "Dedup A" not in s2


@pytest.mark.asyncio
async def test_concurrent_writes_same_scope():
    # INT-08
    res = memory_begin_run(repo="conc_repo", agent_id="a1")
    run_id = res["run_id"]

    await memory_write(content="Same Scope A", scope="repo/same", run_id=run_id)
    await memory_write(content="Same Scope B", scope="repo/same", run_id=run_id)

    s = memory_search(query="Same Scope", run_id="new1", top_k=5)
    assert "Same Scope A" in s
    assert "Same Scope B" in s


@pytest.mark.asyncio
async def test_fact_type_filter_search():
    # INT-09
    res = memory_begin_run(repo="type_repo", agent_id="a1")
    run_id = res["run_id"]

    await memory_write(content="Dep 1", scope="repo", run_id=run_id, fact_type="dependency")
    await memory_write(content="Gotcha 2", scope="repo", run_id=run_id, fact_type="gotcha")

    s = memory_search(query="", fact_type="dependency", run_id="new1", top_k=5)
    assert "Dep 1" in s
    assert "Gotcha 2" not in s


@pytest.mark.asyncio
async def test_invalidate_then_search():
    # INT-10
    res = memory_begin_run(repo="inv_repo", agent_id="a1")
    run_id = res["run_id"]

    f1 = await memory_write(content="Will invalid", scope="repo", run_id=run_id)
    memory_invalidate(run_id="dummy_run", fact_id=f1["fact_ids"][0], reason="Test")

    s = memory_search(query="Will invalid", run_id="new1", top_k=5)
    assert "Will invalid" not in s


@pytest.mark.asyncio
async def test_hybrid_search_fusion():
    # INT-11 (Basic mock test for hybrid search without vec)
    res = memory_begin_run(repo="hyb_repo", agent_id="a1")
    run_id = res["run_id"]

    await memory_write(content="Hybrid Fact", scope="repo", run_id=run_id)
    s = memory_search(query="Hybrid", run_id="new1", top_k=5)
    assert "Hybrid Fact" in s


@pytest.mark.skipif(not os.getenv("OPENROUTER_API_KEY"), reason="No API key")
@pytest.mark.asyncio
async def test_supersession_real_llm(monkeypatch):
    # INT-12
    import swarm_memory.server.mcp_server as mcp_module
    from swarm_memory.ingestion.supersession import ConsolidationEngine

    # Restore the real engine on both module and writer
    mcp_module.engine = ConsolidationEngine()
    mcp_module.writer.engine = mcp_module.engine
    # Use gpt-4o-mini which reliably returns structured JSON for contradiction detection
    mcp_module.engine.llm_service.model_name = "openrouter/openai/gpt-4o-mini"

    res = memory_begin_run(repo="real_repo", agent_id="a1")
    run_id = res["run_id"]

    f1 = await memory_write(content="The server runs on port 8080", scope="repo", run_id=run_id)
    # The real LLM should detect this as a contradiction
    f2 = await memory_write(
        content="The server runs on port 9000 instead of 8080", scope="repo", run_id=run_id
    )

    assert f1["fact_ids"][0] in f2["superseded_ids"]
