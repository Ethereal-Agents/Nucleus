import os

import pytest

# Must set before importing mcp so that db initializes in memory
os.environ["SWARM_MEMORY_DB_PATH"] = ":memory:"

from swarm_memory.server.mcp_server import (  # noqa: E402
    memory_begin_run,
    memory_end_run,
    memory_invalidate,
    memory_list_runs,
    memory_search,
    memory_write,
)


@pytest.fixture(autouse=True)
def reset_mcp_state():
    """
    Reinitialise the in-memory DB and all module-level singletons before each test.

    mcp.py initialises db/writer/reader/session_manager at import time (module scope),
    so without this fixture every test shares the same DB and state leaks between them.
    This fixture replaces each singleton with a fresh instance before every test run.
    """
    import swarm_memory.server.mcp_server as mcp_module
    from swarm_memory.core.embeddings import EmbeddingModel
    from swarm_memory.ingestion.supersession import ContradictionDetector
    from swarm_memory.ingestion.writer import FactWriter
    from swarm_memory.retrieval.reader import FactReader
    from swarm_memory.server.session import SessionManager
    from swarm_memory.store.db import get_initialized_db

    mcp_module.db = get_initialized_db(":memory:")
    mcp_module.embedder = EmbeddingModel()
    mcp_module.detector = ContradictionDetector()
    mcp_module.writer = FactWriter(
        db=mcp_module.db, embedder=mcp_module.embedder, detector=mcp_module.detector
    )
    mcp_module.reader = FactReader(conn=mcp_module.db, embedder=mcp_module.embedder)
    mcp_module.session_manager = SessionManager()
    yield


@pytest.mark.asyncio
async def test_mcp_lifecycle():
    # 1. Begin Run
    begin_res = memory_begin_run(repo="testrepo", agent_id="testagent")
    assert "run_id" in begin_res
    assert begin_res["status"] == "started"
    run_id = begin_res["run_id"]

    # 2. Write initial fact
    write_res1 = await memory_write(
        content="The auth module uses JWT",
        scope="testrepo/src/auth",
        run_id=run_id,
        fact_type="architecture",
    )
    assert write_res1["status"] == "created"
    fact1_id = write_res1["fact_id"]

    # 3. Search and verify
    search_res1 = memory_search(query="auth", scope="testrepo/src/auth", run_id=run_id, top_k=5)
    assert "The auth module uses JWT" in search_res1
    assert fact1_id in search_res1

    # In-session dedup check: searching again with same run_id should NOT return it
    search_res1_dedup = memory_search(
        query="auth", scope="testrepo/src/auth", run_id=run_id, top_k=5
    )
    assert "The auth module uses JWT" not in search_res1_dedup

    # 4. Write superseding fact (using hint to bypass LLM contradiction detector)
    write_res2 = await memory_write(
        content="The auth module now uses sessions",
        scope="testrepo/src/auth",
        run_id=run_id,
        supersedes_hint=fact1_id,
    )
    assert write_res2["status"] == "created"
    fact2_id = write_res2["fact_id"]
    assert fact1_id in write_res2["superseded_ids"]

    # 5. Search for the new fact (using a new run_id to avoid dedup)
    search_res2 = memory_search(
        query="auth", scope="testrepo/src/auth", run_id="new_run_123", top_k=5
    )
    assert "The auth module now uses sessions" in search_res2
    assert "The auth module uses JWT" not in search_res2

    # 6. Manual Invalidation
    inv_res = memory_invalidate(fact_id=fact2_id, reason="Rolled back to JWT")
    assert inv_res["status"] == "invalidated"
    assert inv_res["fact_id"] == fact2_id

    # 7. Search again (should find no active facts since both are superseded/invalidated)
    search_res3 = memory_search(
        query="auth", scope="testrepo/src/auth", run_id="new_run_456", top_k=5
    )
    assert "no relevant facts found" in search_res3

    # 8. End run
    import json

    valid_summary = json.dumps(
        [{"content": "Completed auth testing", "scope": "testrepo/src/auth"}]
    )
    end_res = await memory_end_run(run_id=run_id, summary=valid_summary)
    assert end_res["status"] == "completed"

    # 9. List runs
    runs = memory_list_runs(repo="testrepo")
    assert len(runs) >= 1
    assert runs[0]["id"] == run_id
    assert runs[0]["summary"] == valid_summary


@pytest.mark.asyncio
async def test_memory_end_run_fact_extraction():
    import json

    # 1. Begin a new run
    begin_res = memory_begin_run(repo="testrepo", agent_id="extractor")
    run_id = begin_res["run_id"]

    # 2. End run with a JSON array summary
    summary_json = json.dumps(
        [
            {
                "content": "We migrated from PostgreSQL to MySQL.",
                "scope": "testrepo/infra/db",
                "fact_type": "architecture",
            },
            {
                "content": "Do not use the old user_id field, use uuid instead.",
                "scope": "testrepo/src/users",
                "fact_type": "gotcha",
            },
        ]
    )

    end_res = await memory_end_run(run_id=run_id, summary=summary_json)
    assert end_res["status"] == "completed"

    # 3. Search to verify both facts were written successfully
    search_res1 = memory_search(
        query="PostgreSQL", scope="testrepo/infra/db", run_id="some_new_run", top_k=5
    )
    assert "We migrated from PostgreSQL to MySQL." in search_res1

    search_res2 = memory_search(
        query="user_id", scope="testrepo/src/users", run_id="some_new_run2", top_k=5
    )
    assert "Do not use the old user_id field" in search_res2
    assert "⚠" in search_res2  # Because it's a gotcha


@pytest.mark.asyncio
async def test_memory_end_run_malformed_json_graceful():
    begin_res = memory_begin_run(repo="testrepo", agent_id="bad_extractor")
    run_id = begin_res["run_id"]

    # Provide a malformed JSON that cannot be repaired
    malformed_summary = "This is not json, it's just a string."

    # Should return an error asking the agent to retry
    end_res = await memory_end_run(run_id=run_id, summary=malformed_summary)
    assert end_res["status"] == "error"

    runs = memory_list_runs(repo="testrepo", limit=1)
    assert runs[0]["id"] == run_id
    assert runs[0]["summary"] is None


# ==========================================
# New Tests - memory_begin_run (MCP-01 to MCP-05)
# ==========================================
def test_begin_run_returns_run_id():
    res = memory_begin_run(repo="testrepo", agent_id="agent1")
    assert "run_id" in res
    assert res["status"] == "started"
    assert isinstance(res["run_id"], str)
    assert len(res["run_id"]) > 10


def test_begin_run_inserts_into_db():
    res = memory_begin_run(repo="testrepo", agent_id="agent1", branch="main", model="gpt-4")
    run_id = res["run_id"]
    from swarm_memory.server.mcp_server import db

    row = db.execute("SELECT * FROM runs WHERE id = ?", [run_id]).fetchone()
    assert row is not None
    assert row["repo"] == "testrepo"
    assert row["agent_id"] == "agent1"
    assert row["branch"] == "main"
    assert row["model"] == "gpt-4"
    assert row["arm"] == "arm3"


def test_begin_run_default_arm():
    res = memory_begin_run(repo="testrepo", agent_id="agent1")
    from swarm_memory.server.mcp_server import db

    row = db.execute("SELECT arm FROM runs WHERE id = ?", [res["run_id"]]).fetchone()
    assert row["arm"] == "arm3"


def test_begin_run_custom_arm():
    res = memory_begin_run(repo="testrepo", agent_id="agent1", arm="arm2")
    from swarm_memory.server.mcp_server import db

    row = db.execute("SELECT arm FROM runs WHERE id = ?", [res["run_id"]]).fetchone()
    assert row["arm"] == "arm2"


def test_begin_run_sets_context_vars():
    from swarm_memory.core.log import current_arm_id, current_run_id

    res = memory_begin_run(repo="testrepo", agent_id="agent1", arm="arm1")
    assert current_run_id.get() == res["run_id"]
    assert current_arm_id.get() == "arm1"


# ==========================================
# New Tests - memory_write (MCP-10 to MCP-12)
# ==========================================
@pytest.mark.asyncio
async def test_write_returns_write_result():
    run_res = memory_begin_run(repo="testrepo", agent_id="agent1")
    run_id = run_res["run_id"]
    res = await memory_write(content="Fact 1", scope="repo/path", run_id=run_id)
    assert "fact_id" in res
    assert "superseded_ids" in res
    assert res["status"] == "created"


@pytest.mark.asyncio
async def test_write_duplicate_returns_duplicate_status():
    run_res = memory_begin_run(repo="testrepo", agent_id="agent1")
    run_id = run_res["run_id"]
    res1 = await memory_write(content="Fact Duplicate", scope="repo/path", run_id=run_id)
    assert res1["status"] == "created"
    res2 = await memory_write(content="Fact Duplicate", scope="repo/path", run_id=run_id)
    assert res2["status"] == "duplicate"
    assert res1["fact_id"] == res2["fact_id"]


@pytest.mark.asyncio
async def test_write_with_supersedes_hint():
    run_res = memory_begin_run(repo="testrepo", agent_id="agent1")
    run_id = run_res["run_id"]
    res1 = await memory_write(content="Fact Old", scope="repo/path", run_id=run_id)
    res2 = await memory_write(
        content="Fact New", scope="repo/path", run_id=run_id, supersedes_hint=res1["fact_id"]
    )
    assert res2["status"] == "created"
    assert res1["fact_id"] in res2["superseded_ids"]


# ==========================================
# New Tests - memory_search (MCP-20 to MCP-26)
# ==========================================
@pytest.mark.asyncio
async def test_search_returns_formatted_string():
    res = memory_search(query="test", top_k=5)
    assert isinstance(res, str)
    assert res.startswith("[MEMORY HUB —")


@pytest.mark.asyncio
async def test_search_no_results():
    res = memory_search(query="nothing matches this", top_k=5)
    assert "no relevant facts found" in res


@pytest.mark.asyncio
async def test_search_session_dedup():
    run_res = memory_begin_run(repo="testrepo", agent_id="agent1")
    run_id = run_res["run_id"]
    await memory_write(content="Dedup fact", scope="repo/path", run_id=run_id)
    # Search first time
    res1 = memory_search(query="Dedup fact", run_id=run_id, top_k=5)
    assert "Dedup fact" in res1
    # Search second time, should be empty because of dedup
    res2 = memory_search(query="Dedup fact", run_id=run_id, top_k=5)
    assert "Dedup fact" not in res2


@pytest.mark.asyncio
async def test_search_no_run_id_no_dedup():
    run_res = memory_begin_run(repo="testrepo", agent_id="agent1")
    run_id = run_res["run_id"]
    await memory_write(content="No dedup fact", scope="repo/path", run_id=run_id)
    # Search without run_id
    res1 = memory_search(query="No dedup fact", top_k=5)
    assert "No dedup fact" in res1
    # Search again without run_id, still there
    res2 = memory_search(query="No dedup fact", top_k=5)
    assert "No dedup fact" in res2


@pytest.mark.asyncio
async def test_search_arm2_uses_trajectories(monkeypatch):
    import swarm_memory.server.mcp_server as mcp_module

    called = False

    def fake_search_trajectories(query, top_k):
        nonlocal called
        called = True
        return []

    monkeypatch.setattr(mcp_module.reader, "search_trajectories", fake_search_trajectories)
    memory_search(query="test", arm="arm2", top_k=5)
    assert called


@pytest.mark.asyncio
async def test_search_top_k_respected():
    run_res = memory_begin_run(repo="testrepo", agent_id="agent1")
    run_id = run_res["run_id"]
    await memory_write(content="K fact 1", scope="repo/path", run_id=run_id)
    await memory_write(content="K fact 2", scope="repo/path", run_id=run_id)
    await memory_write(content="K fact 3", scope="repo/path", run_id=run_id)

    res = memory_search(query="K fact", top_k=2)
    assert res.count("K fact") <= 2


@pytest.mark.asyncio
async def test_search_fact_type_filter():
    run_res = memory_begin_run(repo="testrepo", agent_id="agent1")
    run_id = run_res["run_id"]
    await memory_write(content="This is a gotcha", scope="repo", run_id=run_id, fact_type="gotcha")
    await memory_write(
        content="This is an insight", scope="repo", run_id=run_id, fact_type="insight"
    )

    res = memory_search(query="This is a", fact_type="gotcha")
    assert "gotcha" in res
    assert "insight" not in res


# ==========================================
# New Tests - memory_invalidate (MCP-30 to MCP-34)
# ==========================================
@pytest.mark.asyncio
async def test_invalidate_existing_fact():
    run_res = memory_begin_run(repo="testrepo", agent_id="agent1")
    run_id = run_res["run_id"]
    w_res = await memory_write(content="To be invalidated", scope="repo", run_id=run_id)
    fact_id = w_res["fact_id"]

    inv_res = memory_invalidate(fact_id=fact_id, reason="Testing")
    assert inv_res["status"] == "invalidated"
    assert inv_res["fact_id"] == fact_id


def test_invalidate_nonexistent_fact():
    inv_res = memory_invalidate(fact_id="nonexistent-id", reason="Testing")
    assert inv_res["status"] == "error"


@pytest.mark.asyncio
async def test_invalidate_sets_valid_to():
    run_res = memory_begin_run(repo="testrepo", agent_id="agent1")
    run_id = run_res["run_id"]
    w_res = await memory_write(content="To be invalidated valid_to", scope="repo", run_id=run_id)
    fact_id = w_res["fact_id"]

    memory_invalidate(fact_id=fact_id, reason="Testing")
    from swarm_memory.server.mcp_server import db

    row = db.execute("SELECT valid_to FROM facts WHERE id = ?", [fact_id]).fetchone()
    assert row["valid_to"] is not None


@pytest.mark.asyncio
async def test_invalidate_custom_valid_to():
    run_res = memory_begin_run(repo="testrepo", agent_id="agent1")
    run_id = run_res["run_id"]
    w_res = await memory_write(
        content="To be invalidated custom valid_to", scope="repo", run_id=run_id
    )
    fact_id = w_res["fact_id"]
    custom_time = "2024-01-01T00:00:00Z"

    memory_invalidate(fact_id=fact_id, reason="Testing", valid_to=custom_time)
    from swarm_memory.server.mcp_server import db

    row = db.execute("SELECT valid_to FROM facts WHERE id = ?", [fact_id]).fetchone()
    assert row["valid_to"] == custom_time


@pytest.mark.asyncio
async def test_invalidate_already_invalidated():
    run_res = memory_begin_run(repo="testrepo", agent_id="agent1")
    run_id = run_res["run_id"]
    w_res = await memory_write(content="Double invalidate", scope="repo", run_id=run_id)
    fact_id = w_res["fact_id"]

    res1 = memory_invalidate(fact_id=fact_id, reason="First")
    assert res1["status"] == "invalidated"

    res2 = memory_invalidate(fact_id=fact_id, reason="Second")
    assert res2["status"] == "error"


# ==========================================
# New Tests - memory_list_runs (MCP-40 to MCP-44)
# ==========================================
def test_list_runs_empty_db():
    runs = memory_list_runs(repo="empty_repo")
    assert len(runs) == 0


def test_list_runs_returns_correct_fields():
    memory_begin_run(repo="repo1", agent_id="agent1", branch="main")
    runs = memory_list_runs(repo="repo1")
    assert len(runs) >= 1
    run = runs[0]
    assert "id" in run
    assert "agent_id" in run
    assert "repo" in run
    assert "branch" in run
    assert "started_at" in run


def test_list_runs_repo_filter():
    memory_begin_run(repo="repoA", agent_id="agentA")
    memory_begin_run(repo="repoB", agent_id="agentB")
    runs = memory_list_runs(repo="repoA")
    assert len(runs) == 1
    assert runs[0]["repo"] == "repoA"


def test_list_runs_limit():
    memory_begin_run(repo="repoL", agent_id="agent1")
    memory_begin_run(repo="repoL", agent_id="agent2")
    memory_begin_run(repo="repoL", agent_id="agent3")
    runs = memory_list_runs(repo="repoL", limit=2)
    assert len(runs) == 2


def test_list_runs_ordered_by_recency():
    res1 = memory_begin_run(repo="repoO", agent_id="agent1")
    import time

    time.sleep(0.01)
    res2 = memory_begin_run(repo="repoO", agent_id="agent2")
    runs = memory_list_runs(repo="repoO")
    assert runs[0]["id"] == res2["run_id"]
    assert runs[1]["id"] == res1["run_id"]


# ==========================================
# New Tests - memory_end_run (MCP-50 to MCP-58)
# ==========================================
@pytest.mark.asyncio
async def test_end_run_marks_finished():
    res = memory_begin_run(repo="repo_end", agent_id="agent1")
    run_id = res["run_id"]
    await memory_end_run(run_id=run_id, summary="[]")

    from swarm_memory.server.mcp_server import db

    row = db.execute("SELECT finished_at FROM runs WHERE id = ?", [run_id]).fetchone()
    assert row["finished_at"] is not None


@pytest.mark.asyncio
async def test_end_run_stores_token_counts():
    res = memory_begin_run(repo="repo_end_tokens", agent_id="agent1")
    run_id = res["run_id"]
    await memory_end_run(
        run_id=run_id, summary="[]", input_tokens=100, output_tokens=50, total_cost_usd=0.05
    )

    from swarm_memory.server.mcp_server import db

    row = db.execute(
        "SELECT input_tokens, output_tokens, total_cost_usd FROM runs WHERE id = ?", [run_id]
    ).fetchone()
    assert row["input_tokens"] == 100
    assert row["output_tokens"] == 50
    assert row["total_cost_usd"] == 0.05


@pytest.mark.asyncio
async def test_end_run_empty_summary():
    res = memory_begin_run(repo="repo_end_empty", agent_id="agent1")
    run_id = res["run_id"]
    end_res = await memory_end_run(run_id=run_id, summary="[]")
    assert end_res["status"] == "completed"
    assert end_res["facts_saved"] == 0
    assert end_res["facts_errored"] == 0


@pytest.mark.asyncio
async def test_end_run_valid_summary_creates_facts():
    import json

    res = memory_begin_run(repo="repo_end_valid", agent_id="agent1")
    run_id = res["run_id"]
    summary = json.dumps([{"content": "A test fact", "scope": "repo", "fact_type": "insight"}])
    end_res = await memory_end_run(run_id=run_id, summary=summary)
    assert end_res["status"] == "completed"
    assert end_res["facts_saved"] == 1


@pytest.mark.asyncio
async def test_end_run_partial_failure(monkeypatch):
    import json

    import swarm_memory.server.mcp_server as mcp_module

    res = memory_begin_run(repo="repo_end_partial", agent_id="agent1")
    run_id = res["run_id"]
    summary = json.dumps(
        [
            {"content": "Valid fact", "scope": "repo", "fact_type": "insight"},
            {"content": "Invalid fact", "scope": "repo", "fact_type": "insight"},
        ]
    )

    original_write = mcp_module.writer.write_fact

    async def mock_write(*args, **kwargs):
        if kwargs.get("content") == "Invalid fact":
            raise ValueError("Intentional error")
        return await original_write(*args, **kwargs)

    monkeypatch.setattr(mcp_module.writer, "write_fact", mock_write)

    end_res = await memory_end_run(run_id=run_id, summary=summary)
    assert end_res["status"] == "partial"
    assert end_res["facts_saved"] == 1
    assert end_res["facts_errored"] == 1
    assert "errors" in end_res


@pytest.mark.asyncio
async def test_end_run_malformed_json_returns_error():
    res = memory_begin_run(repo="repo_end_malformed", agent_id="agent1")
    run_id = res["run_id"]
    end_res = await memory_end_run(run_id=run_id, summary="{not json}")
    assert end_res["status"] == "error"
    assert "expected_format" in end_res


@pytest.mark.asyncio
async def test_end_run_ends_session():
    res = memory_begin_run(repo="repo_end_session", agent_id="agent1")
    run_id = res["run_id"]
    from swarm_memory.server.mcp_server import session_manager

    session_manager.mark_seen(run_id, {"fact-1"})
    assert session_manager.get_seen_ids(run_id) != set()

    await memory_end_run(run_id=run_id, summary="[]")
    assert len(session_manager.get_seen_ids(run_id)) == 0


@pytest.mark.asyncio
async def test_end_run_arm2_trajectory(monkeypatch):
    import swarm_memory.server.mcp_server as mcp_module

    res = memory_begin_run(repo="repo_end_arm2", agent_id="agent1", arm="arm2")
    run_id = res["run_id"]

    called = False

    def fake_write_trajectory(traj, rid):
        nonlocal called
        called = True
        return [], []

    monkeypatch.setattr(mcp_module.writer, "write_trajectory", fake_write_trajectory)

    await memory_end_run(run_id=run_id, arm="arm2", summary="[]", trajectory='[{"step": 1}]')
    assert called


@pytest.mark.asyncio
async def test_end_run_arm2_no_trajectory():
    res = memory_begin_run(repo="repo_end_arm2_no_traj", agent_id="agent1", arm="arm2")
    run_id = res["run_id"]
    end_res = await memory_end_run(run_id=run_id, arm="arm2", summary="[]")
    assert end_res["status"] == "completed"


# ==========================================
# New Tests - MCP-60 Server Transport
# ==========================================
def test_mcp_server_sse_starts(monkeypatch):
    import swarm_memory.server.mcp_server as mcp_module

    called_host = None
    called_port = None

    def mock_run(transport, host, port):
        nonlocal called_host, called_port
        called_host = host
        called_port = port

    monkeypatch.setattr(mcp_module.mcp, "run", mock_run)

    import sys

    monkeypatch.setattr(sys, "argv", ["mcp_server.py", "--host", "127.0.0.1", "--port", "8080"])
    mcp_module.main()

    assert called_host == "127.0.0.1"
    assert called_port == 8080
