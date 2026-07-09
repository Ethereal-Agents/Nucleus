import os

import pytest

# Must set before importing mcp so that db initializes in memory
os.environ["SWARM_MEMORY_DB_PATH"] = ":memory:"

from swarm_memory.server.mcp import (
    memory_begin_run,
    memory_end_run,
    memory_invalidate,
    memory_list_runs,
    memory_search,
    memory_write,
)


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
    end_res = await memory_end_run(run_id=run_id, summary="Completed auth testing")
    assert end_res["status"] == "completed"

    # 9. List runs
    runs = memory_list_runs(repo="testrepo")
    assert len(runs) >= 1
    assert runs[0]["id"] == run_id
    assert runs[0]["summary"] == "Completed auth testing"


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

    # Should not raise an exception
    end_res = await memory_end_run(run_id=run_id, summary=malformed_summary)
    assert end_res["status"] == "completed"

    runs = memory_list_runs(repo="testrepo", limit=1)
    assert runs[0]["id"] == run_id
    assert runs[0]["summary"] == malformed_summary
