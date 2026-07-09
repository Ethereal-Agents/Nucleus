import os
from datetime import UTC, datetime, timedelta

import pytest

# Set up in-memory DB for tests
os.environ["SWARM_MEMORY_DB_PATH"] = ":memory:"

from unittest.mock import AsyncMock, MagicMock

import swarm_memory.server.mcp_server as mcp_module
from swarm_memory.core.embeddings import EmbeddingModel
from swarm_memory.ingestion.supersession import ContradictionDetector
from swarm_memory.ingestion.writer import FactWriter
from swarm_memory.retrieval.reader import FactReader
from swarm_memory.server.mcp_server import (
    memory_begin_run,
    memory_search,
    memory_write,
)
from swarm_memory.server.session import SessionManager
from swarm_memory.store.db import get_initialized_db


@pytest.fixture(autouse=True)
def reset_mcp_state():
    """Reinitialize all module-level singletons to ensure clean state per test."""
    mcp_module.db = get_initialized_db(":memory:")

    # Mock Embedder to return zeros
    mcp_module.embedder = MagicMock(spec=EmbeddingModel)
    mcp_module.embedder.embed.return_value = b"\x00" * (768 * 4)
    mcp_module.embedder.embed_query.return_value = b"\x00" * (768 * 4)
    mcp_module.embedder.dim = 768

    # Mock Detector to return empty relationships by default
    mcp_module.detector = MagicMock(spec=ContradictionDetector)
    mcp_module.detector.detect_contradictions = AsyncMock(side_effect=lambda *args, **kwargs: [])

    mcp_module.writer = FactWriter(
        db=mcp_module.db, embedder=mcp_module.embedder, detector=mcp_module.detector
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
    fact1_id = write_res1["fact_id"]

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
    assert write_res2["status"] == "created"
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
