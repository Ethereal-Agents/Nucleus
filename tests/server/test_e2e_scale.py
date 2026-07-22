import asyncio
import os
from datetime import UTC, datetime

import pytest

# Test data
BASE_KNOWLEDGE = [
    ("The system architecture is based on microservices.", "arch/overview"),
    ("Database is PostgreSQL for relational data.", "arch/db"),
    ("Redis is used for caching and rate limiting.", "arch/db"),
    ("Auth service handles JWT generation and validation.", "arch/auth"),
    ("User service manages profiles and settings.", "arch/user"),
    ("Frontend is a React SPA.", "arch/frontend"),
    ("Deployment is done via GitHub Actions.", "arch/deploy"),
    ("Infrastructure is hosted on AWS.", "arch/infra"),
    ("Logs are shipped to Datadog.", "arch/observability"),
    ("Metrics are collected by Prometheus.", "arch/observability"),
]

INDEPENDENT_FACTS = [
    (f"UI component Button has variant {i}.", "arch/frontend/ui") for i in range(30)
]

REFINEMENT_FACTS = [
    (
        "The system architecture is based on microservices, using gRPC for internal communication.",
        "arch/overview",
    ),
    ("Database is PostgreSQL for relational data, with read replicas for scaling.", "arch/db"),
    (
        "Auth service handles JWT generation and validation; tokens expire in 15 minutes.",
        "arch/auth",
    ),
    ("Frontend is a React SPA using Vite for building.", "arch/frontend"),
    ("Deployment is done via GitHub Actions with staging and prod environments.", "arch/deploy"),
]


@pytest.mark.asyncio
async def test_e2e_scale_ingestion():
    if not os.getenv("OPENROUTER_API_KEY"):
        pytest.skip("No API key for LLM scale test")

    try:
        import sqlite_vec  # noqa: F401
    except ImportError:
        pytest.skip("sqlite-vec is not available")

    import swarm_memory.server.mcp_server as mcp_module
    from swarm_memory.core.embeddings import EmbeddingModel
    from swarm_memory.ingestion.supersession import ConsolidationEngine
    from swarm_memory.ingestion.writer import FactWriter
    from swarm_memory.store.db import get_initialized_db

    db = get_initialized_db(":memory:")

    embedder = EmbeddingModel()
    engine = ConsolidationEngine()
    engine.llm_service.model_name = "openrouter/openai/gpt-4o-mini"

    writer = FactWriter(db=db, embedder=embedder, engine=engine)
    from swarm_memory.retrieval.reader import FactReader

    reader = FactReader(conn=db, embedder=embedder)

    # Override mcp_server modules with our real ones
    mcp_module.db = db
    mcp_module.embedder = embedder
    mcp_module.engine = engine
    mcp_module.writer = writer
    mcp_module.reader = reader

    run_id = mcp_module.memory_begin_run(repo="scale_repo", agent_id="scale_agent")["run_id"]

    # Phase 1: Base Insertion
    print("\\n[Phase 1] Inserting Base Knowledge...")
    base_fact_ids = []
    for content, subscope in BASE_KNOWLEDGE:
        res = await mcp_module.memory_write(
            content=content, scope=f"scale_repo/{subscope}", run_id=run_id
        )
        assert res["status"] == "created"
        base_fact_ids.append(res["fact_ids"][0])

    assert len(base_fact_ids) == 10

    # Phase 2: Duplicates
    print("[Phase 2] Inserting Duplicates...")
    duplicate_results = []
    for content, subscope in BASE_KNOWLEDGE:
        # Same exact content
        res = await mcp_module.memory_write(
            content=content, scope=f"scale_repo/{subscope}", run_id=run_id
        )
        assert res["status"] == "duplicate"
        duplicate_results.append(res["fact_ids"][0])

    assert duplicate_results == base_fact_ids

    # Phase 3: Independence
    print("[Phase 3] Inserting Independent Facts...")
    independent_ids = []
    for content, subscope in INDEPENDENT_FACTS:
        res = await mcp_module.memory_write(
            content=content, scope=f"scale_repo/{subscope}", run_id=run_id
        )
        assert res["status"] in ["created", "consolidated", "split"]

        # Verify it did not supersede any base facts
        superseded = res.get("superseded_ids", [])
        for sid in superseded:
            assert sid not in base_fact_ids, f"Independent fact superseded base fact {sid}!"

        independent_ids.extend(res["fact_ids"])

    assert len(independent_ids) > 0

    # We need a small sleep to ensure deterministic timestamp boundaries for as_of queries
    await asyncio.sleep(1)
    t_before_refinements = datetime.now(UTC).isoformat()

    # Phase 4: Refinements (Supersession)
    print("[Phase 4] Inserting Refinements...")
    refined_ids = []
    for content, subscope in REFINEMENT_FACTS:
        res = await mcp_module.memory_write(
            content=content, scope=f"scale_repo/{subscope}", run_id=run_id
        )
        assert res["status"] in ["consolidated", "split"]
        # Since it supersedes, superseded_ids should not be empty
        assert len(res.get("superseded_ids", [])) > 0
        refined_ids.append(res["fact_ids"][0])

    # Verify Lineage Table
    for new_id in refined_ids:
        row = db.execute(
            "SELECT predecessor_id FROM fact_lineage WHERE successor_id = ?", [new_id]
        ).fetchone()
        assert row is not None, f"No lineage found for successor {new_id}"
        pred_id = row[0]
        # Verify the predecessor is indeed one of the base facts
        assert pred_id in base_fact_ids

    # Phase 5: Auto-Splitting (Threshold Override)
    print("[Phase 5] Auto-Splitting...")
    # Temporarily drop threshold limit to force split
    import swarm_memory.core.config as config

    original_threshold = config.FACT_WORD_THRESHOLD
    config.FACT_WORD_THRESHOLD = 5  # very low — forces split on any multi-sentence content

    huge_content = (
        "The authentication system uses a dual-token approach. First, short-lived JWT access tokens "
        "are issued with a 15-minute expiration time. Second, long-lived refresh tokens are stored "
        "securely in an HTTP-only cookie to prevent XSS attacks. If an access token expires, the "
        "client can use the refresh token to obtain a new pair without requiring user interaction."
    )

    res = await mcp_module.memory_write(
        content=huge_content, scope="scale_repo/arch/auth", run_id=run_id
    )

    assert res["status"] == "split"
    # It should have returned multiple fact_ids because of splitting!
    assert len(res["fact_ids"]) > 1, f"Expected multiple split facts, got {len(res['fact_ids'])}"

    # Phase 6: Read Verification
    print("[Phase 6] Read Flow Verification...")
    run_id_6 = mcp_module.memory_begin_run(repo="scale_repo", agent_id="scale_agent")["run_id"]

    # 6.1 Robust Hybrid Search
    # Sparse / Keyword Pathway: Exact match for a highly specific term
    read_res_sparse = mcp_module.memory_search(
        query="read replicas", run_id=run_id_6, scope="scale_repo/arch/db", top_k=2
    )
    assert "read replicas" in read_res_sparse, f"Sparse keyword search failed: {read_res_sparse}"

    # Dense / Semantic Pathway: Conceptual query with no keyword overlap
    # Fact from Phase 1: "Infrastructure is hosted on AWS."
    read_res_dense = mcp_module.memory_search(
        query="Where do we deploy our cloud servers?",
        run_id=run_id_6,
        scope="scale_repo/arch/infra",
        top_k=2,
    )
    assert "AWS" in read_res_dense, f"Dense semantic search failed: {read_res_dense}"

    # 6.2 Scope Filtering
    # Search globally for "React SPA"
    read_res_global = mcp_module.memory_search(
        query="frontend SPA", run_id=run_id_6, scope="scale_repo", top_k=1
    )
    assert "React" in read_res_global

    # Search in isolated scope that should NOT have it
    read_res_isolated = mcp_module.memory_search(
        query="frontend SPA", run_id=run_id_6, scope="scale_repo/arch/db", top_k=5
    )
    assert "React" not in read_res_isolated

    # 6.3 Temporal Travel (as_of)
    # Re-use run_id_new to avoid deduplication
    run_id_time = mcp_module.memory_begin_run(repo="scale_repo", agent_id="scale_agent")["run_id"]

    # We superseded Base facts in Phase 4.
    # Let's search as_of t_before_refinements -> Should see PostgreSQL but NOT read replicas
    old_res = mcp_module.memory_search(
        query="read replicas",
        run_id=run_id_time,
        scope="scale_repo/arch/db",
        as_of=t_before_refinements,
    )
    assert "read replicas" not in old_res

    old_res2 = mcp_module.memory_search(
        query="PostgreSQL",
        run_id=run_id_time,
        scope="scale_repo/arch/db",
        as_of=t_before_refinements,
    )
    assert "read replicas" not in old_res2
    assert "PostgreSQL" in old_res2

    # 6.4 Session Deduplication
    # We already searched for "React" under run_id_6.
    # If we search for it again under the SAME run_id_6, it should be deduplicated (hidden).
    read_res_dedup = mcp_module.memory_search(
        query="frontend SPA", run_id=run_id_6, scope="scale_repo"
    )
    assert "React" not in read_res_dedup, "Session deduplication failed!"

    # Clean up
    config.FACT_WORD_THRESHOLD = original_threshold
    db.close()
    print("Scale Test Passed Successfully!")
