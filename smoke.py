import asyncio
import os
import sys

# Ensure we use a test database for the smoke test
os.environ["SWARM_MEMORY_DB_PATH"] = "smoke_test.db"

from swarm_memory.server.mcp import (
    memory_begin_run,
    memory_search,
    memory_write,
)


async def main():
    print("--- SwarmMemory Smoke Test ---")

    # 1. Begin Run
    print("1. Beginning run...")
    begin_res = memory_begin_run(repo="smoke_test_repo", agent_id="smoke_agent")
    run_id = begin_res["run_id"]
    print(f"Run ID: {run_id}")

    # 2. Write 5 Facts
    print("\n2. Writing 5 facts...")
    facts = [
        "The authentication system uses JWT tokens for stateless sessions.",
        "The database is SQLite configured with WAL mode.",
        "All API endpoints require a valid Bearer token.",
        "The frontend is built using React and TypeScript.",
        "Rate limiting is set to 100 requests per minute.",
    ]

    fact_ids = []
    for i, content in enumerate(facts):
        res = await memory_write(
            content=content, scope="architecture", run_id=run_id, fact_type="insight"
        )
        fact_ids.append(res["fact_id"])
        print(f"Fact {i + 1} written: {res['fact_id']}")

    # 3. Supersede 1 Fact
    print("\n3. Superseding Fact 1 (JWT -> Redis Sessions)...")
    supersede_res = await memory_write(
        content="The authentication system now uses stateful Redis sessions instead of JWT.",
        scope="architecture",
        run_id=run_id,
        fact_type="insight",
        supersedes_hint=fact_ids[0],  # Use hint to bypass LLM requirement for smoke test
    )
    print(f"New Fact written: {supersede_res['fact_id']}")
    print(f"Superseded IDs: {supersede_res['superseded_ids']}")

    if fact_ids[0] not in supersede_res["superseded_ids"]:
        print("ERROR: Fact 1 was not superseded!")
        sys.exit(1)

    # 4. Search & Verify
    print("\n4. Searching for 'authentication'...")
    # Using a different run_id for search so we don't hit the session dedup filter
    search_res = memory_search(
        query="authentication", scope="architecture", run_id="smoke_search_run", top_k=5
    )

    print("\nSearch Results:")
    print("-" * 40)
    print(search_res)
    print("-" * 40)

    # Verify
    if "Redis sessions" in search_res and "JWT tokens" not in search_res:
        print(
            "\n✅ Verification SUCCESS: Found the new Redis session fact and NOT the superseded JWT fact."
        )
    else:
        print("\n❌ Verification FAILED.")
        sys.exit(1)

    # Cleanup
    if os.path.exists("smoke_test.db"):
        os.remove("smoke_test.db")
    print("\nSmoke test complete. Cleaned up smoke_test.db.")


if __name__ == "__main__":
    asyncio.run(main())
