import asyncio
import os
import subprocess
import time
import sys
import json
from contextlib import asynccontextmanager

from mcp.client.session import ClientSession
from mcp.client.sse import sse_client

TEST_DB_PATH = "e2e_test.db"

@asynccontextmanager
async def mcp_sse_session():
    if os.path.exists(TEST_DB_PATH):
        os.remove(TEST_DB_PATH)
        
    env = os.environ.copy()
    env["SWARM_MEMORY_DB_PATH"] = TEST_DB_PATH
    
    print("Starting MCP Server on SSE...")
    process = subprocess.Popen(
        [sys.executable, "-m", "swarm_memory.server.mcp_server", "--host", "127.0.0.1", "--port", "8080"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    time.sleep(5) # Wait for server to fully initialize
    
    url = "http://127.0.0.1:8080/sse"
    
    try:
        async with sse_client(url) as streams:
            async with ClientSession(*streams) as session:
                await session.initialize()
                yield session
    finally:
        print("\nShutting down server...")
        process.terminate()
        process.wait()
        
        if os.path.exists(TEST_DB_PATH):
            os.remove(TEST_DB_PATH)


async def call_tool(session: ClientSession, name: str, args: dict):
    result = await session.call_tool(name, arguments=args)
    if result.isError:
        raise RuntimeError(f"Tool error: {result.content[0].text if result.content else result}")
    if not result.content:
        return None
    try:
        return json.loads(result.content[0].text)
    except json.JSONDecodeError:
        return result.content[0].text


async def run_tests():
    async with mcp_sse_session() as session:
        print("\n================= E2E TEST REPORT =================")
        passed = 0
        failed = 0
        
        def check(condition, desc):
            nonlocal passed, failed
            if condition:
                print(f"✅ PASS: {desc}")
                passed += 1
            else:
                print(f"❌ FAIL: {desc}")
                failed += 1

        try:
            # TC-1.1 Begin run
            res = await call_tool(session, "memory_begin_run", {"repo": "Nuclues", "agent_id": "test-agent-1"})
            run_id = res.get("run_id")
            check(res.get("status") == "started" and run_id, "TC-1.1: memory_begin_run happy path")

            # TC-2.1 Write fact
            write_res = await call_tool(session, "memory_write", {
                "content": "The auth module uses JWT tokens stored in HTTP-only cookies.",
                "scope": "Nuclues/src/auth",
                "run_id": run_id
            })
            fact_id = write_res.get("fact_id")
            check(write_res.get("status") == "created" and fact_id, "TC-2.1: memory_write happy path")

            # TC-2.4 Duplicate write
            dup_res = await call_tool(session, "memory_write", {
                "content": "The auth module uses JWT tokens stored in HTTP-only cookies.",
                "scope": "Nuclues/src/auth",
                "run_id": run_id
            })
            check(dup_res.get("status") == "duplicate" and dup_res.get("fact_id") == fact_id, "TC-2.4: memory_write idempotency")

            # TC-2.8 Invalid date format
            try:
                await call_tool(session, "memory_write", {
                    "content": "Some fact.",
                    "scope": "Nuclues",
                    "run_id": run_id,
                    "valid_from": "not-a-date"
                })
                check(False, "TC-2.8: Invalid valid_from should fail")
            except RuntimeError as e:
                check("isoformat" in str(e).lower(), "TC-2.8: Invalid valid_from format rejected")

            # TC-2.9 Invalid fact type
            try:
                await call_tool(session, "memory_write", {
                    "content": "Some fact.",
                    "scope": "Nuclues",
                    "run_id": run_id,
                    "fact_type": "invalid_type"
                })
                check(False, "TC-2.9: Invalid fact_type should fail")
            except RuntimeError as e:
                check("invalid_type" in str(e) or "FactType" in str(e), "TC-2.9: Invalid fact_type rejected")
            
            # TC-2.10 Invalid confidence range (Newly added validation)
            try:
                await call_tool(session, "memory_write", {
                    "content": "Some fact.",
                    "scope": "Nuclues",
                    "run_id": run_id,
                    "confidence": 1.5
                })
                check(False, "TC-2.10: Out of range confidence should fail")
            except RuntimeError as e:
                check("Confidence must be between 0.0 and 1.0" in str(e), "TC-2.10: Confidence range validated")
                
            # TC-2.11 Empty content (Newly added validation)
            try:
                await call_tool(session, "memory_write", {
                    "content": "   \n  ",
                    "scope": "Nuclues",
                    "run_id": run_id
                })
                check(False, "TC-2.11: Empty content should fail")
            except RuntimeError as e:
                check("Fact content cannot be empty" in str(e), "TC-2.11: Empty content validated")

            # TC-3.1 Search
            res_search_run = await call_tool(session, "memory_begin_run", {"repo": "Nuclues", "agent_id": "search-agent"})
            search_run_id = res_search_run["run_id"]
            
            search_res = await session.call_tool("memory_search", arguments={
                "query": "authentication",
                "run_id": search_run_id
            })
            search_text = search_res.content[0].text if search_res.content else ""
            check("JWT tokens" in search_text, "TC-3.1: memory_search returns facts")
            
            # TC-3.5 Dedup
            search_res2 = await session.call_tool("memory_search", arguments={
                "query": "authentication",
                "run_id": search_run_id
            })
            search_text2 = search_res2.content[0].text if search_res2.content else ""
            check("JWT tokens" not in search_text2, "TC-3.5: Session dedup works")

            # TC-4.1 Invalidate
            inv_res = await call_tool(session, "memory_invalidate", {
                "fact_id": fact_id,
                "reason": "Because we changed it.",
                "run_id": run_id
            })
            check(inv_res.get("status") == "invalidated", "TC-4.1: memory_invalidate happy path")
            
            # TC-4.3 Double invalidate
            inv_res2 = await call_tool(session, "memory_invalidate", {
                "fact_id": fact_id,
                "reason": "Double tap.",
                "run_id": run_id
            })
            check(inv_res2.get("status") == "error", "TC-4.3: Double invalidate fails gracefully")

            # TC-5.1 End run
            end_res = await call_tool(session, "memory_end_run", {
                "run_id": run_id,
                "summary": "[]"
            })
            check(end_res.get("status") == "completed", "TC-5.1: memory_end_run happy path")

            # TC-5.7 Double end run (Newly added validation)
            try:
                await call_tool(session, "memory_end_run", {
                    "run_id": run_id,
                    "summary": "[]"
                })
                check(False, "TC-5.7: Double end run should fail")
            except RuntimeError as e:
                check("already finished" in str(e), "TC-5.7: Double memory_end_run rejected")

            # TC-8.1 Invalid run_id (Newly added validation)
            try:
                await call_tool(session, "memory_search", {
                    "query": "test",
                    "run_id": "nonexistent-run-id"
                })
                check(False, "TC-8.1: Nonexistent run_id should fail")
            except RuntimeError as e:
                check("not found" in str(e), "TC-8.1: run_id existence validated")
                
            # TC-6.1 List runs
            list_res = await call_tool(session, "memory_list_runs", {
                "run_id": search_run_id
            })
            check(len(list_res) >= 2, "TC-6.1: memory_list_runs happy path")

        except Exception as e:
            print(f"❌ TEST SCRIPT ERROR: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

        print("===================================================")
        print(f"Total passed: {passed}")
        print(f"Total failed: {failed}")

if __name__ == "__main__":
    asyncio.run(run_tests())
