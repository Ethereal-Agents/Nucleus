"""
swarm_memory/server/mcp.py

FastMCP server for SwarmMemory. Exposes the bi-temporal memory hub tools.
"""

import argparse
import datetime
import logging
import re

from fastmcp import FastMCP

from swarm_memory.core import config
from swarm_memory.core.embeddings import EmbeddingModel
from swarm_memory.core.log import current_arm_id, current_run_id, setup_logging
from swarm_memory.core.models import uuid7
from swarm_memory.ingestion.extraction import parse_extraction_output
from swarm_memory.ingestion.supersession import ConsolidationEngine
from swarm_memory.ingestion.writer import FactWriter
from swarm_memory.retrieval.reader import FactReader
from swarm_memory.server.presentation import format_results_for_agent
from swarm_memory.server.session import SessionManager
from swarm_memory.store.db import get_initialized_db

# Initialize dependencies
log_level = getattr(logging, config.LOG_LEVEL, logging.INFO)
setup_logging(level=log_level)

db = get_initialized_db()
embedder = EmbeddingModel()
engine = ConsolidationEngine()
writer = FactWriter(db=db, embedder=embedder, engine=engine)
reader = FactReader(conn=db, embedder=embedder)
session_manager = SessionManager()

mcp = FastMCP("SwarmMemory")


def _get_arm_for_run(run_id: str) -> str:
    row = db.execute("SELECT arm FROM runs WHERE id = ?", [run_id]).fetchone()
    if not row:
        raise ValueError(f"Run ID '{run_id}' not found. Please call memory_begin_run first.")
    return row["arm"]


@mcp.tool()
async def memory_write(
    content: str,
    scope: str,
    run_id: str,
    valid_from: str | None = None,
    fact_type: str = "insight",
    confidence: float = 1.0,
    supersedes_hint: str | None = None,
) -> dict:
    """
    Write a new architectural decision, gotcha, or insight to the shared memory hub.
    WARNING: This modifies the global shared context for all future agents.

    Use this to persist important knowledge that other agents working on this repo
    should know. Automatically detects and supersedes contradicting older facts using an LLM.

    Parameters:
    - content: The plain-text fact to memorize (e.g., "The auth module uses JWTs, not sessions.")
    - scope: The hierarchical scope this applies to (e.g., "my-repo", "my-repo/src/auth").
      Use narrower scopes for specific module details and root scopes for global conventions.
    - run_id: Your current session ID (obtained from memory_begin_run).
    - fact_type: The category of the fact. Must be one of 'insight', 'gotcha', 'convention', or 'architecture'. Default is 'insight'.
    - confidence: Float from 0.0 to 1.0 indicating your certainty.
    - supersedes_hint: (Optional) If you know this fact explicitly replaces an older fact, pass the old fact's ID here to bypass LLM contradiction detection.
    """
    if not (0.0 <= confidence <= 1.0):
        raise ValueError("Confidence must be between 0.0 and 1.0")
    if not content.strip():
        raise ValueError("Fact content cannot be empty")

    current_run_id.set(run_id)
    arm_id = _get_arm_for_run(run_id)
    current_arm_id.set(arm_id)

    result = await writer.write_fact(
        content=content,
        scope=scope,
        run_id=run_id,
        valid_from=valid_from,
        fact_type=fact_type,
        confidence=confidence,
        supersedes_hint=supersedes_hint,
    )
    return result.model_dump()


@mcp.tool()
def memory_search(
    query: str,
    run_id: str,
    scope: str | None = None,
    as_of: str | None = None,
    top_k: int = 5,
    fact_type: str | None = None,
) -> str:
    """
    Search for relevant facts using hybrid retrieval (semantic + keyword).
    Returns a formatted context block ready for injection into your context.

    Use this tool when you need to understand existing conventions, architecture,
    or past gotchas in a specific part of the codebase.

    Parameters:
    - query: Natural language search query (e.g., "How does authentication work?").
    - run_id: Your current session ID from memory_begin_run. This enables in-session deduplication so you aren't shown the same fact twice.
    - scope: (Optional) The hierarchical scope to restrict the search to (e.g., "my-repo/src/auth"). If omitted, searches globally across the repo.
    - as_of: (Optional) ISO-8601 timestamp for time-travel queries.
    - top_k: Maximum number of results to return (default 5).
    - fact_type: (Optional) Filter by a specific fact type (e.g., "gotcha").
    """
    if not run_id:
        raise ValueError(
            "run_id is required to perform in-session deduplication. Please pass the run_id from memory_begin_run."
        )
    current_run_id.set(run_id)
    arm_id = _get_arm_for_run(run_id)
    current_arm_id.set(arm_id)

    seen_ids = session_manager.get_seen_ids(run_id)
    fetch_k = top_k + len(seen_ids)

    if arm_id == "arm2":
        candidates = reader.search_trajectories(
            query=query,
            top_k=fetch_k,
        )
    else:
        candidates = reader.search(
            query=query,
            scope=scope,
            as_of=as_of,
            top_k=fetch_k,
            fact_type=fact_type,
        )

    fresh = [r for r in candidates if r.fact.id not in seen_ids][:top_k]

    # Only track seen IDs when there is a real session run_id
    if run_id:
        session_manager.mark_seen(run_id, [r.fact.id for r in fresh])

    display_scope = scope if scope else "all scopes"
    return format_results_for_agent(fresh, display_scope)


@mcp.tool()
def memory_invalidate(
    fact_id: str,
    reason: str,
    run_id: str,
    valid_to: str | None = None,
) -> dict:
    """
    Manually invalidate a fact that is no longer true.
    WARNING: This is a destructive action that hides the fact from future searches.

    Use this only when you are absolutely certain a fact is outdated (e.g., a deprecated library was removed).

    Parameters:
    - fact_id: The unique ID of the fact to invalidate (obtained from memory_search results).
    - reason: A brief explanation of why this fact is no longer valid.
    - run_id: Your current session ID (obtained from memory_begin_run).
    - valid_to: (Optional) ISO-8601 timestamp for when the fact became invalid. Defaults to now.
    """
    current_run_id.set(run_id)
    arm_id = _get_arm_for_run(run_id)
    current_arm_id.set(arm_id)
    if not valid_to:
        valid_to = datetime.datetime.now(datetime.UTC).isoformat()

    changes = writer.invalidate_fact(fact_id, valid_to=valid_to)
    db.commit()

    if changes == 0:
        return {"status": "error", "error": f"Fact '{fact_id}' not found or already invalidated."}
    return {"status": "invalidated", "fact_id": fact_id, "reason": reason}


@mcp.tool()
def memory_list_runs(
    run_id: str,
    repo: str | None = None,
    limit: int = 10,
) -> list[dict]:
    """
    List recent agent runs for a repository.

    Use this tool to see what other agents have recently worked on.

    Parameters:
    - run_id: Your current session ID (obtained from memory_begin_run).
    - repo: (Optional) Filter by repository name.
    - limit: Maximum number of runs to return (default 10).
    """
    current_run_id.set(run_id)
    arm_id = _get_arm_for_run(run_id)
    current_arm_id.set(arm_id)
    if repo:
        rows = db.execute(
            "SELECT * FROM runs WHERE repo = ? ORDER BY started_at DESC LIMIT ?", [repo, limit]
        ).fetchall()
    else:
        rows = db.execute("SELECT * FROM runs ORDER BY started_at DESC LIMIT ?", [limit]).fetchall()
    return [dict(row) for row in rows]


@mcp.tool()
def memory_begin_run(
    repo: str,
    agent_id: str,
    arm: str = "arm3",
    branch: str | None = None,
    model: str | None = None,
) -> dict:
    """
    Register the start of an agent run. Returns a run_id to pass to all
    subsequent memory_write and memory_search calls.

    You MUST call this tool once at the beginning of your workflow before using
    other memory tools. It ensures your queries are deduplicated so you don't
    see the same results repeatedly.

    Parameters:
    - repo: The root name of the repository you are working on (e.g., "Nuclues").
    - agent_id: Your unique identifier or role name.
    - arm: The ARM variant identifying the behavior mode (e.g., "arm3"). Defaults to "arm3".
    - branch: (Optional) The specific branch you are working on.
    - model: (Optional) The LLM model name you are using.
    """
    repo = repo.strip()
    if not re.match(r"^[a-zA-Z0-9_.-]+(?:/[a-zA-Z0-9_.-]+)*$", repo):
        raise ValueError(
            f"Invalid repository name provided: '{repo}'. "
            "Expected a valid name like 'org/repo' or 'repo'."
        )

    current_arm_id.set(arm)
    run_id = str(uuid7())
    current_run_id.set(run_id)

    started_at = datetime.datetime.now(datetime.UTC).isoformat()
    created_at = started_at

    db.execute(
        """INSERT INTO runs (id, agent_id, repo, branch, model, started_at, created_at, arm)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        [run_id, agent_id, repo, branch, model, started_at, created_at, arm],
    )
    db.commit()
    return {"run_id": run_id, "status": "started"}


@mcp.tool()
async def memory_end_run(
    run_id: str,
    summary: str = "[]",
    trajectory: str | None = None,
    input_tokens: int = 0,
    output_tokens: int = 0,
    total_cost_usd: float = 0.0,
) -> dict:
    """
    Mark an agent run as complete and extract durable facts from it.

    BEFORE calling this tool, reflect on your completed work and produce a
    JSON array of durable, transferable facts. Pass it as `summary`.

    Extract facts that are structurally true about the FINAL state of the codebase:
    - Architecture decisions, project conventions, hidden gotchas, core dependencies.

    IGNORE: transient debugging steps, syntax errors you fixed, failed attempts.
    CRITICAL: Do NOT include any facts that you already saved manually using `memory_write` during this run.

    Output format for `summary`:
    [
      {
        "content": "Precise fact in 1-2 sentences. Be specific — cite paths/names.",
        "scope": "repo-name/path/to/module",
        "fact_type": "insight|convention|architecture|gotcha|dependency",
        "supersedes_hint": "<id from a fact you saw in memory_search if replacing it, else null>"
      }
    ]

    If you learned nothing durable, pass summary="[]".

    Args:
        run_id:          ID returned by memory_begin_run().
        summary:         JSON array of extracted facts (see format above).
        trajectory:      Optional JSON string of the agent's trajectory (used primarily in arm2).
        input_tokens:    Total input tokens used in this run (for cost tracking).
        output_tokens:   Total output tokens used in this run.
        total_cost_usd:  Total cost of this run in USD.
    """
    current_run_id.set(run_id)
    arm_id = _get_arm_for_run(run_id)
    current_arm_id.set(arm_id)

    row = db.execute("SELECT finished_at FROM runs WHERE id = ?", [run_id]).fetchone()
    if row and row["finished_at"] is not None:
        raise ValueError(f"Run ID '{run_id}' is already finished.")

    import json

    try:
        # We parse it eagerly here to validate the structure using our Pydantic models.
        # This will raise JSONDecodeError or ValueError if fundamentally malformed.
        drafts = parse_extraction_output(summary, run_id)
    except (json.JSONDecodeError, ValueError) as e:
        return {
            "status": "error",
            "error": f"Invalid learnings JSON: {e}",
            "expected_format": [
                {
                    "content": "Precise fact in 1-2 sentences.",
                    "scope": "repo/path",
                    "fact_type": "insight|convention|architecture|gotcha|dependency",
                }
            ],
            "received": summary[:500],
        }

    finished_at = datetime.datetime.now(datetime.UTC).isoformat()
    db.execute(
        """UPDATE runs
           SET summary = ?, input_tokens = ?, output_tokens = ?, total_cost_usd = ?, finished_at = ?
           WHERE id = ?""",
        [summary, input_tokens, output_tokens, total_cost_usd, finished_at, run_id],
    )
    db.commit()

    saved = []
    errors = []

    if arm_id == "arm2" and trajectory:
        saved, errors = writer.write_trajectory(trajectory, run_id)
    else:
        # drafts is already computed at the top of the function
        for draft in drafts:
            try:
                result = await writer.write_fact(
                    content=draft.content,
                    scope=draft.scope,
                    run_id=run_id,
                    fact_type=draft.fact_type,
                    supersedes_hint=draft.supersedes_hint,
                )
                if result.fact_ids:
                    saved.append(
                        {
                            "content": draft.content[:80],
                            "fact_ids": result.fact_ids,
                            "status": result.status,
                        }
                    )
                else:
                    errors.append(
                        {"content": draft.content[:80], "error": "write returned no fact_ids"}
                    )
            except Exception as e:
                errors.append({"content": draft.content[:80], "error": str(e)})

    session_manager.end_session(run_id)

    response = {
        "run_id": run_id,
        "status": "completed" if not errors else "partial",
        "facts_saved": len(saved),
        "facts_errored": len(errors),
    }
    if errors:
        response["errors"] = errors
    return response


def main():
    parser = argparse.ArgumentParser(description="Start SwarmMemory MCP Server")
    parser.add_argument("--host", default="0.0.0.0", help="SSE host binding")
    parser.add_argument("--port", type=int, default=8000, help="SSE port binding")
    args = parser.parse_args()

    print(f"Starting SwarmMemory MCP Server on SSE http://{args.host}:{args.port}/sse")
    mcp.run(transport="sse", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
