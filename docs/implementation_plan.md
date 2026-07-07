# §13 — End-of-Run Fact Extraction: Final Implementation Plan

## Architecture: Docstring-Driven Extraction

The extraction instruction lives in the `memory_end_run` **tool docstring**.
MCP clients (Claude Code, Antigravity, etc.) automatically inject all tool
descriptions into the agent's context. This means every agent using this MCP
server gets the extraction instruction for free — no system prompt editing,
no per-agent wiring.

> [!NOTE]
> **Future upgrade path — MCP Sampling**: The MCP spec includes a
> `sampling/createMessage` capability where the server can ask the client to
> run an LLM inference on its behalf (using the agent's own model + warm KV cache).
> This would let the server own the extraction prompt 100% while still leveraging
> the agent's cache. Worth adding post-V0 once we confirm which clients support it.
> For now the docstring approach works universally.

---

## How the Full Flow Works

```
1. Agent works on its task...
        └─ Calls memory_search() on demand → gets formatted ⚠ context blocks.
           (This is where the agent sees existing fact IDs, e.g. "→ id: fact_abc123").

2. Agent finishes. Sees memory_end_run docstring instruction. Reflects on its work.
        └─ Calls memory_end_run(run_id=..., summary='[{"content": "...", ...}]')
                                                       ↑
                                       Agent-produced JSON fact extraction. 
                                       Uses IDs seen in step 1 for supersedes_hint.

3. Server: memory_end_run handler
        ├─ parse_extraction_output(summary, run_id)  → list[FactDraft]
        ├─ For each draft: writer.write_fact(..., supersedes_hint=draft.supersedes_hint)
        ├─ finalize_run(run_id, tokens, cost)
        └─ end_session(run_id)   ← clears Phase 2 session dedup state
```

---

## New Files

| File | What it does |
|---|---|
| `swarm_memory/ingestion/extraction.py` | `FactDraft`, `parse_extraction_output` |
| `tests/ingestion/test_extraction.py` | Tests for the parser |

---

## `extraction.py` — Full Spec

### 1. `FactDraft` model

```python
from pydantic import BaseModel
from swarm_memory.core.models import FactType

class FactDraft(BaseModel):
    content: str
    scope: str
    fact_type: FactType = FactType.INSIGHT
    supersedes_hint: str | None = None  # fact.id the agent thinks this replaces
    # NOTE: valid_from is NOT in the output format — set server-side at parse time.
    # LLMs cannot reliably produce precise ISO-8601 timestamps. We use
    # datetime.now(UTC).isoformat() at the moment memory_end_run is called.
```

> [!IMPORTANT]
> `valid_from` is deliberately excluded from the agent's output format.
> It is set by the server at the moment `memory_end_run` is called.
> `supersedes_hint` is a hint only — `ContradictionDetector` makes the final call.

---

### 2. `parse_extraction_output(raw, run_id)` → `list[FactDraft]`

```python
import json
import re
import logging
from typing import Any
from swarm_memory.ingestion.extraction import FactDraft

logger = logging.getLogger(__name__)

def parse_extraction_output(raw: str, run_id: str) -> list[FactDraft]:
    """
    Robustly parse agent's summary string into FactDraft objects.

    Three-stage pipeline:
      1. Strip markdown fences (agents occasionally wrap output in ```json)
      2. Try json.loads() directly
      3. On failure: repair trailing commas with regex, try again
      4. On second failure: log warning, return []

    Then validates each item individually with Pydantic — a single
    malformed item is skipped; the rest are kept.

    Never raises — memory_end_run must always complete successfully
    even if the agent's extraction output is garbage.
    """
```

---

## `memory_end_run` Docstring (extraction instruction)

This is how the agent learns to extract facts — automatically, via the MCP tool description.
**Person A will add this in Phase 4**:

```python
@mcp.tool
async def memory_end_run(
    run_id: str,
    summary: str = "[]",
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
        input_tokens:    Total input tokens used in this run (for cost tracking).
        output_tokens:   Total output tokens used in this run.
        total_cost_usd:  Total cost of this run in USD.
    """
```

---

## Flag for Person A — `write_fact()` needs `supersedes_hint`

Person A's `write_fact()` currently does not accept `supersedes_hint`.
It needs one new optional parameter:

```python
async def write_fact(
    self,
    content: str,
    scope: str,
    run_id: str,
    valid_from: str | None = None,
    fact_type: str = "insight",
    confidence: float = 1.0,
    supersedes_hint: str | None = None,   # ← new
) -> WriteResult:
    """
    If supersedes_hint is set, skip the KNN candidate search and go directly
    to the contradiction check for that specific fact pair. This short-circuits
    ~80% of LLM contradiction checks for runs where the agent correctly identifies
    what it is replacing.
    """
```

---

## Test Plan (`test_extraction.py`)

All tests run with zero LLM calls and no internet access.

| Test | Covers |
|---|---|
| `test_parse_clean_json` | Clean JSON → correct `FactDraft` list |
| `test_parse_fenced_json` | Markdown fences stripped |
| `test_parse_trailing_comma` | Trailing commas repaired |
| `test_parse_garbage_returns_empty` | Unrecoverable input → `[]`, never raises |
| `test_parse_empty_string` | `""` → `[]` |
| `test_parse_empty_array` | `"[]"` → `[]` |
| `test_parse_partial_valid` | Malformed items skipped, valid ones kept |
| `test_valid_from_not_in_output` | `FactDraft` has no `valid_from` field |
| `test_supersedes_hint_preserved` | Hint from agent preserved in `FactDraft` |
| `test_fact_type_defaults_to_insight` | Missing `fact_type` → `"insight"` |
