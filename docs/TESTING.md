# SwarmMemory Testing Documentation

This document outlines the testing strategy, frameworks, and specific test suites used to ensure the reliability and correctness of the SwarmMemory ingestion pipeline, consolidation engine, and Model Context Protocol (MCP) server.

## Overview

SwarmMemory employs a multi-tiered testing strategy comprising unit tests, integration tests, and comprehensive end-to-end (E2E) scale tests. 

- **Framework**: `pytest` with `pytest-asyncio` for asynchronous tests.
- **In-Memory Database**: We use an in-memory SQLite database (`:memory:`) coupled with the `sqlite-vec` extension for high-performance vector search in isolation.
- **Mocks & Stubbing**: 
  - Standard integration tests heavily utilize `unittest.mock.AsyncMock` to isolate the `ConsolidationEngine` (LLM interactions) and `EmbeddingModel` (SentenceTransformers) for deterministic, fast execution.
  - End-to-end tests run against live infrastructure (e.g., real LLM endpoints via OpenRouter and real embedding generation).

## Test Suites

### 1. `test_e2e_mcp.py`
This test suite verifies the end-to-end functionality of the MCP server's exposed tools. It focuses on the behavioral correctness of the API endpoints, ensuring state is tracked properly per agent run.

**Key Scenarios Tested:**
- **Lifecycle Management**: `memory_begin_run`, `memory_list_runs`, and contextual bounding.
- **Cross-Scope Visibility**: Validating hierarchical scope restrictions and scope filtering.
- **Search Capabilities**: Validating FTS (Full-Text Search), hybrid search fusion, semantic threshold enforcement, and session deduplication (`run_id` state tracking to prevent returning the same fact twice in one run).
- **Invalidation**: Hiding superseded facts from search results and managing the `valid_to` lifecycle.

*Note: In standard CI, the LLM engine in this suite is stubbed to return the provided text as-is (`mock_consolidate`), allowing deterministic testing of the pipeline and database logic.*

### 2. `test_integration.py`
This suite dives deeper into the integration between core pipeline components (e.g., `FactWriter`, `ConsolidationEngine`, and `sqlite-vec`).

**Key Scenarios Tested:**
- **Multi-Scope Hierarchy**: Writing across boundaries and verifying isolation.
- **Supersession Chains**: Ensuring that A -> B -> C supersession chains are correctly maintained and that only the latest fact (C) is returned by search.
- **Concurrent Writes**: Ensuring the `sqlite-vec` transaction boundaries and unique constraints (e.g., `content_hash`) hold up under simultaneous ingestion tasks.

### 3. `test_e2e_scale.py` (The E2E Scale Stress Test)
This is the most critical test file for validating the integrity of the semantic consolidation logic. It disables mocks and executes against:
- A real `sqlite-vec` instance.
- The real `EmbeddingModel` (`nomic-embed-text-v1.5`).
- A real LLM service (`gpt-4o-mini` via OpenRouter).

The scale test sequentially executes 5 distinct phases to simulate a rigorous production workload of 100+ entries:

#### Phase 1: Base Insertion
Inserts foundational architecture facts across various subscopes (`arch/overview`, `arch/db`, `arch/auth`, etc.).
- **Assertion**: Facts are created cleanly with `status="created"`.

#### Phase 2: Idempotency & Deduplication
Re-inserts the exact same knowledge chunks from Phase 1.
- **Assertion**: The system's hashing fallback intercepts these immediately, bypassing the LLM and vector search, returning `status="duplicate"`.

#### Phase 3: Independent Facts
Inserts 30 completely independent, granular concepts (e.g., UI variants) into the same scope.
- **Assertion**: The LLM engine evaluates them against the base facts, correctly recognizes them as unrelated, and returns them as non-superseding (`created` or `consolidated`). Crucially, no base facts are overwritten.

#### Phase 4: Refinement (Supersession)
Inserts updated insights that build upon the foundational facts (e.g., "JWT tokens expire in 15 minutes").
- **Assertion**: The LLM detects the semantic overlap, merges the contexts, and correctly flags the base fact to be superseded. The test verifies that the `fact_lineage` table correctly records the `predecessor_id` (old fact) mapping to the `successor_id` (new fact), and the status is returned as `"consolidated"` or `"split"`.

#### Phase 5: Auto-Splitting (Threshold Overrides)
Injects a massive paragraph that exceeds the static token threshold (`FACT_TOKEN_THRESHOLD`).
- **Assertion**: The engine detects the violation and triggers the auto-split fallback. The LLM breaks the bloated context into multiple semantically rich, atomic facts, and returns `status="split"`.

#### Phase 6: Comprehensive Read Flow
Executes complex search queries against the live, populated database from Phases 1-5 to validate hybrid search capabilities, relevance ranking, and contextual bounding.
- **Assertion 1 (Robust Hybrid Search)**: Queries must validate both pathways of the fusion engine:
  - *Dense/Semantic Pathway*: Queries using vague terminology retrieve correct facts purely through vector similarities, proving the embedding engine works.
  - *Sparse/Keyword Pathway*: Queries using specific jargon or exact variable names retrieve the exact facts via Full-Text Search (FTS), proving exact matches aren't lost in semantic noise.
- **Assertion 2 (Scope Isolation)**: Queries constrained to a specific scope (e.g., `arch/db`) absolutely do not leak independent facts from sibling scopes (e.g., `arch/auth`), validating strict hierarchical boundary enforcement.
- **Assertion 3 (Time-Travel / As Of)**: Queries with an `as_of` timestamp successfully retrieve historical, superseded facts that were valid at that specific point in time, whilst correctly excluding newer facts that had not yet been ingested.

## Running the Tests

To run the full suite (fast, mocked):
```bash
uv run pytest tests/
```

To run the live scale test (requires API keys, takes ~1-2 minutes):
```bash
OPENROUTER_API_KEY=your_key uv run pytest tests/server/test_e2e_scale.py -v -s
```

## Continuous Integration Notes
- When updating schemas (especially `fact_lineage`), ensure that both the `e2e_reset` mocks and the real `ConsolidationResult` outputs remain structurally equivalent to prevent false positives in mocked tests.
- Because `memory_search` handles in-session deduplication natively, if a test asserts that a fact appears in two consecutive searches under the *same* `run_id`, the second search will intentionally omit the fact. Use a fresh `run_id` for independent search assertions.
