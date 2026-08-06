# Swarm Memory

Bi-Temporal SQLite Memory Hub for coding agents.

This project implements a shared coding context hub using SQLite (with `sqlite-vec` and `FTS5`) and exposes it via an MCP Server.

## Features
- **Bi-Temporal Design:** Facts are never deleted, only superseded.
- **Hybrid Search:** Dense vector search (via `sqlite-vec`) and BM25 (via `FTS5`).
- **MCP Server:** FastMCP integration for coding agents.
- **Fact Supersession:** Uses an LLM to detect contradictions and update existing facts.

## Architecture
- **Data Store**: Uses SQLite enhanced with `sqlite-vec` for vector embeddings and `FTS5` for full-text search (BM25), enabling robust hybrid search capabilities.
- **Bi-Temporal Design**: Maintains a complete history of knowledge. Facts are never deleted; when a contradiction is found, the older fact is marked as superseded, maintaining a clear `Fact Lineage`.
- **MCP Server**: Built with FastMCP, it exposes a set of tools (`memory_begin_run`, `memory_search`, `memory_write`, `memory_end_run`, etc.) to client agents, enabling them to seamlessly read and write shared context.
- **End-of-Run Fact Extraction**: Utilizes a docstring-driven extraction approach where the MCP client's LLM automatically extracts durable, transferable facts (insights, conventions, architecture decisions, gotchas, dependencies) at the end of an agent run.
- **Consolidation Engine**: An LLM-powered engine that evaluates new facts against existing knowledge to detect contradictions, short-circuiting with `supersedes_hint` when agents explicitly replace older facts.
