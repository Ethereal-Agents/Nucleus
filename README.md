# Swarm Memory

Bi-Temporal SQLite Memory Hub for coding agents.

This project implements a shared coding context hub using SQLite (with `sqlite-vec` and `FTS5`) and exposes it via an MCP Server.

## Features
- **Bi-Temporal Design:** Facts are never deleted, only superseded.
- **Hybrid Search:** Dense vector search (via `sqlite-vec`) and BM25 (via `FTS5`).
- **MCP Server:** FastMCP integration for coding agents.
- **Fact Supersession:** Uses an LLM to detect contradictions and update existing facts.
