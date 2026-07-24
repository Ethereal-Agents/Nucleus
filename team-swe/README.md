# Team-SWE Benchmark Suite

A curated multi-task benchmark designed to evaluate multi-agent memory systems (Nucleus) on chronologically ordered engineering task streams.

## Structure

```
team-swe/
├── README.md
├── .env.example
├── clusters/               # Cluster manifests (execution order specs)
│   └── django__forms.json
├── dataset/                # Full SWE-bench-compatible instance payloads
│   └── django__forms/
│       └── instances.json
└── scripts/                # Mining, extraction, and validation tools
    ├── mine_django.py
    ├── extract_instance.py
    └── validate_cluster.py
```

## Setup & Prerequisites

1. Copy `.env.example` to `.env` and set your `GITHUB_TOKEN` (required for mining PRs via GitHub API):
   ```bash
   cp team-swe/.env.example team-swe/.env
   ```
2. Python environment with `requests`, `pandas`, and `python-dotenv`.
