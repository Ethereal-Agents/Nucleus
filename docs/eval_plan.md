# Measurement Framework: Team-SWE Benchmark (using Inspect AI)

This plan details the implementation of the evaluation framework, aligning entirely with the **Team-SWE Benchmark** defined in your Pivot Plan. It uses **Inspect AI** for a modular design that supports both current public benchmarks (`SWE-ContextBench`) and your future custom evaluation datasets.

## User Review Required

> [!IMPORTANT]
> The plan has been updated to reflect the exact ablation arms (Stateless, Naive RAG, Swarm Hub), the cost-optimized model tiering (Claude 3.5 Haiku), and the graph-traversal task curation methodology.

---

## 1. Dataset Extensibility (Custom Eval Datasets)

While our first test streams `SWE-ContextBench` clusters from Hugging Face, the harness will be designed so that swapping to a custom dataset requires zero changes to the underlying evaluation logic.

**Future-Proofing Architecture:**
- We will establish an `evals/datasets/` directory.
- The evaluation harness will accept a dataset source as a parameter.
- When you build your custom dataset, you simply drop a `.json` or `.jsonl` file into `evals/datasets/`, and the Inspect solver will iterate over it using `inspect_ai.dataset.json_dataset()`.

---

## 2. Measurement Framework Architecture

### Metrics Defined
- **Cumulative Token Cost:** Tracked via Inspect's model logs. 
  > [!WARNING]
  > OpenRouter cost tracking is notoriously broken in many external frameworks. **Before running the full benchmark**, we must write a minimal explicit test script to verify that Inspect AI accurately parses and records OpenRouter token costs/headers. If it fails, we will need to manually calculate cost using token counts * price per token.
- **Cost/Task Curve:** Exported from `inspect view` run logs to calculate cumulative token cost per task sequentially.
- **Resolution Rate:** Success vs Failure per task.
- **Redundant-Exploration Rate:** Calculated by analyzing the tool call histories recorded natively by Inspect.
- **Total Time to Resolution (TTR):** Wall-clock time taken by the agent to solve each task. This will reveal if the Hub saves time by reducing exploration, or if it slows the agent down.
- **Memory Hub Latency Overhead:** Wall-clock time spent explicitly inside `memory_search` and `memory_end_run` calls to isolate the exact latency tax of the Hub.

### Logging Integration
1. **Agent-1 Integration (`~/Projects/Agent-1`)**:
   - Wrap `Agent-1` into customizable Inspect `Solver` instances for each ablation arm.
2. **Hub Logging Hooks (`Nuclues`)**:
   - Track memory reads and writes; the agent's Inspect Solver will pull these metrics and attach them to `TaskState.metadata`.

---

## 3. The Team-SWE Benchmark (A/B/C Comparison)

We will run a controlled experiment using a multi-principal sequential task runner via Inspect, evaluating across 3 ablation arms.

### Task Identification Methodology (Custom DAG Extraction)
> [!NOTE]
> `SWEContextBench` natively only provides task relationship chains of length 2. Therefore, we constructed our own sequence via chronological diff-overlap analysis.

To rigorously curate the exact 10-task sequence without bias:
1. **Repository Filtering & Chronological Sorting:** Select dense repositories and sort their base tasks chronologically by `created_at`.
2. **Custom DAG Construction (Diff-Overlap):** Iterate through the sorted tasks. If a later task modifies files that an earlier task touched, draw a directed dependency edge.
3. **Chain Extraction:** Query the custom DAG to find the longest connected dependency chains.
4. **Contradiction Check (The "Moment of Truth"):** Parse the diffs for all extracted chains ($\ge$ 10 tasks) to find line-level inversion pairs (where a later task deletes/modifies code added by an earlier task). Score chains by max inversions.

#### Dataset Curation Results
Running this methodology yielded the following DAG analysis for dense repositories:
* `django/django`: Max length 2
* `sympy/sympy`: Max length 8
* `sphinx-doc/sphinx`: Max length 22
* `pydata/xarray`: Max length 12
* `jqlang/jq`: Max length 10

**Selected Sequence:** The 10-task sequence from `jqlang/jq` was selected as the absolute winner because it achieved a massive inversion score of **1578**, guaranteeing that late tasks fundamentally contradict foundational facts established in early tasks. The dataset has been exported to `evals/datasets/team_swe_cluster.json`.

### Setup
- **Dataset**: A continuous 10-task window selected via the methodology above.
- **Agent Codebase**: Wrapped around `~/Projects/Agent-1`.
- **Model Config**: `anthropic/claude-3-5-haiku-20241022` (for a highly cost-optimized sweep of ~$10-$18). We will reserve `claude-3-5-sonnet` for one confirmation row later.

### The Three Ablation Arms
1. **Baseline (Stateless Control)**: No hub. State/context is reset between every task.
2. **Naive RAG (Strawman)**: Vector database without temporal supersession. Facts are strictly appended.
3. **Hub-Active**: Our hub with summarized extraction + dense/BM25 + temporal supersession active.

### Execution Plan

1. **Test Harness (`evals/team_swe_eval.py`)**:
   - Define a Dataset loader that pulls the curated 10-task cluster.
   - Define the three agent solvers corresponding to the variants above.
   - Run tasks *sequentially* where the output state of Task N's memory persists to Task N+1 (except for Baseline).
   
2. **Running the Evals**:
   ```bash
   inspect eval evals/team_swe_eval.py@baseline --model anthropic/claude-3-5-haiku-20241022
   inspect eval evals/team_swe_eval.py@naive_rag --model anthropic/claude-3-5-haiku-20241022
   inspect eval evals/team_swe_eval.py@hub_active --model anthropic/claude-3-5-haiku-20241022
   ```

3. **Reporting (`evals/dashboard.py`)**:
   - A script using the `inspect_ai.log` API to parse the run logs across both variants and generate the cumulative cost curves, resolution rate tables, and temporal-contradiction deep-dive metrics, outputting to `docs/team_swe_results.md`.
