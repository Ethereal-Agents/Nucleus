# SWE-ContextBench Cluster Selection Report

> **Date:** 2026-07-16  
> **Authors:** Ayush Dubey  
> **Status:** Cluster selection finalised. Ready for ablation runs.  
> **Data Sources:** [SWE-ContextBench](https://huggingface.co/datasets/jiayuanz3/SWEContextBench) (arXiv 2602.08316) · GitHub REST API v3

---

## 1. Objective

Select a subset of SWE-ContextBench task clusters that maximises the **measurable signal** of a shared memory system (Nucleus) in a simulated multi-principal enterprise team setting.

The selected clusters will serve as the substrate for a 3-arm ablation study:

| Arm | Description |
|---|---|
| **Stateless control** | No memory hub. Each task runs in isolation. |
| **Naive vector RAG** | Raw trajectory retrieval over prior runs. |
| **Nucleus hub** | Summarised facts + hybrid retrieval + bi-temporal supersession. |

The primary metric is **cumulative team token cost** vs. task index. The thesis predicts a sub-linear curve for the Nucleus arm.

---

## 2. Dataset Overview

SWE-ContextBench extends SWE-Bench with a relational structure that groups tasks by shared codebase context. The dataset is distributed as three Parquet files:

| File | Rows | Description |
|---|---|---|
| `SWEContextBench_Experience` | 1,100 | "Experience" tasks — the parent/hub reference points |
| `SWEContextBench_Related` | 376 | "Related" tasks — child tasks linked to a parent |
| `SWEContextBench_Relationship` | 376 | Mapping table with PR and issue URLs for both sides |

Each row in the Relationship table defines a directed `A → B` edge:

- **A** (`experience_instance_id`): The parent task whose resolution provides architectural context
- **B** (`related_instance_id`): A child task that shares codebase overlap with the parent

### 2.1 Cluster Definition

A **cluster** is defined as: one Experience task (the hub) together with all Related tasks that reference it. We group by `experience_instance_id` in the Relationship table.

This yields **229 unique clusters** containing a total of **605 tasks** (229 parents + 376 children).

---

## 3. Cluster Size Distribution

We computed the size of every cluster (parent + all children) across the full dataset:

| Cluster Size | Count | Percentage | Cumulative Tasks |
|---|---|---|---|
| 2 (1 parent + 1 child) | 151 | 65.9% | 302 |
| 3 (1 parent + 2 children) | 43 | 18.8% | 129 |
| 4 | 16 | 7.0% | 64 |
| 5 | 12 | 5.2% | 60 |
| 6 | 4 | 1.7% | 24 |
| 7 | 1 | 0.4% | 7 |
| 9 | 1 | 0.4% | 9 |
| 10 | 1 | 0.4% | 10 |

> [!IMPORTANT]
> **66% of clusters contain only 2 tasks** (1 parent + 1 child). These provide a single data point for measuring cost decline and are insufficient for plotting a meaningful cumulative cost curve. We therefore restrict selection to clusters with **≥ 4 tasks** (35 clusters, 174 tasks).

---

## 4. Chronological Ordering Methodology

### 4.1 Why Ordering Matters

In a realistic enterprise simulation, knowledge accumulates **forward in time**. An engineer solves a bug on Monday; a colleague encounters a related bug on Tuesday and benefits from the first engineer's documented experience. To faithfully simulate this, tasks within each cluster must be executed in the order their fixes were actually integrated into the codebase.

### 4.2 Choosing the Correct Timestamp

Three candidate timestamps exist for each task:

| Timestamp | Source | What It Represents |
|---|---|---|
| `created_at` (dataset) | SWE-ContextBench Parquet | Unreliable — often reflects the dataset compilation date rather than the true event date |
| PR `created_at` | GitHub API | When the fix was first submitted for review |
| PR `merged_at` | GitHub API | When the fix was accepted and merged into the main branch |

We use **PR `merged_at`** because it marks the moment knowledge becomes available to the team: the code is in the main branch, other engineers build on top of it, and a memory system would finalise its learnings at this point.

### 4.3 Data Collection

We extracted the `related_pr_url` and `experience_pr_url` fields from the Relationship table, which contain direct links to the GitHub Pull Requests for every task. We then queried the GitHub REST API (authenticated, 5,000 req/hr) to retrieve the true `merged_at` timestamp for each unique PR.

**Total unique PRs fetched:** ~100 across all ≥ 4 clusters.

### 4.4 Sorting Rule

All tasks within a cluster (parent and children alike) are sorted primarily by PR `merged_at` in ascending order. If multiple tasks share the exact same `merged_at` timestamp (which happens when a single PR closes multiple issues), we use the numeric issue/PR number extracted from the `instance_id` (e.g., `2750` from `jqlang__jq-2750`) as an ascending tiebreaker. This ensures older bugs are naturally encountered before newer ones. The parent task is **not** locked to position 0 — it naturally falls into its correct chronological position. In several clusters, the parent's PR was merged *after* the children's PRs, and the sorting correctly reflects this.

---

## 5. Forward Overlap Analysis

### 5.1 Motivation

Cluster size alone is insufficient for selection. The memory system's effectiveness depends on whether tasks within a cluster actually share code: if task `i` modifies the same files that were already explored by tasks `0..i-1`, the memory system can provide directly actionable context. If they touch completely different files, memory is noise.

### 5.2 Metric: Forward Overlap Score

For each cluster, sorted chronologically by PR `merged_at`:

1. **Extract modified files** from each task's gold `patch` field by parsing `diff --git a/<path> b/<path>` headers.
2. **For each task at position `i`** (where `i > 0`), compute the intersection of its modified files with the **union** of all files modified by tasks `0..i-1`.
3. **Forward overlap** for task `i` = `|intersection| / |task_i_files|`.
4. **Cluster forward overlap** = mean of all per-task forward overlaps.

A score of 100% means every file that every subsequent task touches has already been explored by an earlier task in the chronological stream. This is the maximum theoretical utility of a memory system.

### 5.3 Ranking Formula

Clusters are ranked by a composite **Memory Utility Score**:

$$\text{Memory Utility Score} = \text{Forward Overlap} \times \text{Cluster Size}$$

This balances two requirements: (1) enough data points for a cost-decline curve, and (2) high file overlap where the memory system is most likely to demonstrate value.

---

## 6. Results: Full Ranking (≥ 4 tasks)

All 35 clusters with ≥ 4 tasks, ranked by Memory Utility Score. PR `merged_at` dates verified via authenticated GitHub API.

| Rank | Cluster | Size | Forward Overlap | Score | Language |
|---|---|---|---|---|---|
| 1 | `jqlang__jq-2750` | 9 | 100.0% | 9.00 | C |
| 2 | `django__django-12708` | 10 | 88.9% | 8.89 | Python |
| 3 | `sharkdp__bat-2650` | 7 | 91.7% | 6.42 | Rust |
| 4 | `burntsushi__ripgrep-2209` | 6 | 100.0% | 6.00 | Rust |
| 5 | `matplotlib__matplotlib-24149` | 6 | 90.0% | 5.40 | Python |
| 6 | `axios__axios-5316` | 5 | 100.0% | 5.00 | JavaScript |
| 7 | `django__django-12209` | 5 | 100.0% | 5.00 | Python |
| 8 | `sympy__sympy-18199` | 5 | 100.0% | 5.00 | Python |
| 9 | `sympy__sympy-24661` | 5 | 100.0% | 5.00 | Python |
| 10 | `sympy__sympy-20801` | 6 | 82.9% | 4.97 | Python |
| 11 | `pydata__xarray-6992` | 5 | 91.7% | 4.58 | Python |
| 12 | `babel__babel-15445` | 4 | 100.0% | 4.00 | JavaScript |
| 13 | `django__django-15382` | 4 | 100.0% | 4.00 | Python |
| 14 | `gin-gonic__gin-3227` | 4 | 100.0% | 4.00 | Go |
| 15 | `nushell__nushell-12901` | 4 | 100.0% | 4.00 | Rust |
| 16 | `sphinx-doc__sphinx-9367` | 4 | 100.0% | 4.00 | Python |
| 17 | `django__django-15930` | 6 | 62.8% | 3.77 | Python |
| 18 | `django__django-13964` | 5 | 75.0% | 3.75 | Python |
| 19 | `scikit-learn__scikit-learn-25638` | 5 | 75.0% | 3.75 | Python |
| 20 | `sympy__sympy-14976` | 5 | 75.0% | 3.75 | Python |
| 21 | `django__django-16910` | 5 | 70.8% | 3.54 | Python |
| 22 | `sympy__sympy-12419` | 5 | 66.7% | 3.33 | Python |
| 23 | `django__django-15695` | 4 | 83.3% | 3.33 | Python |
| 24 | `pydata__xarray-6461` | 4 | 83.3% | 3.33 | Python |
| 25 | `sphinx-doc__sphinx-8265` | 4 | 83.3% | 3.33 | Python |
| 26 | `sympy__sympy-16988` | 4 | 83.3% | 3.33 | Python |
| 27 | `sphinx-doc__sphinx-11510` | 4 | 75.0% | 3.00 | Python |
| 28 | `django__django-15252` | 5 | 58.3% | 2.92 | Python |
| 29 | `sphinx-doc__sphinx-10614` | 4 | 72.2% | 2.89 | Python |
| 30 | `django__django-12125` | 4 | 66.7% | 2.67 | Python |
| 31 | `gin-gonic__gin-2755` | 4 | 66.7% | 2.67 | Go |
| 32 | `sympy__sympy-21930` | 5 | 50.0% | 2.50 | Python |
| 33 | `sympy__sympy-21055` | 4 | 44.4% | 1.78 | Python |
| 34 | `matplotlib__matplotlib-25332` | 4 | 41.7% | 1.67 | Python |
| 35 | `phpoffice__phpspreadsheet-3659` | 4 | 36.6% | 1.46 | PHP |

---

## 7. Selected Clusters for Headline Run

The top 7 clusters by Memory Utility Score form the **Gold Standard Set** for the headline ablation run:

| Cluster | Size | Overlap | Score | Repo | Language |
|---|---|---|---|---|---|
| `jqlang__jq-2750` | 9 | 100.0% | 9.00 | jqlang/jq | C |
| `django__django-12708` | 10 | 88.9% | 8.89 | django/django | Python |
| `sharkdp__bat-2650` | 7 | 91.7% | 6.42 | sharkdp/bat | Rust |
| `burntsushi__ripgrep-2209` | 6 | 100.0% | 6.00 | BurntSushi/ripgrep | Rust |
| `matplotlib__matplotlib-24149` | 6 | 90.0% | 5.40 | matplotlib/matplotlib | Python |
| `axios__axios-5316` | 5 | 100.0% | 5.00 | axios/axios | JavaScript |
| `django__django-12209` | 5 | 100.0% | 5.00 | django/django | Python |

### 7.1 Summary Statistics

| Metric | Value |
|---|---|
| **Total tasks** | 48 |
| **Unique repositories** | 5 |
| **Languages** | 4 (C, Python, Rust, JavaScript) |
| **Mean forward overlap** | 95.8% |
| **Estimated cost (3 arms × Haiku)** | ~$45–50 |

### 7.2 Selection Rationale

1. **Statistical power.** Every cluster has ≥ 5 tasks, providing enough data points per cluster to plot a per-cluster cost-decline curve and detect a sub-linear trend.
2. **Maximum memory signal.** Mean forward overlap of 95.8% means that on average, nearly every file a task touches has already been explored by a prior task in the chronological stream. If the memory system cannot demonstrate value here, it cannot demonstrate value anywhere.
3. **Language and repo diversity.** 4 languages across 5 repositories prevents the result from being dismissed as repo-specific or language-specific.
4. **Budget compliance.** 48 tasks × 3 arms × ~$0.30/run ≈ $43, within the spec's ~$50 headline budget.

---

## 8. Chronological Task Order (Selected Clusters)

Each cluster is sorted by the true PR `merged_at` date. The hub task's position is determined purely by chronology — it is not artificially pinned.

### 8.1 `jqlang__jq-2750` (9 tasks · C)

All 9 tasks were resolved by a single monolithic PR ([#2750](https://github.com/jqlang/jq/pull/2750)), merged on 2023-07-24. The PR rewrote the `try/catch` opcode architecture (`FORK_OPT` → `TRY_BEGIN`/`TRY_END`) and resolved 8 long-standing bugs simultaneously. Because they share the same merge timestamp, they are sorted by their issue number (oldest bugs first).

| Position | Task | PR Merged | Role |
|---|---|---|---|
| 0 | `jqlang__jq-1858` | 2023-07-24 | Child (Issue #1858) |
| 1–7 | `jq-1859`, `jq-1885`, `jq-2011`, `jq-2073`, `jq-2140`, `jq-2220`, `jq-2230` | 2023-07-24 | Children |
| 8 | `jqlang__jq-2750` | 2023-07-24 | Hub (PR #2750) |

### 8.2 `django__django-12708` (10 tasks · Python)

The children's PRs were merged in **July 2018**, nearly two years before the hub PR was merged in **April 2020**. The hub naturally falls to position 9 (last), demonstrating correct chronological behaviour: the children's code changes preceded the hub's architectural unification.

| Position | Task | PR Merged | Role |
|---|---|---|---|
| 0–8 | `django-28862`, `django-29123`, `django-28916`, `django-29124`, `django-26180` (+ duplicates) | 2018-07-19 | Children |
| 9 | `django__django-12708` | 2020-04-23 | Hub |

### 8.3 `sharkdp__bat-2650` (7 tasks · Rust)

Six children were merged on 2022-05-07; the hub PR was merged over a year later in September 2023.

| Position | Task | PR Merged | Role |
|---|---|---|---|
| 0–5 | `bat-915`, `bat-1980`, `bat-951`, `bat-1630`, `bat-1846`, `bat-1854` | 2022-05-07 | Children |
| 6 | `sharkdp__bat-2650` | 2023-09-14 | Hub |

### 8.4 `burntsushi__ripgrep-2209` (6 tasks · Rust)

All 6 tasks share the same merge date (2022-05-11), with the hub at position 0.

### 8.5 `matplotlib__matplotlib-24149` (6 tasks · Python)

The hub is at position 1, merged between the first child (September 2022) and the remaining children (May 2023).

### 8.6 `axios__axios-5316` (5 tasks · JavaScript)

All 5 tasks share the same merge date (2023-01-30), with the hub at position 0.

### 8.7 `django__django-12209` (5 tasks · Python)

The hub is at position 1, merged between the first child (August 2019) and the remaining children (March 2020).

---

## 9. Observations and Caveats

### 9.1 Same-PR Clusters

In several clusters (notably `jqlang__jq-2750` and `burntsushi__ripgrep-2209`), all children reference the **same PR URL** as the hub. This means the parent and all children were resolved by a single monolithic pull request. Consequently:

- All tasks within these clusters share identical gold patches.
- The forward overlap score is trivially 100%.
- The memory signal is strongest here but also least realistic — in practice, a single PR would be a single agent run, not multiple sequential runs.

> [!WARNING]
> These same-PR clusters should be interpreted carefully. They represent an upper bound on memory utility — the "best case" scenario where the codebase knowledge is maximally transferable.

### 9.2 Duplicate Instance IDs

Some clusters (e.g., `django__django-12708`) contain **duplicate instance IDs** in their Relationship entries. This appears to be a data quality issue in the SWEContextBench dataset where the same child task is listed multiple times against the same hub. The duplicates were preserved as-is to avoid silently altering the dataset.

### 9.3 Timestamp Ties

When multiple tasks share the same `merged_at` timestamp (common for same-PR clusters), we break the tie using the numeric issue number extracted from the instance ID in ascending order. This simulates a realistic scenario where an engineer addresses the oldest reported bugs before moving to newer ones or finalising the overarching fix.

---
