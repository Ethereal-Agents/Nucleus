#!/usr/bin/env python3
"""
Extractor script for Cluster #1 (Django Forms).
Reads team-swe/clusters/django_forms_candidates.json and generates full
SWE-bench-compatible instance payloads in team-swe/dataset/django__forms/instances.json.
"""

import json
import os
import sys
import urllib.parse
import urllib.request

from dotenv import load_dotenv

# Load root .env from Nucleus repository root
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
env_path = os.path.join(root_dir, ".env")
if os.path.exists(env_path):
    load_dotenv(env_path)

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
if not GITHUB_TOKEN:
    print(f"Error: GITHUB_TOKEN not found in {env_path}")
    sys.exit(1)

HEADERS = {
    "Authorization": f"Bearer {GITHUB_TOKEN}",
    "Accept": "application/vnd.github.v3+json",
    "User-Agent": "Nucleus-Team-SWE-Extractor",
}

# SWE-bench environment setup commit for Django 5.0 (Spring 2023 release window)
DJANGO_5_0_ENV_COMMIT = "4a72da71001f154ea60906a2f74898d32b7322a7"


def fetch_pr_patch(pr_number: int) -> str:
    """Fetch raw PR patch from GitHub API."""
    url = f"https://api.github.com/repos/django/django/pulls/{pr_number}"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {GITHUB_TOKEN}", "Accept": "application/vnd.github.v3.patch"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except Exception as e:
        print(f"  [Error fetching patch for PR #{pr_number}]: {e}")
        return ""


def split_patch_into_source_and_tests(raw_patch: str) -> tuple[str, str]:
    """
    Split unified diff patch into:
    - patch: source files diff (excluding tests/ and docs/)
    - test_patch: test files diff (only tests/)
    """
    source_chunks = []
    test_chunks = []

    current_chunk = []
    current_is_test = False

    for line in raw_patch.split("\n"):
        if line.startswith("diff --git "):
            if current_chunk:
                chunk_str = "\n".join(current_chunk) + "\n"
                if current_is_test:
                    test_chunks.append(chunk_str)
                else:
                    source_chunks.append(chunk_str)
                current_chunk = []

            # Check target filename
            parts = line.split()
            filename = parts[-1].lstrip("b/") if len(parts) >= 4 else ""
            current_is_test = filename.startswith("tests/")
            current_chunk.append(line)
        else:
            current_chunk.append(line)

    if current_chunk:
        chunk_str = "\n".join(current_chunk) + "\n"
        if current_is_test:
            test_chunks.append(chunk_str)
        else:
            source_chunks.append(chunk_str)

    return "".join(source_chunks), "".join(test_chunks)


def extract_instances():
    candidates_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "clusters", "django_forms_candidates.json"))
    if not os.path.exists(candidates_file):
        print(f"Error: Candidate file not found at {candidates_file}")
        sys.exit(1)

    with open(candidates_file) as f:
        data = json.load(f)

    tasks = data.get("tasks", [])
    print("=" * 80)
    print(f"EXTRACTING SWE-BENCH INSTANCE PAYLOADS FOR {len(tasks)} TASKS")
    print("=" * 80)

    instances = []

    for idx, task in enumerate(tasks, 1):
        pr_num = task["pr_number"]
        merge_commit = task["merge_commit_sha"]
        instance_id = f"django__django-{pr_num}"

        print(f"\n[Task {idx}/{len(tasks)}] Extracting {instance_id} (PR #{pr_num})...")

        raw_patch = fetch_pr_patch(pr_num)
        source_patch, test_patch = split_patch_into_source_and_tests(raw_patch)

        # Base commit is parent of squash merge commit
        base_commit = f"{merge_commit}~1"

        inst = {
            "instance_id": instance_id,
            "repo": "django/django",
            "base_commit": base_commit,
            "patch": source_patch,
            "test_patch": test_patch,
            "FAIL_TO_PASS": "[]",  # Populated during Docker Discovery (Step 4)
            "PASS_TO_PASS": "[]",  # Populated during Docker Discovery (Step 4)
            "problem_statement": task["problem_statement"],
            "hints_text": "",
            "created_at": task["merged_at"],
            "version": "5.0",
            "environment_setup_commit": DJANGO_5_0_ENV_COMMIT
        }

        instances.append(inst)

        print(f"  - Base Commit: {base_commit}")
        print(f"  - Source Patch: {len(source_patch)} bytes")
        print(f"  - Test Patch: {len(test_patch)} bytes")
        print(f"  - Problem Statement: {len(inst['problem_statement'])} chars")

    # Write dataset instances to team-swe/dataset/django__forms/instances.json
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dataset", "django__forms"))
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "instances.json")

    with open(out_path, "w") as f:
        json.dump(instances, f, indent=2)

    print("\n" + "=" * 80)
    print(f"SUCCESS: Saved {len(instances)} full instance payloads to:")
    print(f"  {out_path}")
    print("=" * 80)

    # Also write cluster manifest to team-swe/clusters/django__forms.json
    manifest_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "clusters", "django__forms.json"))
    with open(manifest_path, "w") as f:
        json.dump({
            "cluster_name": "django__forms",
            "repo": "django/django",
            "instance_ids": [inst["instance_id"] for inst in instances]
        }, f, indent=2)
    print(f"Cluster manifest saved to:\n  {manifest_path}")


if __name__ == "__main__":
    extract_instances()
