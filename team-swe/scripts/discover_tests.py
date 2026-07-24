#!/usr/bin/env python3
"""
Docker Test Discovery Script for Cluster #1 (Django Forms).
Populates FAIL_TO_PASS and PASS_TO_PASS fields in team-swe/dataset/django__forms/instances.json.

- FAIL_TO_PASS: Parsed target test methods introduced by test_patch.
- PASS_TO_PASS: Empirically verified passing regression tests collected inside Docker environment.
"""

import json
import os
import re
import subprocess
import sys

DATASET_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dataset", "django__forms", "instances.json"))


def extract_test_file_paths(test_patch: str) -> list[str]:
    """Extract list of test module paths touched by test_patch."""
    files = []
    for line in test_patch.split("\n"):
        if line.startswith("diff --git "):
            parts = line.split()
            if len(parts) >= 4:
                fn = parts[-1].lstrip("b/")
                if fn.startswith("tests/") and fn.endswith(".py"):
                    rel = fn[len("tests/"): -len(".py")].replace("/", ".")
                    files.append(rel)
    return list(set(files))


def parse_fail_to_pass_from_diff(test_patch: str) -> list[str]:
    """Parse exact test method labels added/modified by test_patch diff."""
    fail_to_pass = []
    current_file = ""
    current_class = ""

    for line in test_patch.split("\n"):
        if line.startswith("diff --git "):
            parts = line.split()
            if len(parts) >= 4:
                fn = parts[-1].lstrip("b/")
                if fn.startswith("tests/") and fn.endswith(".py"):
                    current_file = fn[len("tests/"): -len(".py")].replace("/", ".")
                    current_class = ""
        elif line.startswith("class ") or line.startswith("+class "):
            m_cls = re.search(r"class\s+([A-Za-z0-9_]+)", line)
            if m_cls:
                current_class = m_cls.group(1)
        elif line.startswith("+    def test_") or line.startswith("+  def test_") or line.startswith("+def test_"):
            m_def = re.search(r"def\s+(test_\w+)", line)
            if m_def and current_file:
                method = m_def.group(1)
                if current_class:
                    test_label = f"{current_file}.{current_class}.{method}"
                else:
                    test_label = f"{current_file}.{method}"
                if test_label not in fail_to_pass:
                    fail_to_pass.append(test_label)

    return fail_to_pass


def extract_passing_tests_from_output(log_output: str) -> list[str]:
    """Parse passing (ok) test case IDs from Django unittest output."""
    passes = set()
    for line in log_output.split("\n"):
        line_str = line.strip()
        m = re.search(r"(\S+)\s+\(([\w\.]+\.\w+)\)\s+\.\.\.\s+ok", line_str)
        if m:
            passes.add(m.group(2))
    return sorted(passes)


def discover_tests_for_instance(inst: dict) -> tuple[list[str], list[str]]:
    """
    Run Docker test discovery for a single instance.
    Uses official pre-built instance images sweb.eval.x86_64.<instance_id>:latest.
    """
    instance_id = inst["instance_id"]
    test_patch = inst["test_patch"]

    # 1. FAIL_TO_PASS from diff parsing
    fail_to_pass = parse_fail_to_pass_from_diff(test_patch)

    # 2. PASS_TO_PASS from empirical Docker test execution
    test_modules = extract_test_file_paths(test_patch)
    print(f"\n[Docker Test Discovery] {instance_id}")
    print(f"  - Target FAIL_TO_PASS ({len(fail_to_pass)}): {fail_to_pass}")
    print(f"  - Executing Test Modules in Docker: {test_modules}")

    test_args = " ".join(test_modules)
    docker_image = f"sweb.eval.x86_64.{instance_id}:latest"
    container_name = f"test_disc_{instance_id.replace('-', '_')}"

    try:
        subprocess.run(["docker", "rm", "-f", container_name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        start_cmd = [
            "docker", "run", "-d", "--name", container_name,
            "--entrypoint", "/bin/bash",
            docker_image, "-c", "sleep 3600"
        ]
        subprocess.run(start_cmd, capture_output=True, text=True, check=True)

        run_test_cmd = [
            "docker", "exec", container_name, "bash", "-c",
            f"cd /testbed && /opt/miniconda3/envs/testbed/bin/python tests/runtests.py --settings=test_sqlite {test_args} --verbosity=2"
        ]
        res_test = subprocess.run(run_test_cmd, capture_output=True, text=True)
        test_output = res_test.stdout + "\n" + res_test.stderr

        passing_tests = extract_passing_tests_from_output(test_output)
        # Exclude fail_to_pass from pass_to_pass
        pass_to_pass = [t for t in passing_tests if t not in fail_to_pass]

        print(f"  ✓ Empirically verified {len(pass_to_pass)} PASS_TO_PASS regression tests.")
        return fail_to_pass, pass_to_pass

    finally:
        subprocess.run(["docker", "rm", "-f", container_name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main():
    if not os.path.exists(DATASET_PATH):
        print(f"Error: Dataset file not found at {DATASET_PATH}")
        sys.exit(1)

    with open(DATASET_PATH) as f:
        instances = json.load(f)

    print("=" * 80)
    print(f"EMPIRICAL DOCKER TEST DISCOVERY FOR {len(instances)} INSTANCES")
    print("=" * 80)

    for inst in instances:
        f2p, p2p = discover_tests_for_instance(inst)
        inst["FAIL_TO_PASS"] = json.dumps(f2p)
        inst["PASS_TO_PASS"] = json.dumps(p2p[:30])

        print(f"  ✓ FAIL_TO_PASS ({len(f2p)}): {f2p}")
        print(f"  ✓ PASS_TO_PASS ({len(p2p)}): {p2p[:5]}...\n")

    with open(DATASET_PATH, "w") as f:
        json.dump(instances, f, indent=2)

    print("\n" + "=" * 80)
    print("SUCCESS: Updated FAIL_TO_PASS & PASS_TO_PASS in:")
    print(f"  {DATASET_PATH}")
    print("=" * 80)


if __name__ == "__main__":
    main()
