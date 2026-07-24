#!/usr/bin/env python3
"""
Mining script for Django PRs touching django/forms/.
Fetches up to 150 PRs, extracts real Django Trac ticket descriptions (code.djangoproject.com/ticket/<num>?format=csv),
shows a progress bar, and computes optimal 5-task sliding windows with >70% overlap.
"""

import argparse
import csv
import io
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime

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
    "User-Agent": "Nucleus-Team-SWE-Miner",
}

CACHE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "cache"))
CACHE_FILE = os.path.join(CACHE_DIR, "django_forms_prs_cache.json")


def progress_bar(current: int, total: int, prefix: str = "", length: int = 40):
    """Render an ASCII progress bar."""
    percent = float(current) / float(total) if total > 0 else 1.0
    filled = int(length * percent)
    bar = "█" * filled + "░" * (length - filled)
    sys.stdout.write(f"\r{prefix} [{bar}] {current}/{total} ({percent*100:.1f}%)")
    sys.stdout.flush()
    if current >= total:
        sys.stdout.write("\n")


def http_request(url: str, headers: dict = None, retries: int = 3) -> str | None:
    """Generic HTTP GET request returning text body."""
    req_headers = headers or {}
    for _attempt in range(retries):
        req = urllib.request.Request(url, headers=req_headers)
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as e:
            if e.code in (403, 429):
                time.sleep(10)
            elif e.code == 404:
                return None
            else:
                time.sleep(3)
        except Exception:
            time.sleep(3)
    return None


def fetch_github_json(url: str) -> dict | list:
    """Make authenticated request to GitHub API with rate limit handling."""
    res = http_request(url, headers=HEADERS)
    if res:
        try:
            return json.loads(res)
        except json.JSONDecodeError:
            return {}
    return {}


def extract_linked_ticket(pr_body: str, title: str = "") -> int | None:
    """Extract linked Django Trac ticket number from PR body or title."""
    text = (pr_body or "") + " " + (title or "")
    patterns = [
        r"(?:fixes|fixed|closes|closed|re|refs|references)\s+#(\d{5})",
        r"(?:fixes|fixed|closes|closed|re|refs|references)\s+https?://code\.djangoproject\.com/ticket/(\d{5})",
        r"code\.djangoproject\.com/ticket/(\d{5})",
        r"Ticket\s+#?(\d{5})",
        r"#(\d{5})",
    ]
    for pattern in patterns:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            return int(m.group(1))
    return None


def fetch_trac_ticket_details(ticket_num: int) -> dict | None:
    """Fetch original ticket description directly from Django's Trac issue tracker CSV export."""
    url = f"https://code.djangoproject.com/ticket/{ticket_num}?format=csv"
    csv_text = http_request(url, headers={"User-Agent": "Mozilla/5.0"})

    if not csv_text or "id,summary" not in csv_text:
        return None

    try:
        reader = csv.DictReader(io.StringIO(csv_text))
        for row in reader:
            summary = row.get("summary", "").strip()
            description = row.get("description", "").strip()
            if summary or description:
                return {
                    "ticket_number": ticket_num,
                    "summary": summary,
                    "description": description,
                    "url": f"https://code.djangoproject.com/ticket/{ticket_num}",
                }
    except Exception as e:
        print(f"  [Trac Parse Error for #{ticket_num}]: {e}")

    return None


def search_django_forms_prs(target_subsystem: str = "django/forms", max_prs: int = 150) -> list[dict]:
    """Search closed, merged PRs modifying target subsystem via GitHub API + Trac Ticket scraper."""
    print(f"\n[1/2] Searching PRs touching '{target_subsystem}' on GitHub API (Target: {max_prs} PRs)...")

    candidate_pr_numbers = []
    page = 1

    # Phase A: Collect PR numbers from Search API
    while len(candidate_pr_numbers) < max_prs and page <= 10:
        query = urllib.parse.quote("repo:django/django is:pr is:merged forms")
        url = f"https://api.github.com/search/issues?q={query}&sort=created&order=desc&per_page=50&page={page}"
        res = fetch_github_json(url)

        if not isinstance(res, dict) or "items" not in res:
            break

        items = res.get("items", [])
        if not items:
            break

        for item in items:
            candidate_pr_numbers.append(item["number"])
            if len(candidate_pr_numbers) >= max_prs:
                break
        page += 1

    print(f"Found {len(candidate_pr_numbers)} candidate PRs. Fetching details and Trac tickets...\n")

    prs = []
    total_candidates = len(candidate_pr_numbers)

    # Phase B: Detailed PR Inspection & Trac Scraping with Progress Bar
    for idx, pr_num in enumerate(candidate_pr_numbers, 1):
        progress_bar(idx, total_candidates, prefix="Processing PRs")

        pr_url = f"https://api.github.com/repos/django/django/pulls/{pr_num}"
        pr_data = fetch_github_json(pr_url)

        if not isinstance(pr_data, dict) or not pr_data.get("merged_at"):
            continue

        merged_at = pr_data["merged_at"]
        merge_commit_sha = pr_data.get("merge_commit_sha")
        body = pr_data.get("body") or ""
        title = pr_data.get("title") or ""

        # Files API
        files_url = f"https://api.github.com/repos/django/django/pulls/{pr_num}/files?per_page=100"
        files_data = fetch_github_json(files_url)
        if not isinstance(files_data, list):
            files_data = []

        filenames = [f["filename"] for f in files_data if isinstance(f, dict)]
        forms_files = [f for f in filenames if f.startswith(target_subsystem)]

        if not forms_files:
            continue

        # Ticket Extraction & Trac Fetching
        linked_ticket = extract_linked_ticket(body, title)
        trac_info = fetch_trac_ticket_details(linked_ticket) if linked_ticket else None

        problem_statement = ""
        if trac_info and len(trac_info.get("description") or "") > 50:
            problem_statement = f"Ticket #{linked_ticket}: {trac_info['summary']}\n\n{trac_info['description']}"
        elif len(body) > 100:
            problem_statement = f"{title}\n\n{body}"

        # Ignore if problem statement is insufficient
        if len(problem_statement) < 100:
            continue

        prs.append({
            "pr_number": pr_num,
            "title": title,
            "merged_at": merged_at,
            "merge_commit_sha": merge_commit_sha,
            "linked_ticket": linked_ticket,
            "trac_summary": trac_info["summary"] if trac_info else "",
            "problem_statement": problem_statement,
            "all_files": filenames,
            "subsystem_files": forms_files,
        })

    prs.sort(key=lambda x: x["merged_at"])
    print(f"\nSuccessfully mined {len(prs)} qualified PRs with Trac descriptions.")
    return prs


def get_cached_or_fetch_prs(refresh: bool = False, max_prs: int = 150) -> list[dict]:
    """Retrieve PR data from local cache if available, otherwise fetch from GitHub + Trac."""
    os.makedirs(CACHE_DIR, exist_ok=True)

    if os.path.exists(CACHE_FILE) and not refresh:
        print(f"Loading cached PR data from: {CACHE_FILE} (0 API calls)...")
        with open(CACHE_FILE) as f:
            prs = json.load(f)
        print(f"Loaded {len(prs)} PRs from local cache. Pass --refresh to re-fetch.")
        return prs

    prs = search_django_forms_prs(target_subsystem="django/forms", max_prs=max_prs)

    if prs:
        with open(CACHE_FILE, "w") as f:
            json.dump(prs, f, indent=2)
        print(f"Saved {len(prs)} PRs to local cache: {CACHE_FILE}")

    return prs


def find_best_sliding_window(prs: list[dict], window_size: int = 5, max_days: int = 120) -> tuple[list[dict], float]:
    """Find sliding window of N PRs within max_days that maximizes forward file overlap."""
    best_window = []
    best_overlap = -1.0

    for i in range(len(prs) - window_size + 1):
        window = prs[i : i + window_size]

        t_start = datetime.strptime(window[0]["merged_at"], "%Y-%m-%dT%H:%M:%SZ")
        t_end = datetime.strptime(window[-1]["merged_at"], "%Y-%m-%dT%H:%M:%SZ")
        days_span = (t_end - t_start).days

        if days_span > max_days:
            continue

        overlaps = []
        accumulated_files = set()

        for idx, task in enumerate(window):
            task_files = set(task["subsystem_files"])
            if idx > 0:
                intersection = task_files.intersection(accumulated_files)
                overlap_ratio = len(intersection) / len(task_files) if task_files else 0.0
                overlaps.append(overlap_ratio)
            accumulated_files.update(task_files)

        mean_overlap = sum(overlaps) / len(overlaps) if overlaps else 0.0
        has_valid_statements = all(len(t.get("problem_statement", "")) > 100 for t in window)

        if mean_overlap > best_overlap and has_valid_statements:
            best_overlap = mean_overlap
            best_window = window

    return best_window, best_overlap


def main():
    parser = argparse.ArgumentParser(description="Mine Django forms PRs for NucleusBench")
    parser.add_argument("--refresh", action="store_true", help="Bypass cache and force re-fetch from GitHub + Trac")
    parser.add_argument("--size", type=int, default=5, help="Sequence window size (default: 5)")
    parser.add_argument("--max-days", type=int, default=120, help="Max time window span in days (default: 120)")
    parser.add_argument("--max-prs", type=int, default=150, help="Search depth max PRs (default: 150)")
    args = parser.parse_args()

    print("=" * 80)
    print("DJANGO FORMS PR & TRAC TICKET MINER FOR NUCLEUS-BENCH")
    print("=" * 80)

    prs = get_cached_or_fetch_prs(refresh=args.refresh, max_prs=args.max_prs)

    if len(prs) < args.size:
        print(f"\nNot enough PRs in dataset ({len(prs)}). Run with --refresh to fetch more.")
        return

    print(f"\n[2/2] Evaluating sliding sequence windows (size={args.size}, max span={args.max_days} days)...")
    best_window, best_overlap = find_best_sliding_window(prs, window_size=args.size, max_days=args.max_days)

    if best_window:
        t_start = best_window[0]["merged_at"][:10]
        t_end = best_window[-1]["merged_at"][:10]
        t_span = (datetime.strptime(best_window[-1]["merged_at"], "%Y-%m-%dT%H:%M:%SZ") - datetime.strptime(best_window[0]["merged_at"], "%Y-%m-%dT%H:%M:%SZ")).days

        print("\n" + "=" * 80)
        print(f"SELECTED CLUSTER #1 CANDIDATE SEQUENCE ({args.size} TASKS, Mean Overlap: {best_overlap*100:.1f}%)")
        print("=" * 80)
        print(f"Time Span: {t_start} to {t_end} ({t_span} days)\n")

        for idx, task in enumerate(best_window):
            print(f"Task {idx+1}: PR #{task['pr_number']} (Ticket #{task['linked_ticket']}) - {task['title']}")
            print(f"  - Merged At: {task['merged_at']}")
            print(f"  - Merge Commit: {task['merge_commit_sha']}")
            print(f"  - Subsystem Files: {task['subsystem_files']}")
            print(f"  - Problem Statement Length: {len(task['problem_statement'])} chars (Trac description)")
            print()

        out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "clusters"))
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "django_forms_candidates.json")
        with open(out_path, "w") as f:
            json.dump({
                "cluster_name": "django__forms",
                "repo": "django/django",
                "sequence_length": args.size,
                "time_span_days": t_span,
                "mean_forward_overlap": best_overlap,
                "tasks": best_window
            }, f, indent=2)
        print(f"Candidate sequence saved to: {out_path}")
    else:
        print(f"No window found for size={args.size} within {args.max_days} days. Try increasing --max-days.")


if __name__ == "__main__":
    main()
