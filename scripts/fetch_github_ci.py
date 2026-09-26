#!/usr/bin/env python3
"""Fetch and diagnose GitHub Actions CI workflow runs, job statuses, and logs.

Uses git credential helper or environment variables (GITHUB_TOKEN / GH_TOKEN) to
download and parse test logs directly from the GitHub REST API.

Usage:
    python scripts/fetch_github_ci.py [--branch BRANCH] [--run-id RUN_ID]
"""

import argparse
import io
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
import zipfile


class AuthRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Prevent Authorization header from being leaked to external CDN redirects."""

    def http_error_302(self, req, fp, code, msg, headers):
        loc = headers.get("Location")
        if loc:
            new_host = urllib.parse.urlparse(loc).netloc.lower()
            if not new_host.endswith("github.com") and not new_host.endswith(
                "githubusercontent.com"
            ):
                clean_headers = {
                    k: v for k, v in req.headers.items() if k.lower() != "authorization"
                }
                new_req = urllib.request.Request(loc, headers=clean_headers)
                return urllib.request.urlopen(new_req)
        return super().http_error_302(req, fp, code, msg, headers)

    http_error_301 = http_error_302
    http_error_303 = http_error_302
    http_error_307 = http_error_302
    http_error_308 = http_error_302


_opener = urllib.request.build_opener(AuthRedirectHandler)


def api_request(url, token=None):
    """Make an HTTP GET request to GitHub API."""
    headers = {
        "User-Agent": "scikit-kamila-ci-reader/1.0",
        "Accept": "application/vnd.github.v3+json",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"

    req = urllib.request.Request(url, headers=headers)
    try:
        with _opener.open(req) as resp:
            return resp.read()
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        print(f"[-] HTTP Error {e.code}: {e.reason}\n{body}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"[-] Request error: {e}", file=sys.stderr)
        return None


def get_workflow_runs(repo, token=None, branch=None, limit=5):
    """Fetch recent workflow runs."""
    url = f"https://api.github.com/repos/{repo}/actions/runs?per_page={limit}"
    if branch:
        url += f"&branch={branch}"
    data = api_request(url, token=token)
    if not data:
        return []
    return json.loads(data.decode("utf-8")).get("workflow_runs", [])


def get_jobs_for_run(repo, run_id, token=None):
    """Fetch all jobs for a specific workflow run."""
    url = f"https://api.github.com/repos/{repo}/actions/runs/{run_id}/jobs?per_page=100"
    data = api_request(url, token=token)
    if not data:
        return []
    return json.loads(data.decode("utf-8")).get("jobs", [])


def get_run_logs_zip(repo, run_id, token=None):
    """Download workflow run logs as a zip archive."""
    url = f"https://api.github.com/repos/{repo}/actions/runs/{run_id}/logs"
    data = api_request(url, token=token)
    if data:
        try:
            return zipfile.ZipFile(io.BytesIO(data))
        except Exception as e:
            print(f"[-] Failed to read zip archive: {e}", file=sys.stderr)
    return None


def get_job_log(repo, job_id, token=None):
    """Fetch raw log text for an individual job."""
    url = f"https://api.github.com/repos/{repo}/actions/jobs/{job_id}/logs"
    data = api_request(url, token=token)
    if data:
        return data.decode("utf-8", errors="replace")
    return None


def get_git_credentials():
    """Extract GitHub token from environment or git credential helper."""
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token:
        return token

    try:
        p = subprocess.Popen(
            ["git", "credential", "fill"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        out, _ = p.communicate(b"protocol=https\nhost=github.com\n\n")
        creds = dict(
            line.split("=", 1)
            for line in out.decode("utf-8", errors="replace").splitlines()
            if "=" in line
        )
        return creds.get("password")
    except Exception:
        return None


def get_current_git_info():
    """Extract owner/repo and current branch from git repository."""
    repo = None
    branch = None
    try:
        remote_url = subprocess.check_output(
            ["git", "config", "--get", "remote.origin.url"], universal_newlines=True
        ).strip()
        m = re.search(r"github\.com[:/]([^/]+)/([^/.]+)", remote_url)
        if m:
            repo = f"{m.group(1)}/{m.group(2)}"
    except Exception:
        pass

    try:
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], universal_newlines=True
        ).strip()
    except Exception:
        pass

    return repo, branch


def extract_failure_summary(log_text):
    """Extract concise failure tracebacks and summary from pytest logs."""
    if not log_text:
        return "No log content retrieved."

    lines = log_text.splitlines()
    summary_lines = []
    in_failures_section = False

    for line in lines:
        clean = re.sub(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z\s*", "", line)

        if "=== FAILURES ===" in clean or "=== ERRORS ===" in clean:
            in_failures_section = True

        if in_failures_section:
            summary_lines.append(clean)
            if re.search(
                r"===\s*(\d+\s+failed|\d+\s+passed|short test summary info)", clean
            ):
                pass
            if re.search(r"===\s*\d+\s+failed", clean):
                in_failures_section = False

    if summary_lines:
        return "\n".join(summary_lines[-100:])

    # Fallback: look for error / failure markers
    error_lines = [
        re.sub(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z\s*", "", l)
        for l in lines
        if any(
            w in l
            for w in [
                "FAILED ",
                "ERROR: ",
                "AssertionError",
                "Traceback",
                "TypeError",
                "ValueError",
                "ImportError",
            ]
        )
    ]
    if error_lines:
        return "\n".join(error_lines[-50:])

    return "\n".join(lines[-40:])


def main():
    parser = argparse.ArgumentParser(
        description="Read and diagnose GitHub Actions CI results."
    )
    parser.add_argument(
        "--repo", help="GitHub repository (owner/repo). Auto-detected if omitted."
    )
    parser.add_argument(
        "--branch", help="Git branch. Auto-detected from current HEAD if omitted."
    )
    parser.add_argument(
        "--run-id", type=int, help="Specific workflow run ID to inspect."
    )
    parser.add_argument(
        "--all-runs", action="store_true", help="Do not filter by branch."
    )
    parser.add_argument(
        "--limit", type=int, default=5, help="Number of workflow runs to list."
    )
    parser.add_argument("--job", help="Filter failure logs by job name substring.")
    args = parser.parse_args()

    token = get_git_credentials()
    if token:
        print("[+] GitHub token authenticated successfully.")
    else:
        print(
            "[-] Warning: No GitHub token found. "
            "Some API endpoints may be rate-limited or forbidden."
        )

    detected_repo, detected_branch = get_current_git_info()
    repo = args.repo or detected_repo or "ahfoss/scikit-kamila"
    branch = None if args.all_runs else (args.branch or detected_branch)

    print(f"[+] Target repository: {repo}")
    if branch:
        print(f"[+] Filtering by branch: {branch}")

    if args.run_id:
        target_run_id = args.run_id
        target_run_name = f"Run {args.run_id}"
    else:
        runs = get_workflow_runs(repo, token=token, branch=branch, limit=args.limit)
        if not runs:
            print(f"[-] No workflow runs found for {repo} (branch: {branch})")
            return

        print("\n" + "=" * 80)
        hdr = (
            f"{'RUN ID':<14} {'WORKFLOW':<25} "
            f"{'BRANCH':<15} {'STATUS':<12} {'CONCLUSION':<12}"
        )
        print(hdr)
        print("=" * 80)
        for r in runs:
            r_id = r.get("id")
            r_name = r.get("name", "Unknown")[:24]
            r_branch = r.get("head_branch", "Unknown")[:14]
            r_status = r.get("status", "")
            r_conclusion = r.get("conclusion") or "running"
            row = (
                f"{r_id:<14} {r_name:<25} "
                f"{r_branch:<15} {r_status:<12} {r_conclusion:<12}"
            )
            print(row)

        # Pick most recent failing run or the first run
        failing_runs = [r for r in runs if r.get("conclusion") == "failure"]
        target_run = failing_runs[0] if failing_runs else runs[0]
        target_run_id = target_run["id"]
        target_run_name = target_run.get("name", "Unknown")

    print("\n" + "-" * 80)
    print(f"[*] Diagnosing Run ID: {target_run_id} ({target_run_name})")
    print("-" * 80)

    jobs = get_jobs_for_run(repo, target_run_id, token=token)
    if not jobs:
        print("[-] No jobs found for this run.")
        return

    failed_jobs = [j for j in jobs if j.get("conclusion") == "failure"]
    print(f"[+] Total jobs: {len(jobs)}, Failed: {len(failed_jobs)}")

    for j in jobs:
        name = j.get("name")
        conclusion = j.get("conclusion") or j.get("status")
        symbol = (
            "PASS"
            if conclusion == "success"
            else ("FAIL" if conclusion == "failure" else "....")
        )
        print(f"  [{symbol:<4}] {name:<45} : {conclusion}")

    if not failed_jobs:
        print("\n[+] All jobs succeeded in this run!")
        return

    print("\n" + "=" * 80)
    print("[!] DOWNLOADING AND PARSING LOGS FOR FAILED JOBS:")
    print("=" * 80)

    # Download zip archive containing all logs, or fetch per-job logs
    log_zip = get_run_logs_zip(repo, target_run_id, token=token)
    zip_files = log_zip.namelist() if log_zip else []

    for j in failed_jobs:
        job_name = j.get("name")
        if args.job and args.job.lower() not in job_name.lower():
            continue

        print("\n" + "#" * 80)
        print(f"### Job: {job_name} (ID: {j.get('id')})")
        print("#" * 80)

        log_content = None
        if log_zip:
            # Match log filename in zip (e.g. "0_ubuntu-latest - Python 3.10.txt")
            matched_file = None
            for zf in zip_files:
                if job_name in zf and zf.endswith(".txt") and "system.txt" not in zf:
                    matched_file = zf
                    break
            if not matched_file:
                for zf in zip_files:
                    if job_name in zf:
                        matched_file = zf
                        break
            if matched_file:
                log_content = log_zip.read(matched_file).decode(
                    "utf-8", errors="replace"
                )

        if not log_content:
            # Fallback to direct job log API
            log_content = get_job_log(repo, j.get("id"), token=token)

        if log_content:
            print(extract_failure_summary(log_content))
        else:
            print(f"[-] Could not retrieve logs for {job_name}")


if __name__ == "__main__":
    main()
