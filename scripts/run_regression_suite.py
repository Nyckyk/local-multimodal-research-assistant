"""Run fast, real-model, and optional UI regression layers with artifacts."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.regression_support import ModelAvailabilityError, require_configured_models


def _arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--fast", action="store_true", help="run tests that mock local models")
    mode.add_argument("--ollama", action="store_true", help="run fixture-driven real-model tests")
    mode.add_argument("--all", action="store_true", help="run fast then real-model tests")
    parser.add_argument("--smoke", action="store_true", help="also run the optional Streamlit UI smoke test")
    return parser.parse_args()


def _load_cases():
    path = ROOT / "tests" / "fixtures" / "regression_cases.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))["cases"]


def _result_root():
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%z")
    root = ROOT / "test_results" / timestamp
    for folder in ("raw_responses", "structured_outputs", "failure_details"):
        (root / folder).mkdir(parents=True, exist_ok=True)
    return root


def _run_pytest(root, shard, arguments, timeout=None, extra_env=None):
    env = os.environ.copy()
    env.update({
        "REGRESSION_RESULTS_DIR": str(root),
        "REGRESSION_SHARD": shard,
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        **(extra_env or {}),
    })
    # Keep pytest's temporary files inside the timestamped result directory.
    # This avoids inherited ACL problems in the system-wide pytest temp root on
    # Windows and keeps each independently executed shard isolated.
    base_temp = root / "_pytest_tmp" / shard
    base_temp.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, "-m", "pytest", "--basetemp", str(base_temp),
        *arguments,
    ]
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command, cwd=ROOT, env=env, text=True, encoding="utf-8",
            errors="replace", capture_output=True, timeout=timeout,
        )
        output = completed.stdout + ("\n" + completed.stderr if completed.stderr else "")
        exit_code, timed_out = completed.returncode, False
    except subprocess.TimeoutExpired as error:
        stdout = error.stdout.decode("utf-8", errors="replace") if isinstance(error.stdout, bytes) else (error.stdout or "")
        stderr = error.stderr.decode("utf-8", errors="replace") if isinstance(error.stderr, bytes) else (error.stderr or "")
        output = stdout + "\n" + stderr + f"\nTIMEOUT after {timeout} seconds"
        exit_code, timed_out = 124, True
    duration = round(time.perf_counter() - started, 3)
    (root / "failure_details" / f"{shard}_pytest.log").write_text(output, encoding="utf-8")
    if timed_out:
        payload = {"exit_code": exit_code, "duration_seconds": duration, "tests": [{
            "nodeid": shard, "status": "FAIL", "duration_seconds": duration,
            "detail": f"timeout after {timeout} seconds", "failure_kind": "timeout",
        }]}
        (root / f"pytest_{shard}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    elif exit_code:
        # Collection/setup errors can make pytest exit nonzero without producing
        # a failed test call. Never allow such a shard to appear as a green run.
        report_path = root / f"pytest_{shard}.json"
        if report_path.exists():
            payload = json.loads(report_path.read_text(encoding="utf-8"))
        else:
            payload = {"duration_seconds": duration, "tests": []}
        payload["exit_code"] = exit_code
        payload.setdefault("duration_seconds", duration)
        tests = payload.setdefault("tests", [])
        if not any(row.get("status") == "FAIL" for row in tests):
            tests.append({
                "nodeid": f"{shard}::pytest_process",
                "status": "FAIL",
                "duration_seconds": duration,
                "detail": (
                    f"pytest exited with code {exit_code}; see "
                    f"failure_details/{shard}_pytest.log"
                ),
                "failure_kind": "test_error",
            })
        report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return exit_code


def _read_results(root):
    tests, duration = [], 0.0
    for path in sorted(root.glob("pytest_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        duration += float(payload.get("duration_seconds", 0))
        tests.extend(payload.get("tests", []))
    return tests, round(duration, 3)


def _enrich_failure_details(root, tests):
    """Put concise failure categories/messages into JSON and Markdown reports."""
    for row in tests:
        if row.get("status") != "FAIL":
            continue
        match = re.search(r"\[([^\]]+)\]$", row.get("nodeid", ""))
        if not match:
            continue
        path = root / "failure_details" / f"{match.group(1)}.json"
        if not path.exists():
            continue
        detail = json.loads(path.read_text(encoding="utf-8"))
        row["failure_kind"] = detail.get("failure_kind", "test_error")
        row["detail"] = detail.get("message", row.get("detail", ""))
    return tests


def _write_report(root, mode, preflight, wall_duration):
    tests, pytest_duration = _read_results(root)
    tests = _enrich_failure_details(root, tests)
    counts = {status: sum(row.get("status") == status for row in tests) for status in ("PASS", "FAIL", "SKIP")}
    payload = {
        "mode": mode, "generated_at": datetime.now().astimezone().isoformat(),
        "preflight": preflight, "counts": counts,
        "total_duration_seconds": round(wall_duration, 3),
        "pytest_shard_duration_seconds": pytest_duration, "tests": tests,
    }
    (root / "results.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    status = "FAIL" if counts["FAIL"] or preflight.get("status") == "FAIL" else "PASS"
    lines = [
        "# Regression test report", "", f"- Mode: `{mode}`", f"- Status: **{status}**",
        f"- Tests: {counts['PASS']} PASS, {counts['FAIL']} FAIL, {counts['SKIP']} SKIP",
        f"- Total duration: {wall_duration:.3f} seconds",
        f"- Model preflight: {preflight.get('status', 'not required')}",
    ]
    if preflight.get("detail"):
        lines.append(f"- Preflight detail: {preflight['detail']}")
    lines.extend(["", "## Cases", "", "| Status | Test | Duration (s) | Detail |", "|---|---|---:|---|"])
    for row in tests:
        detail_lines = str(row.get("detail", "")).splitlines()
        detail = (detail_lines[0] if detail_lines else "").replace("|", "\\|")[:180]
        lines.append(f"| {row['status']} | `{row['nodeid']}` | {row.get('duration_seconds', 0)} | {detail} |")
    lines.extend([
        "", "## Artifacts", "", "- Raw responses: `raw_responses/`",
        "- Validated structured outputs: `structured_outputs/`",
        "- Failure details and pytest logs: `failure_details/`", "",
        "> Scientific expected values are protected. Do not change fixtures merely to make tests pass; changes require explicit user approval.",
    ])
    (root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return counts


def main():
    args, root, started = _arguments(), _result_root(), time.perf_counter()
    cases = _load_cases()
    missing = [case["pdf_filename"] for case in cases if not (ROOT / "papers" / case["pdf_filename"]).exists()]
    preflight, exit_code = {"status": "PASS", "detail": "PDF fixtures found"}, 0
    if missing:
        preflight, exit_code = {"status": "FAIL", "detail": "Missing PDF fixture(s): " + ", ".join(sorted(set(missing)))}, 2
    needs_ollama = args.ollama or args.all or args.smoke
    if not missing and needs_ollama:
        try:
            models = require_configured_models()
            preflight = {"status": "PASS", "detail": "Configured models found: " + ", ".join(sorted(models))}
        except ModelAvailabilityError as error:
            preflight, exit_code = {"status": "FAIL", "detail": str(error)}, 2
    if exit_code == 0 and (args.fast or args.all):
        exit_code |= _run_pytest(root, "fast", ["-m", "not ollama and not smoke"])
    if exit_code == 0 and (args.ollama or args.all):
        for case in cases:
            case_id, timeout = case["case_id"], int(case.get("timeout", 180)) + 30
            exit_code |= _run_pytest(
                root, case_id,
                ["tests/test_ollama_regression.py", "-k", case_id, "-m", "ollama"],
                timeout=timeout, extra_env={"REGRESSION_CASE_ID": case_id},
            )
        if exit_code == 0:
            exit_code |= _run_pytest(
                root,
                "automatic_visual_resolution",
                [
                    "tests/test_ollama_regression.py", "-k",
                    "automatic_resolution", "-m", "ollama",
                ],
                timeout=120,
            )
    if exit_code == 0 and args.smoke:
        exit_code |= _run_pytest(root, "smoke", ["tests/test_streamlit_smoke.py", "-m", "smoke"], timeout=300)
    wall = time.perf_counter() - started
    mode = "all" if args.all else "ollama" if args.ollama else "fast"
    counts = _write_report(root, mode + ("+smoke" if args.smoke else ""), preflight, wall)
    print(f"Regression report: {root / 'summary.md'}")
    print(f"Results JSON: {root / 'results.json'}")
    print(f"PASS={counts['PASS']} FAIL={counts['FAIL']} SKIP={counts['SKIP']} duration={wall:.3f}s")
    if preflight["status"] == "FAIL":
        print(preflight["detail"])
    return 1 if exit_code or counts["FAIL"] or preflight["status"] == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
