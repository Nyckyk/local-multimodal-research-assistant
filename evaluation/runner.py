"""Execution and artifact generation for real end-to-end RAG suites."""

from __future__ import annotations

import json
import traceback
import re
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Callable

from evaluation.scoring import aggregate_runs, score_case


def load_suite(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("tests"), list):
        raise ValueError("Evaluation suite must be an object with a tests array.")
    ids = [row.get("id") for row in value["tests"]]
    if any(not isinstance(case_id, str) or not case_id for case_id in ids):
        raise ValueError("Every evaluation case needs a non-empty string id.")
    if len(ids) != len(set(ids)):
        raise ValueError("Evaluation case ids must be unique.")
    return value


def select_cases(suite: dict, question_id: str | None = None) -> tuple[list[dict], set[str]]:
    """Select a case and any declared conversational setup cases."""
    if not question_id:
        return list(suite["tests"]), {row["id"] for row in suite["tests"]}
    by_id = {row["id"]: row for row in suite["tests"]}
    if question_id not in by_id:
        raise ValueError(f"Unknown evaluation question id: {question_id}")
    wanted = by_id[question_id]
    setup_ids = wanted.get("setup_case_ids") or []
    missing = [case_id for case_id in setup_ids if case_id not in by_id]
    if missing:
        raise ValueError("Missing setup case(s): " + ", ".join(missing))
    return [*[by_id[case_id] for case_id in setup_ids], wanted], {question_id}


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, set):
        return sorted(value)
    if hasattr(value, "to_dict"):
        return value.to_dict()
    return repr(value)


def execute_run(
    suite: dict,
    backend,
    *,
    run_number: int = 1,
    question_id: str | None = None,
    allow_vision: bool = True,
    on_result: Callable[[dict], None] | None = None,
) -> dict:
    """Execute a suite serially, retaining only deliberate follow-up history."""
    cases, scored_ids = select_cases(suite, question_id)
    messages: list[dict] = []
    results = []
    started = perf_counter()
    for case in cases:
        if case.get("fresh_conversation", True):
            messages = []
        history_used = [dict(message) for message in messages]
        case_started = perf_counter()
        try:
            preferred = suite.get("document") if case.get("select_document", True) else None
            response_object = backend.ask(
                case["question"],
                conversation_messages=messages,
                preferred_pdf=preferred,
                automatic_visual_detection=True,
                save_vision_crops=bool(case.get("save_vision_crops", False)),
                allow_vision=allow_vision,
            )
            response = response_object.to_dict() if hasattr(response_object, "to_dict") else dict(response_object)
            resolution = response.get("resolution") or {}
            debug = response.get("debug") or {}
            response.setdefault("selected_document", resolution.get("pdf_name") or (
                (response.get("evidence") or {}).get("pdf")
                if isinstance(response.get("evidence"), dict) else None
            ))
            response.setdefault("resolved_target_type", resolution.get("target_type"))
            response.setdefault("resolved_target_number", resolution.get("target_number"))
            response.setdefault("resolved_page", resolution.get("page_number"))
            response.setdefault("vision_status", (
                "failed" if response.get("vision_error") else
                "used" if response.get("used_vision") else "not_used"
            ))
            response.setdefault("final_answer_code_path", debug.get("final_answer_code_path"))
            response.setdefault("requested_answer_slots", debug.get("requested_answer_slots", []))
            response.setdefault("unresolved_answer_slots", debug.get("missing_answer_slots", []))
            response.setdefault("final_answer_evidence", debug.get("final_answer_evidence"))
            scored = score_case(case, response)
            result = {
                "id": case["id"],
                "question": case["question"],
                "fresh_conversation": bool(case.get("fresh_conversation", True)),
                "setup_only": case["id"] not in scored_ids,
                "conversation_history_used": history_used,
                "duration_seconds": round(perf_counter() - case_started, 3),
                **response,
                **scored,
            }
            messages.extend([
                {"role": "user", "content": case["question"]},
                response_object.assistant_message() if hasattr(response_object, "assistant_message") else {
                    "role": "assistant", "content": response.get("answer", ""),
                    "sources": response.get("sources", []), "evidence": response.get("evidence"),
                    "visual_target": response.get("resolution"),
                },
            ])
        except Exception as error:
            result = {
                "id": case["id"], "question": case["question"],
                "fresh_conversation": bool(case.get("fresh_conversation", True)),
                "setup_only": case["id"] not in scored_ids,
                "conversation_history_used": history_used,
                "duration_seconds": round(perf_counter() - case_started, 3),
                "status": "ERROR", "score": 0.0,
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
                "checks": [], "passed": [], "missing": [], "contradictions": [],
            }
        if case["id"] in scored_ids:
            results.append(result)
            if on_result:
                on_result(result)
    return {
        "run": run_number,
        "duration_seconds": round(perf_counter() - started, 3),
        "results": results,
    }


def default_output_directory(root: Path, suite_name: str) -> Path:
    timestamp = datetime.now().astimezone().strftime("%Y-%m-%d_%H%M%S")
    return root / "results" / "e2e_eval" / suite_name / timestamp


def _answer_markdown(runs: list[dict]) -> str:
    lines = ["# Scientific RAG answers", ""]
    for run in runs:
        lines.extend([f"## Run {run['run']}", ""])
        for result in run["results"]:
            lines.extend([
                f"### {result['id']} — {result['status']}", "",
                f"**Question:** {result['question']}", "",
                str(result.get("answer") or result.get("error") or "No answer"), "",
            ])
    return "\n".join(lines).rstrip() + "\n"


def _scope_excerpt(answer: str, marker: str, next_markers: list[str]) -> str:
    text = str(answer or "")
    match = re.search(marker, text, re.I)
    if not match:
        return ""
    tail = text[match.end():]
    stops = [found.start() for pattern in next_markers if (found := re.search(pattern, tail, re.I))]
    return tail[:min(stops)] if stops else tail


def diagnose_results(suite: dict, runs: list[dict]) -> list[dict]:
    """Flag likely evaluator false positives without changing fixture contracts."""
    by_id = {case["id"]: case for case in suite.get("tests", [])}
    issues = []
    for run in runs:
        for result in run.get("results", []):
            if not result.get("contradictions"):
                continue
            case = by_id.get(result["id"], {})
            forbidden = {
                item.get("id"): item for field in ("forbidden_terms", "forbidden_claims")
                for item in (case.get("expect", case).get(field, [])) if isinstance(item, dict)
            }
            for check_id in result["contradictions"]:
                item = forbidden.get(check_id, {})
                if not item.get("scope_after"):
                    continue
                later_markers = [
                    other.get("scope_after") for other in forbidden.values()
                    if other.get("scope_after") and other.get("scope_after") != item.get("scope_after")
                ]
                excerpt = _scope_excerpt(result.get("answer", ""), item["scope_after"], later_markers)
                patterns = item.get("all_of") or item.get("any_of") or []
                if patterns and not _matches_for_diagnosis(excerpt, patterns, bool(item.get("all_of"))):
                    issues.append({
                        "run": run["run"], "id": result["id"], "check": check_id,
                        "message": "Possible fixture issue: scoped contradiction was not reproduced in its answer section.",
                    })
    return issues


def _matches_for_diagnosis(text: str, patterns: list[str], require_all: bool) -> bool:
    found = [bool(re.search(pattern, text, re.I)) for pattern in patterns]
    return all(found) if require_all else any(found)


def render_report(suite: dict, runs: list[dict], preflight: dict | None = None) -> str:
    aggregate = aggregate_runs(runs)
    counts = aggregate["counts"]
    lines = [
        "# Scientific RAG Evaluation", "",
        f"- Suite: `{suite['name']}`",
        f"- Runs: {len(runs)}",
        f"- PASS: {counts.get('PASS', 0)}",
        f"- PARTIAL: {counts.get('PARTIAL', 0)}",
        f"- FAIL: {counts.get('FAIL', 0)}",
        f"- ERROR: {counts.get('ERROR', 0)}",
        f"- Overall pass rate: {aggregate['pass_rate'] * 100:.1f}%", "",
    ]
    if preflight:
        lines.extend(["## Infrastructure preflight", "", f"- Status: **{preflight.get('status', 'ERROR')}**"])
        for key, value in preflight.items():
            if key != "status":
                lines.append(f"- {key.replace('_', ' ').title()}: {value}")
        lines.append("")
    lines.extend(["## Results", ""])
    for run in runs:
        if len(runs) > 1:
            lines.extend([f"### Run {run['run']}", ""])
        for result in run["results"]:
            lines.extend([f"#### {result['id']} — {result['status']}", ""])
            if result["status"] == "ERROR":
                lines.extend([f"Infrastructure/runtime error: `{result.get('error', '')}`", ""])
                continue
            lines.extend(["Passed:", ""])
            lines.extend(f"- {item}" for item in result.get("passed", []))
            lines.extend(["", "Missing:", ""])
            lines.extend(f"- {item}" for item in result.get("missing", []))
            if not result.get("missing"):
                lines.append("- none")
            lines.extend(["", "Contradictions:", ""])
            lines.extend(f"- {item}" for item in result.get("contradictions", []))
            if not result.get("contradictions"):
                lines.append("- none")
            if result.get("missing") and result.get("score", 0) >= 0.9:
                lines.extend([
                    "", "Possible fixture issue:", "",
                    "- Review narrowly missed patterns against the saved answer; do not change expectations automatically.",
                ])
            lines.append("")
    if len(runs) > 1:
        lines.extend([
            "## Stability", "",
            f"- Stable passes: {aggregate['stable_passes']}",
            f"- Flaky questions: {', '.join(aggregate['flaky_questions']) or 'none'}",
            f"- Persistent failures: {', '.join(aggregate['persistent_failures']) or 'none'}", "",
        ])
        for row in aggregate["cases"]:
            lines.append(f"- {row['id']}: {', '.join(row['statuses'])}; Stable: {'YES' if row['stable'] else 'NO'}")
    fixture_issues = diagnose_results(suite, runs)
    if fixture_issues:
        lines.extend(["", "## Possible fixture issues", ""])
        lines.extend(
            f"- Run {row['run']} {row['id']} / {row['check']}: {row['message']}"
            for row in fixture_issues
        )
    lines.extend([
        "", "> Evaluation fixtures are contracts. A suspected expectation error is reported as a possible fixture issue and is never rewritten automatically.",
    ])
    return "\n".join(lines).rstrip() + "\n"


def write_artifacts(output: Path, suite: dict, runs: list[dict], preflight: dict | None = None) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    (output / "debug").mkdir()
    answers = {
        "suite": suite["name"],
        "generated_at": datetime.now().astimezone().isoformat(),
        "runs": runs,
    }
    aggregate = aggregate_runs(runs)
    fixture_issues = diagnose_results(suite, runs)
    scores = {
        "suite": suite["name"], "preflight": preflight or {},
        "aggregate": aggregate, "possible_fixture_issues": fixture_issues,
        "runs": [{
            "run": run["run"], "duration_seconds": run["duration_seconds"],
            "results": [{key: result.get(key) for key in (
                "id", "status", "score", "passed", "missing", "contradictions", "hard_failures", "error"
            ) if key in result} for result in run["results"]],
        } for run in runs],
    }
    (output / "answers.json").write_text(json.dumps(answers, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    (output / "answers.md").write_text(_answer_markdown(runs), encoding="utf-8")
    (output / "scores.json").write_text(json.dumps(scores, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    (output / "report.md").write_text(render_report(suite, runs, preflight), encoding="utf-8")
    for run in runs:
        run_folder = output / "debug" / f"run_{run['run']:02d}"
        run_folder.mkdir()
        for result in run["results"]:
            (run_folder / f"{result['id']}.json").write_text(
                json.dumps(result, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8",
            )
    return scores
