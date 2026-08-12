"""Generic deterministic scoring for scientific RAG answers."""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from typing import Any


_HYPHENS = "\u2010\u2011\u2012\u2013\u2014\u2212"


def normalize_text(value: Any) -> str:
    """Normalize typography while preserving scientific distinctions."""
    text = unicodedata.normalize("NFKC", str(value or ""))
    text = text.translate(str.maketrans({char: "-" for char in _HYPHENS}))
    text = text.replace("β", "beta").replace("Β", "beta")
    text = text.replace("α", "alpha").replace("γ", "gamma").replace("δ", "delta")
    text = text.replace("×", "x")
    text = re.sub(r"\s+", " ", text).strip().casefold()
    return text


def _matches(text: str, patterns: list[str], mode: str = "any") -> bool:
    results = [bool(re.search(pattern, text, re.IGNORECASE)) for pattern in patterns]
    return all(results) if mode == "all" else any(results)


def _numeric_occurrences(text: str) -> list[float]:
    values = []
    for raw in re.findall(r"(?<![A-Za-z])[-+]?\d+(?:,\d{3})*(?:\.\d+)?(?:[eE][-+]?\d+)?", text):
        try:
            values.append(float(raw.replace(",", "")))
        except ValueError:
            pass
    return values


def numeric_match(answer: str, requirement: dict) -> bool:
    """Match a number, optionally within a local scientific context."""
    normalized = normalize_text(answer)
    contexts = requirement.get("context_any") or []
    search_texts = [normalized]
    if contexts:
        search_texts = []
        for pattern in contexts:
            for match in re.finditer(pattern, normalized, re.IGNORECASE):
                start, end = max(0, match.start() - 100), min(len(normalized), match.end() + 100)
                search_texts.append(normalized[start:end])
    expected = float(requirement["value"])
    tolerance = float(requirement.get("tolerance", 0.0))
    if requirement.get("operator") in {"lt", "lte"}:
        phrases = r"(?:less than|under|below|<|at most|no more than)\s*"
        return any(
            re.search(phrases + re.escape(str(requirement["value"])), text)
            for text in search_texts
        )
    if any(any(math.isclose(value, expected, abs_tol=tolerance, rel_tol=0.0) for value in _numeric_occurrences(text)) for text in search_texts):
        return True
    return False


def _document_identity(response: dict) -> str:
    if response.get("selected_document"):
        return str(response["selected_document"])
    resolution = response.get("resolution") or {}
    if resolution.get("pdf_name"):
        return str(resolution["pdf_name"])
    evidence = response.get("evidence") or {}
    if evidence.get("pdf"):
        return str(evidence["pdf"])
    sources = response.get("sources") or []
    names = [str(row.get("pdf", row.get("source", ""))) for row in sources if isinstance(row, dict)]
    return names[0] if names and len({normalize_text(name) for name in names}) == 1 else ""


def _document_key(value: Any) -> str:
    """Treat legacy .txt chunk sources as the corresponding indexed PDF."""
    name = re.sub(r"\.(?:pdf|txt)$", "", str(value or ""), flags=re.IGNORECASE)
    return normalize_text(name)


def _check(check_id: str, passed: bool, *, kind: str, detail: str, weight: float = 1.0, hard: bool = False) -> dict:
    return {
        "id": check_id,
        "kind": kind,
        "passed": bool(passed),
        "detail": detail,
        "weight": float(weight),
        "hard": bool(hard),
    }


def score_case(case: dict, response: dict) -> dict:
    """Score one response without model calls or paper-specific logic."""
    answer = str(response.get("answer") or "")
    text = normalize_text(answer)
    expected = case.get("expect", case)
    checks: list[dict] = []

    expected_document = expected.get("document") or case.get("expected_document")
    if expected_document:
        actual = _document_identity(response)
        checks.append(_check(
            "document_identity", _document_key(actual) == _document_key(expected_document),
            kind="identity", detail=f"expected {expected_document!r}; got {actual!r}", hard=True,
        ))

    target = expected.get("target") or {}
    resolution = response.get("resolution") or {}
    for field, response_field in (("type", "target_type"), ("number", "target_number"), ("page", "page_number"), ("status", "status")):
        if target.get(field) is None:
            continue
        actual, wanted = resolution.get(response_field), target[field]
        checks.append(_check(
            f"target_{field}", normalize_text(actual) == normalize_text(wanted),
            kind="identity", detail=f"expected {field}={wanted!r}; got {actual!r}", hard=True,
        ))

    for term in expected.get("required_terms", []):
        if isinstance(term, str):
            term = {"term": term}
        value = str(term["term"])
        present = normalize_text(value) in text
        checks.append(_check(
            str(term.get("id") or f"term:{value}"), present,
            kind="required_term", detail=f"required term: {value}",
            weight=term.get("weight", 1), hard=term.get("hard", False),
        ))

    pattern_fields = (("required_facts", "fact"), ("required_concepts", "concept"), ("semantic_requirements", "semantic"))
    for field, kind in pattern_fields:
        for index, item in enumerate(expected.get(field, []), 1):
            if isinstance(item, str):
                item = {"id": item, "any_of": [re.escape(normalize_text(item))]}
            patterns = item.get("all_of") or item.get("any_of") or []
            mode = "all" if item.get("all_of") else "any"
            passed = _matches(text, patterns, mode) if patterns else False
            checks.append(_check(
                str(item.get("id") or f"{kind}_{index}"), passed,
                kind=kind, detail=str(item.get("description") or f"{mode}: {patterns}"),
                weight=item.get("weight", 1), hard=item.get("hard", False),
            ))

    for index, item in enumerate(expected.get("numeric_requirements", []), 1):
        checks.append(_check(
            str(item.get("id") or f"numeric_{index}"), numeric_match(answer, item),
            kind="numeric", detail=f"required numeric value {item.get('value')}",
            weight=item.get("weight", 1.5), hard=item.get("hard", False),
        ))

    forbidden = []
    for field in ("forbidden_terms", "forbidden_claims"):
        for index, item in enumerate(expected.get(field, []), 1):
            if isinstance(item, str):
                item = {"id": item, "any_of": [re.escape(normalize_text(item))]}
            scope = text
            if item.get("scope_after"):
                scope_match = re.search(item["scope_after"], text, re.I)
                scope = text[scope_match.end():] if scope_match else ""
                if item.get("scope_before") and scope:
                    before = re.search(item["scope_before"], scope, re.I)
                    scope = scope[:before.start()] if before else scope
            patterns = item.get("all_of") or item.get("any_of") or []
            mode = "all" if item.get("all_of") else "any"
            contradicted = _matches(scope, patterns, mode) if patterns else False
            check = _check(
                str(item.get("id") or f"forbidden_{index}"), not contradicted,
                kind="forbidden", detail=str(item.get("description") or f"forbidden {mode}: {patterns}"),
                weight=item.get("weight", 3), hard=item.get("hard", True),
            )
            checks.append(check)
            if contradicted:
                forbidden.append(check["id"])

    debug = response.get("debug") or {}
    debug_expect = expected.get("debug") or {}
    if debug_expect.get("allowed_code_paths"):
        actual = str(debug.get("final_answer_code_path") or "")
        checks.append(_check(
            "debug_code_path", actual in debug_expect["allowed_code_paths"],
            kind="debug", detail=f"final code path: {actual!r}", hard=debug_expect.get("code_path_hard", False),
        ))
    if "max_missing_slots" in debug_expect:
        missing = debug.get("missing_answer_slots") or []
        checks.append(_check(
            "debug_missing_slots", len(missing) <= int(debug_expect["max_missing_slots"]),
            kind="debug", detail=f"missing answer slots: {missing}",
        ))
    if "consistency_errors" in debug_expect:
        errors = debug.get("final_evidence_consistency_errors") or []
        wanted = debug_expect["consistency_errors"]
        checks.append(_check(
            "debug_consistency_errors", len(errors) == int(wanted),
            kind="debug", detail=f"final evidence consistency errors: {errors}", hard=True,
        ))

    provenance = expected.get("expected_provenance") or []
    provenance_text = normalize_text(
        f"{answer} {response.get('sources', [])} {debug.get('final_answer_evidence', '')}"
    )
    for item in provenance:
        patterns = item.get("all_of") or item.get("any_of") or []
        passed = _matches(provenance_text, patterns, "all" if item.get("all_of") else "any")
        checks.append(_check(
            str(item.get("id") or item.get("domain") or "provenance"), passed,
            kind="provenance", detail=str(item.get("description") or patterns),
            weight=item.get("weight", 2), hard=item.get("hard", False),
        ))

    for domain in expected.get("expected_domains", []):
        item = domain if isinstance(domain, dict) else {"id": str(domain), "any_of": [re.escape(normalize_text(domain))]}
        patterns = item.get("all_of") or item.get("any_of") or []
        passed = _matches(provenance_text, patterns, "all" if item.get("all_of") else "any")
        checks.append(_check(
            str(item.get("id") or "domain"), passed, kind="domain",
            detail=str(item.get("description") or patterns), weight=item.get("weight", 1.5),
            hard=item.get("hard", False),
        ))

    total = sum(row["weight"] for row in checks) or 1.0
    earned = sum(row["weight"] for row in checks if row["passed"])
    ratio = earned / total
    failed_hard = [row for row in checks if row["hard"] and not row["passed"]]
    failed = [row for row in checks if not row["passed"]]
    if not failed:
        status = "PASS"
    elif failed_hard or ratio < float(expected.get("partial_threshold", 0.6)):
        status = "FAIL"
    else:
        status = "PARTIAL"
    return {
        "status": status,
        "score": round(ratio, 4),
        "checks": checks,
        "passed": [row["id"] for row in checks if row["passed"]],
        "missing": [row["id"] for row in checks if not row["passed"] and row["kind"] != "forbidden"],
        "contradictions": forbidden,
        "hard_failures": [row["id"] for row in failed_hard],
    }


def aggregate_runs(runs: list[dict]) -> dict:
    """Summarize repeated per-case statuses and stability."""
    grouped: dict[str, list[str]] = {}
    for run in runs:
        for result in run.get("results", []):
            grouped.setdefault(result["id"], []).append(result["status"])
    cases = []
    for case_id, statuses in grouped.items():
        stable = len(set(statuses)) == 1
        cases.append({
            "id": case_id,
            "statuses": statuses,
            "stable": stable,
            "stable_pass": stable and statuses[0] == "PASS",
            "flaky": not stable,
            "persistent_failure": stable and statuses[0] in {"FAIL", "ERROR"},
        })
    counts = Counter(status for values in grouped.values() for status in values)
    scientific = counts["PASS"] + counts["PARTIAL"] + counts["FAIL"]
    return {
        "cases": cases,
        "stable_passes": sum(row["stable_pass"] for row in cases),
        "flaky_questions": [row["id"] for row in cases if row["flaky"]],
        "persistent_failures": [row["id"] for row in cases if row["persistent_failure"]],
        "counts": dict(counts),
        "pass_rate": round(counts["PASS"] / scientific, 4) if scientific else 0.0,
    }
