"""Grounded metric semantics and conservative numeric trend classification."""

from __future__ import annotations

import math
import json
import re


def infer_metric_semantics(text: str) -> dict[str, dict]:
    evidence = re.sub(r"\s+", " ", str(text or ""))
    semantics: dict[str, dict] = {}
    if re.search(r"\bRDM\b", evidence) and re.search(
        r"RDM.{0,500}(?:difference|−|-).{0,120}(?:Ana|analytic|reference)",
        evidence, re.I,
    ):
        semantics["RDM"] = {
            "objective": "target_value", "target_value": 0.0,
            "comparison": "smaller absolute deviation from 0 is better",
            "grounding": "explicit metric definition",
        }
    if re.search(r"\bMAG\b", evidence) and re.search(
        r"(?:Magnitude ratio|MAG).{0,500}(?:Num|numerical).{0,100}(?:Ana|analytical|reference)",
        evidence, re.I,
    ):
        semantics["MAG"] = {
            "objective": "target_value", "target_value": 1.0,
            "comparison": "smaller absolute deviation from 1 is better",
            "grounding": "explicit numerical/reference ratio definition",
        }
    return semantics


def classify_numeric_trend(values: list[float], tolerance: float = 1e-12) -> str:
    """Classify all adjacent differences; endpoints alone never establish a trend."""
    numeric = [float(value) for value in values if isinstance(value, (int, float)) and math.isfinite(float(value))]
    if len(numeric) != len(values) or len(numeric) < 2:
        return "insufficient precision"
    differences = [right - left for left, right in zip(numeric, numeric[1:])]
    signs = [1 if delta > tolerance else -1 if delta < -tolerance else 0 for delta in differences]
    nonzero = [sign for sign in signs if sign]
    if not nonzero:
        return "constant"
    if all(sign > 0 for sign in nonzero) and len(nonzero) == len(signs):
        return "monotonic increasing"
    if all(sign < 0 for sign in nonzero) and len(nonzero) == len(signs):
        return "monotonic decreasing"
    positive, negative = nonzero.count(1), nonzero.count(-1)
    if positive and not negative:
        return "broadly increasing with plateaus"
    if negative and not positive:
        return "broadly decreasing with plateaus"
    if positive >= 3 * negative:
        return "broadly increasing with exceptions"
    if negative >= 3 * positive:
        return "broadly decreasing with exceptions"
    return "non-monotonic"


def best_by_metric(values: dict[str, float], semantics: dict) -> str | None:
    if not values or semantics.get("objective") == "unknown":
        return None
    objective = semantics.get("objective")
    if objective == "minimize":
        return min(values, key=values.get)
    if objective == "maximize":
        return max(values, key=values.get)
    if objective == "target_value" and isinstance(semantics.get("target_value"), (int, float)):
        target = float(semantics["target_value"])
        return min(values, key=lambda key: abs(float(values[key]) - target))
    return None


def grounded_table_trends(evidence_text: str) -> dict[str, dict]:
    """Read locally extracted statistical-table rows embedded in graph evidence."""
    marker = "[STRUCTURED STATISTICAL TABLE]"
    _, found, remainder = str(evidence_text or "").partition(marker)
    if not found or (start := remainder.find("{")) < 0:
        return {}
    try:
        table, _ = json.JSONDecoder().raw_decode(remainder[start:])
    except (json.JSONDecodeError, TypeError):
        return {}
    columns, rows = table.get("columns", []), table.get("rows", [])
    caption = str(evidence_text).split(marker, 1)[0].casefold()
    orientation_match = re.search(r"\[target figure orientation:\s*(radial|tangential)\]", caption)
    orientation = (
        orientation_match.group(1) if orientation_match
        else "radial" if "radial dipole" in caption
        else "tangential" if "tangential dipole" in caption else ""
    )
    indices = [index for index, column in enumerate(columns) if index >= 2 and (not orientation or orientation in str(column).casefold())]
    trends = {}
    for row in rows:
        if not isinstance(row, list) or len(row) != len(columns) or len(row) < 3:
            continue
        values = []
        for index in indices:
            match = re.match(r"\s*([-+]?\d+(?:\.\d+)?)", str(row[index] or ""))
            if not match:
                values = []
                break
            values.append(float(match.group(1)))
        method = str(row[1]) if len(row) > 1 else ""
        if not values or "p-value" in method.casefold():
            continue
        labels = [str(columns[index]).split("/")[-1].strip() for index in indices]
        metric = str(row[0])
        trends[f"{metric}|{method}"] = {
            "metric": metric, "series": method, "orientation": orientation or "all",
            "classification": classify_numeric_trend(values), "values": values,
            "positions": labels, "maximum_position": labels[values.index(max(values))],
            "minimum_position": labels[values.index(min(values))],
            "source": "statistical_table",
        }
    return trends
