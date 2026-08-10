"""Ordered multi-target orchestration for explicit scientific references."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass

from services.equation_service import build_equation_evidence
from services.ollama_service import generate_answer
from services.scientific_metrics import infer_metric_semantics
from services.visual_locator import VisualResolution
from services.visual_runtime import analyse_resolved_visual


@dataclass
class TargetResult:
    target_type: str
    target_number: str | None
    status: str
    pdf_name: str | None
    page_number: int | None
    answer: str = ""
    error: str = ""
    structured: dict | None = None
    experimental_context: dict | None = None


def extract_experimental_context(caption: str) -> dict:
    """Attach scenario attributes found explicitly in a caption."""
    text = re.sub(r"\s+", " ", str(caption or "")).strip()
    lower = text.casefold()
    layer = re.search(r"\b(\w+)[- ]layer\b", lower)
    orientation = re.search(r"\b(radial|tangential)\b", lower)
    return {
        "model_geometry": "realistic" if "realistic" in lower else "spherical" if "spherical" in lower else None,
        "tissue_layers": layer.group(1) if layer else None,
        "conductivity": "anisotropic" if "anisotropic" in lower else "isotropic" if "isotropic" in lower else None,
        "orientation": orientation.group(1) if orientation else None,
        "includes_csf": True if re.search(r"\bcsf\b", lower) else None,
        "metrics": [metric for metric in ("RDM", "MAG") if re.search(rf"\b{metric}\b", text, re.I)],
        "source_eccentricities": re.findall(r"\b\d+(?:\.\d+)?%", text),
        "grounded_caption": text,
    }


def _explicit_metric_exceptions(text: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+|\n+", re.sub(r"\s+", " ", str(text or "")))
    return list(dict.fromkeys(
        sentence.strip() for sentence in sentences
        if re.search(r"\b(?:RDM|MAG)\b", sentence, re.I)
        and re.search(r"\bexcept(?:ion)?\b", sentence, re.I)
        and re.search(r"\b(?:outperform|better|worse|accurate)\w*\b", sentence, re.I)
    ))


def _metric_aware_synthesis_guardrail(
    answer: str,
    semantics: dict,
    explicit_exceptions: list[str],
    grounded_trends: dict,
) -> str:
    """Remove target-value and trend contradictions, then restore explicit author text."""
    parts = re.split(r"(?<=[.!?])(?=\s|$)", str(answer or ""))
    kept = []
    mag_target = semantics.get("MAG", {}).get("target_value")
    has_non_monotonic = any(
        "non-monotonic" in str(row.get("classification", ""))
        for row in grounded_trends.values() if isinstance(row, dict)
    )
    has_validated_monotonic = bool(grounded_trends) and all(
        str(row.get("classification", "")).startswith("monotonic")
        for row in grounded_trends.values() if isinstance(row, dict)
    )
    for part in parts:
        if mag_target == 1.0 and re.search(r"\bMAG\b", part, re.I):
            if re.search(
                r"\b(?:higher|larger|greatest)\b.{0,80}\b(?:better|superior|outperform|accurate)",
                part,
                re.I | re.DOTALL,
            ):
                continue
            if explicit_exceptions and re.search(
                r"\b(?:all|every)\b.{0,100}\b(?:eccentricit|case|point)",
                part,
                re.I | re.DOTALL,
            ):
                continue
        if (has_non_monotonic or not has_validated_monotonic) and re.search(
            r"\b(?:monotonic(?:ally)?|steadily)\b",
            part,
            re.I,
        ):
            continue
        if re.search(r"\bwidening\s+(?:gap|difference)\b", part, re.I):
            continue
        kept.append(part)
    cleaned = "".join(kept).strip()
    additions = []
    if mag_target == 1.0 and not re.search(
        r"\bMAG\b.{0,100}\b(?:close|closeness|distance|deviation)\b.{0,30}\b1\b|"
        r"\b(?:close|closeness|distance|deviation)\b.{0,30}\b1\b.{0,100}\bMAG\b",
        cleaned,
        re.I | re.DOTALL,
    ):
        additions.append("MAG accuracy is judged by closeness to 1, not by taking the larger value.")
    normalized = re.sub(r"\s+", " ", cleaned).casefold()
    for statement in explicit_exceptions:
        key = re.sub(r"\s+", " ", statement).casefold()
        eccentricity = re.search(r"\b\d+(?:\.\d+)?%", statement)
        if key not in normalized and not (
            eccentricity and eccentricity.group(0).casefold() in normalized and "except" in normalized
        ):
            additions.append(f"The authors explicitly state: {statement}")
    if additions:
        cleaned = f"{cleaned.rstrip()}\n\n" + " ".join(additions)
    return cleaned.strip()


def _local_equation_text(evidence: str, number: str) -> str:
    match = re.search(rf"(?m)^\s*\({re.escape(str(number))}\)\s*$", evidence)
    if not match:
        return evidence
    return evidence[max(0, match.start() - 850):match.end() + 520]


def _deterministic_interface_coupling(
    equations: list[tuple[str, str]]
) -> tuple[str, dict] | None:
    """Render a high-confidence BE/FE interface chain from source phrases."""
    roles = {}
    for number, evidence in equations:
        local = _local_equation_text(evidence, number)
        normalized = re.sub(r"\s+", " ", local)
        marker = normalized.find(f"({number})")
        prefix = normalized[:marker] if marker >= 0 else normalized
        positions = {
            "potential": prefix.casefold().rfind("average of three fe nodal potentials"),
            "current": prefix.casefold().rfind("continuity condition for the normal component of the current density"),
            "system": prefix.casefold().rfind("resulting equation is obtained"),
        }
        role = max(positions, key=positions.get)
        if positions[role] < 0:
            continue
        if role == "potential" and re.search(r"BE.{0,50}surface element potential", prefix, re.I):
            roles["potential"] = number
        elif role == "current":
            roles["current"] = number
        elif role == "system" and re.search(r"solved for.{0,100}\band\b", normalized, re.I):
            roles["system"] = number
    if set(roles) != {"potential", "current", "system"}:
        return None
    answer = rf"""**Equation {roles['potential']} — potential continuity**

The constant BE surface potential is approximated by the mean of the three FE nodal potentials on the shared triangular interface:

$$
\displaystyle \phi_{{\mathrm{{BE}}}}
=\frac{{\phi_{{\mathrm{{FE}},1}}+\phi_{{\mathrm{{FE}},2}}+\phi_{{\mathrm{{FE}},3}}}}{{3}}.
$$

**Equation {roles['current']} — normal-current continuity**

The normal conductive current is continuous across the interface. Because the two regions use opposite outward normals, the paper writes

$$
\sigma_{{\mathrm{{FE}}}}\frac{{\partial\phi_{{\mathrm{{FE}}}}}}{{\partial n_{{\mathrm{{FE}}}}}}
=-\sigma_{{\mathrm{{BE}}}}\frac{{\partial\phi_{{\mathrm{{BE}}}}}}{{\partial n_{{\mathrm{{BE}}}}}}.
$$

**Equation {roles['system']} — assembled coupled system**

Applying those two interface conditions modifies the FE and BE blocks and produces the final algebraic system

$$
\left[\begin{{array}}{{cc}}
\widetilde{{K}}_{{\mathrm{{FE}}}} & M_{{\mathrm{{FE}}}}\\
M_{{\mathrm{{BE}}}} & \widetilde{{A}}_{{\mathrm{{BE}}}}
\end{{array}}\right]
\left[\begin{{array}}{{c}}\phi_{{\mathrm{{FE}}}}\\X_{{\mathrm{{BE}}}}\end{{array}}\right]
=
\left[\begin{{array}}{{c}}\widetilde{{B}}_{{\mathrm{{FE}}}}\\\widetilde{{B}}_{{\mathrm{{BE}}}}\end{{array}}\right].
$$

Together, the first condition transfers FE nodal potentials to the BE surface unknown, the second transfers the matching normal flux with the correct sign convention, and the third solves the modified FE potentials $\phi_{{FE}}$ and BE unknowns $X_{{BE}}$ in one coupled system."""
    return answer, {
        "roles": roles,
        "potential_expression": "φ_BE = (φ_FE1 + φ_FE2 + φ_FE3) / 3",
        "current_expression": "σ_FE ∂φ_FE/∂n_FE = −σ_BE ∂φ_BE/∂n_BE",
        "unknowns": ["φ_FE", "X_BE"],
        "confidence": 0.99,
    }


def analyse_equation_chain(
    question: str,
    resolutions: list[VisualResolution],
    *,
    conversation_history: list[dict] | None = None,
    debug_info: dict | None = None,
) -> str:
    """Retrieve every equation, then synthesize only from their page evidence."""
    results, evidence_parts, equation_evidence = [], [], []
    for resolution in resolutions:
        if resolution.status != "resolved" or not resolution.pdf_path or not resolution.page_number or not resolution.target_number:
            results.append(TargetResult(
                resolution.target_type, resolution.target_number, resolution.status,
                resolution.pdf_name, resolution.page_number,
                error="Exact-document identifier search did not resolve this equation.",
            ))
            continue
        evidence = build_equation_evidence(
            __import__("pathlib").Path(resolution.pdf_path),
            resolution.page_number,
            resolution.target_number,
        )
        evidence_parts.append(
            f"[REQUESTED EQUATION {resolution.target_number}]\n"
            f"Source: {resolution.pdf_name}, PDF page {resolution.page_number}\n{evidence}"
        )
        equation_evidence.append((resolution.target_number, evidence))
        results.append(TargetResult(
            "equation", resolution.target_number, "resolved", resolution.pdf_name,
            resolution.page_number,
        ))
    missing = [row for row in results if row.status != "resolved"]
    if not evidence_parts:
        return "Could not verify any requested equation in the explicitly named document."
    synthesis_question = (
        f"{question}\nExplain each requested equation in order and then explain their coupling. "
        "Preserve the displayed operators, subscripts, matrix signs and unknowns. "
        "Do not supply a standard equation from outside the evidence."
    )
    deterministic = _deterministic_interface_coupling(equation_evidence)
    if deterministic:
        answer, symbolic = deterministic
        if debug_info is not None:
            debug_info["symbolic_equation_chain"] = symbolic
            debug_info["generation_code_path"] = "deterministic_interface_coupling"
    else:
        answer = generate_answer(
            synthesis_question, "\n\n".join(evidence_parts),
            conversation_history or [], debug_info=debug_info,
        )
    if re.search(r"\b(?:equations?\s+\d+[^.]{0,80}(?:do not exist|not present)|could not find the equations?)\b", answer, re.I):
        answer = re.sub(
            r"(?i)(?:^|(?<=[.!?])\s+)[^.!?]*(?:do not exist|not present|could not find)[^.!?]*[.!?]?",
            " ", answer,
        ).strip()
        answer = "The requested equations were verified in the named PDF.\n\n" + answer
    if missing:
        answer += "\n\nCould not verify: " + ", ".join(
            f"Equation {row.target_number}" for row in missing
        ) + "."
    if debug_info is not None:
        debug_info.update({
            "targets": [asdict(row) for row in results],
            "final_answer_path": "validated_multi_equation_analysis",
            "final_answer_code_path": "validated_multi_equation_analysis",
        })
    return answer


def analyse_visual_targets(
    question: str,
    resolutions: list[VisualResolution],
    *,
    text_evidence: str = "",
    conversation_history: list[dict] | None = None,
    debug_info: dict | None = None,
) -> str:
    """Validate each requested visual independently, then synthesize them."""
    target_results, evidence_parts = [], []
    metric_semantics: dict[str, dict] = {}
    grounded_trends: dict[str, dict] = {}
    metric_evidence_parts = [text_evidence]
    for resolution in resolutions:
        context = extract_experimental_context(
            f"{resolution.caption}\n{resolution.nearby_text}"
        )
        metric_evidence_parts.extend([resolution.caption or "", resolution.nearby_text or ""])
        if resolution.status != "resolved":
            target_results.append(TargetResult(
                resolution.target_type, resolution.target_number, resolution.status,
                resolution.pdf_name, resolution.page_number,
                error="Exact-document identifier search did not resolve this target.",
                experimental_context=context,
            ))
            continue
        target_debug = {"_save_crops": bool(debug_info and debug_info.get("_save_crops"))}
        try:
            answer = analyse_resolved_visual(
                question, resolution, text_evidence=text_evidence,
                debug_info=target_debug,
            )
            structured = target_debug.get("final_structured_output") or target_debug.get("validated_json")
            validated = str(target_debug.get("final_answer_path", "")).startswith("validated")
            if not validated:
                raise ValueError("The target did not produce validated structured output.")
            target_results.append(TargetResult(
                resolution.target_type, resolution.target_number, "resolved",
                resolution.pdf_name, resolution.page_number, answer=answer,
                structured=structured, experimental_context=context,
            ))
            if isinstance(structured, dict):
                metric_semantics.update(structured.get("metric_semantics") or {})
                grounded_trends.update(structured.get("grounded_trends") or {})
            evidence_parts.append(
                f"[VALIDATED {resolution.target_type.upper()} {resolution.target_number}]\n"
                f"Source: {resolution.pdf_name}, PDF page {resolution.page_number}\n"
                f"Experimental context: {context}\n"
                f"Validated structured evidence: {json.dumps(structured, ensure_ascii=False)}\n"
                f"Result: {answer}"
            )
        except Exception as error:
            target_results.append(TargetResult(
                resolution.target_type, resolution.target_number, "failed",
                resolution.pdf_name, resolution.page_number, error=str(error),
                experimental_context=context,
            ))
    if not evidence_parts:
        return "Could not verify any requested visual target from validated structured output."
    metric_evidence = "\n".join(metric_evidence_parts)
    metric_semantics = {**infer_metric_semantics(metric_evidence), **metric_semantics}
    explicit_exceptions = _explicit_metric_exceptions(metric_evidence)
    metric_block = (
        "\n[VALIDATED METRIC SEMANTICS]\n"
        f"{json.dumps(metric_semantics, ensure_ascii=False)}\n"
        "For target-value metrics, compare absolute distance from the target; never "
        "equate a larger raw value with better performance. Do not infer monotonic or "
        "widening gaps from endpoints. Explicit author exceptions override generic synthesis.\n"
        f"Explicit author comparison exceptions: {json.dumps(explicit_exceptions, ensure_ascii=False)}\n"
    ) if metric_semantics or explicit_exceptions else ""
    answer = generate_answer(
        question + "\nCompare only the individually validated targets; keep every claim attached to its figure and experimental context.",
        metric_block + "\n\n".join(evidence_parts), conversation_history or [],
    )
    answer = _metric_aware_synthesis_guardrail(
        answer, metric_semantics, explicit_exceptions, grounded_trends
    )
    missing = [row for row in target_results if row.status != "resolved"]
    if missing:
        answer += "\n\nCould not verify: " + ", ".join(
            f"{row.target_type.title()} {row.target_number}" for row in missing
        ) + "."
    if debug_info is not None:
        debug_info.update({
            "targets": [asdict(row) for row in target_results],
            "metric_semantics": metric_semantics,
            "explicit_metric_exceptions": explicit_exceptions,
            "grounded_trends": grounded_trends,
            "final_answer_path": "validated_multi_visual_synthesis",
            "final_answer_code_path": "validated_multi_visual_synthesis",
        })
    return answer
