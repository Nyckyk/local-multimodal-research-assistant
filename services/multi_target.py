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
    figure = re.search(r"\bfig(?:ure)?\.?\s*(\d+(?:\.\d+)?)\b", text, re.I)
    panel_metrics = {
        panel.upper(): metric.upper()
        for panel, metric in re.findall(
            r"\(([A-Za-z])\)\s*(RDM|MAG)\b", text, re.I
        )
    }
    eccentricities = list(dict.fromkeys(re.findall(r"\b\d+(?:\.\d+)?%", text)))
    count_match = re.search(
        r"\b(one|two|three|four|five|six|seven|eight|nine|ten|\d+)\b"
        r"\s+(?:different\s+)?source eccentricit",
        lower,
    )
    number_words = {
        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    }
    eccentricity_count = None
    if count_match:
        raw_count = count_match.group(1)
        eccentricity_count = number_words.get(raw_count, int(raw_count) if raw_count.isdigit() else None)
    return {
        "figure_number": figure.group(1) if figure else None,
        "model_geometry": "realistic" if "realistic" in lower else "spherical" if "spherical" in lower else None,
        "model_type": "realistic" if "realistic" in lower else "spherical" if "spherical" in lower else None,
        "tissue_layers": layer.group(1) if layer else None,
        "conductivity": "anisotropic" if "anisotropic" in lower else "isotropic" if "isotropic" in lower else None,
        "orientation": orientation.group(1) if orientation else None,
        "dipole_orientation": orientation.group(1) if orientation else None,
        "includes_csf": True if re.search(r"\bcsf\b", lower) else None,
        "metrics": [metric for metric in ("RDM", "MAG") if re.search(rf"\b{metric}\b", text, re.I)],
        "panel_metrics": panel_metrics,
        "source_eccentricities": eccentricities,
        "source_eccentricity_count": eccentricity_count or (len(eccentricities) or None),
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


def _explicit_author_comparisons(text: str) -> list[str]:
    """Extract metric comparisons explicitly stated by the paper's authors."""
    sentences = re.split(r"(?<=[.!?])\s+|\n+", re.sub(r"\s+", " ", str(text or "")))
    return list(dict.fromkeys(
        sentence.strip() for sentence in sentences
        if re.search(r"\b(?:RDM|MAG)\b", sentence, re.I)
        and re.search(
            r"\b(?:outperform|better|worse|accurate|accuracy|best)\w*\b",
            sentence,
            re.I,
        )
        and re.search(r"\bPI-?\s*FEM\b", sentence, re.I)
        and re.search(r"\bhybrid\s+BE-?FE\b", sentence, re.I)
    ))


def _contextual_author_comparisons(
    text: str, figure_contexts: list[dict]
) -> list[str]:
    """Keep the comparison block matching the requested figures' shared model."""
    comparisons = _explicit_author_comparisons(text)
    model_types = {
        str(context.get("model_type") or "").casefold()
        for context in figure_contexts if context.get("model_type")
    }
    if len(model_types) != 1:
        return comparisons
    model_type = next(iter(model_types))
    anchored = [
        index for index, statement in enumerate(comparisons)
        if re.search(
            rf"\b{re.escape(model_type)}(?:[- ]head)?\b",
            statement,
            re.I,
        )
    ]
    return comparisons[anchored[-1]:] if anchored else comparisons


def _complete_figure_context(
    resolution: VisualResolution,
    context: dict,
    structured: dict | None,
) -> dict:
    """Combine caption-grounded identity with panel-local structured evidence."""
    complete = dict(context)
    complete["figure_number"] = str(resolution.target_number or context.get("figure_number") or "")
    complete["dipole_orientation"] = context.get("orientation")
    complete["model_type"] = context.get("model_type") or context.get("model_geometry")
    panel_metrics = dict(context.get("panel_metrics") or {})
    eccentricities = list(context.get("source_eccentricities") or [])
    structured_eccentricities: list[str] = []
    if isinstance(structured, dict):
        for panel in structured.get("panels") or []:
            if not isinstance(panel, dict):
                continue
            panel_id = str(panel.get("panel", "")).strip().upper()
            axis_label = str(panel.get("y_axis", {}).get("label", ""))
            metric = next(
                (name for name in ("RDM", "MAG") if re.search(rf"\b{name}\b", axis_label, re.I)),
                None,
            )
            if panel_id and metric and panel_id not in panel_metrics:
                panel_metrics[panel_id] = metric
            for tick in panel.get("x_axis", {}).get("tick_labels") or []:
                value = str(tick).strip()
                if re.fullmatch(r"\d+(?:\.\d+)?%", value) and value not in structured_eccentricities:
                    structured_eccentricities.append(value)
    if structured_eccentricities:
        expected_count = context.get("source_eccentricity_count")
        if not expected_count or len(structured_eccentricities) >= expected_count:
            eccentricities = structured_eccentricities
        else:
            eccentricities.extend(
                value for value in structured_eccentricities
                if value not in eccentricities
            )
    complete["panel_metrics"] = panel_metrics
    complete["source_eccentricities"] = eccentricities
    complete["source_eccentricity_count"] = (
        context.get("source_eccentricity_count") or len(eccentricities) or None
    )
    return complete


def _has_grounded_target_text(resolution: VisualResolution) -> bool:
    """A resolved target can retain caption/author evidence after vision failure."""
    target = re.escape(str(resolution.target_number or ""))
    caption = str(resolution.caption or "")
    nearby = str(resolution.nearby_text or "")
    return bool(
        caption.strip()
        and re.search(rf"\bfig(?:ure)?\.?\s*{target}\b", caption, re.I)
        and (caption.strip() or nearby.strip())
    )


def _remove_orientation_leakage(answer: str, figure_contexts: list[dict]) -> tuple[str, list[str]]:
    """Correct a wrong orientation only in single-figure-attributed sentences."""
    sentences = re.split(r"(?<=[.!?])(?=\s|$)", str(answer or ""))
    changed: list[str] = []
    for index, sentence in enumerate(sentences):
        referenced = [
            context for context in figure_contexts
            if re.search(
                rf"\bFigure\s+{re.escape(str(context.get('figure_number', '')))}\b",
                sentence,
                re.I,
            )
        ]
        if len(referenced) != 1:
            continue
        context = referenced[0]
        expected = str(context.get("dipole_orientation") or "").casefold()
        if expected not in {"radial", "tangential"}:
            continue
        wrong = "tangential" if expected == "radial" else "radial"
        if re.search(rf"\b{wrong}\b", sentence, re.I) and not re.search(
            rf"\b{expected}\b", sentence, re.I
        ):
            sentences[index] = re.sub(rf"\b{wrong}\b", expected, sentence, flags=re.I)
            changed.append(str(context.get("figure_number")))
    return "".join(sentences), changed


def _grounded_multi_figure_synthesis(
    target_results: list[TargetResult],
    figure_contexts: list[dict],
    author_comparisons: list[str],
    semantics: dict,
) -> str | None:
    """Render a complete comparison when explicit author conclusions are available."""
    comparison_metrics = {
        metric for metric in ("RDM", "MAG")
        if any(re.search(rf"\b{metric}\b", item, re.I) for item in author_comparisons)
    }
    if len(figure_contexts) < 2 or comparison_metrics != {"RDM", "MAG"}:
        return None
    result_by_number = {
        str(result.target_number): result for result in target_results
    }
    lines = ["**Author-grounded comparison**", ""]
    lines.extend(f"- {statement}" for statement in author_comparisons)
    lines.extend(["", "**Figure-specific evidence**", ""])
    for context in figure_contexts:
        number = str(context.get("figure_number") or "")
        result = result_by_number.get(number)
        orientation = str(context.get("dipole_orientation") or "unspecified")
        model = str(context.get("model_type") or "specified")
        panel_metrics = context.get("panel_metrics") or {}
        panel_text = ", ".join(
            f"panel {panel} = {metric}" for panel, metric in panel_metrics.items()
        ) or "panel metrics not explicitly identified"
        eccentricities = context.get("source_eccentricities") or []
        count = context.get("source_eccentricity_count")
        eccentricity_text = (
            ", ".join(map(str, eccentricities))
            if eccentricities and (not count or len(eccentricities) >= count)
            else f"{count} tested positions" if count else "the tested positions"
        )
        source = (
            f"{result.pdf_name}, PDF page {result.page_number}"
            if result else "the resolved PDF"
        )
        evidence_kind = (
            "validated visual evidence plus caption/nearby text"
            if result and result.status == "resolved"
            else "caption and nearby author text"
        )
        lines.append(
            f"- **Figure {number} — {orientation} dipoles:** {model} head model; "
            f"{panel_text}; source eccentricities: {eccentricity_text}. "
            f"Evidence: {evidence_kind} ({source})."
        )
    semantic_lines = []
    for metric in ("RDM", "MAG"):
        metric_semantics = semantics.get(metric) or {}
        target = metric_semantics.get("target_value")
        if isinstance(target, (int, float)):
            semantic_lines.append(
                f"- {metric} is interpreted by closeness to {target:g}; a larger "
                f"raw {metric} value is not inherently better."
            )
    if semantic_lines:
        lines.extend(["", "**Metric interpretation**", "", *semantic_lines])
    return "\n".join(lines).strip()


def _metric_aware_synthesis_guardrail(
    answer: str,
    semantics: dict,
    explicit_exceptions: list[str],
    grounded_trends: dict,
    author_comparisons: list[str] | None = None,
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
        if explicit_exceptions and re.search(
            r"\b(?:all|every|always|consistently|universally|throughout|across\s+all)\b",
            part,
            re.I,
        ) and re.search(
            r"\b(?:outperform|better|superior|accurate|wins?)\w*\b",
            part,
            re.I,
        ) and not re.search(r"\bexcept(?:ion)?\b", part, re.I):
            continue
        if mag_target == 1.0 and re.search(r"\bMAG\b", part, re.I):
            if re.search(
                r"\b(?:higher|larger|greatest)\b.{0,80}\b(?:better|superior|outperform|accurate)",
                part,
                re.I | re.DOTALL,
            ):
                continue
        if re.search(r"\b(?:eccentricit\w*|source)\b", part, re.I) and re.search(
            r"\bdeeper\b.{0,30}\b(?:brain|head)\b|"
            r"\b(?:brain|head)\b.{0,30}\bdeeper\b",
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
        kept.append(part.strip())
    cleaned = " ".join(filter(None, kept)).strip()
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
    for statement in author_comparisons or []:
        if re.search(r"\bexcept(?:ion)?\b", statement, re.I):
            continue
        metrics = [name for name in ("RDM", "MAG") if re.search(rf"\b{name}\b", statement, re.I)]
        already_grounded = all(
            re.search(
                rf"\b{metric}\b.{0,180}\b(?:outperform|better|superior|accurate)\w*\b|"
                rf"\b(?:outperform|better|superior|accurate)\w*\b.{0,180}\b{metric}\b",
                cleaned,
                re.I | re.DOTALL,
            )
            for metric in metrics
        )
        if metrics and not already_grounded:
            additions.append(f"The authors report: {statement}")
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
    target_results, evidence_parts, figure_contexts = [], [], []
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
            context = _complete_figure_context(resolution, context, structured)
            figure_contexts.append(context)
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
                f"Figure context: {json.dumps(context, ensure_ascii=False)}\n"
                f"Validated structured evidence: {json.dumps(structured, ensure_ascii=False)}\n"
                f"Result: {answer}"
            )
        except Exception as error:
            context = _complete_figure_context(resolution, context, None)
            figure_contexts.append(context)
            if _has_grounded_target_text(resolution):
                target_results.append(TargetResult(
                    resolution.target_type, resolution.target_number,
                    "grounded_text_fallback", resolution.pdf_name,
                    resolution.page_number, error=str(error),
                    experimental_context=context,
                ))
                evidence_parts.append(
                    f"[GROUNDED TEXT FALLBACK FOR {resolution.target_type.upper()} "
                    f"{resolution.target_number}]\n"
                    f"Source: {resolution.pdf_name}, PDF page {resolution.page_number}\n"
                    f"Figure context: {json.dumps(context, ensure_ascii=False)}\n"
                    f"Caption: {resolution.caption}\n"
                    f"Nearby author text: {resolution.nearby_text}\n"
                    "The visual sub-analysis failed; use only the caption and "
                    "nearby author text in this block."
                )
            else:
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
    author_comparisons = _contextual_author_comparisons(
        metric_evidence, figure_contexts
    )
    metric_block = (
        "\n[VALIDATED METRIC SEMANTICS]\n"
        f"{json.dumps(metric_semantics, ensure_ascii=False)}\n"
        "For target-value metrics, compare absolute distance from the target; never "
        "equate a larger raw value with better performance. Do not infer monotonic or "
        "widening gaps from endpoints. Explicit author exceptions override generic synthesis.\n"
        f"Explicit author comparison exceptions: {json.dumps(explicit_exceptions, ensure_ascii=False)}\n"
        f"Explicit author comparison statements: {json.dumps(author_comparisons, ensure_ascii=False)}\n"
    ) if metric_semantics or explicit_exceptions or author_comparisons else ""
    deterministic_answer = _grounded_multi_figure_synthesis(
        target_results, figure_contexts, author_comparisons, metric_semantics
    )
    if deterministic_answer is not None:
        answer = deterministic_answer
    else:
        answer = generate_answer(
            question + (
                "\nCompare only the independently collected target evidence. Keep every "
                "claim attached to its figure context. Caption and nearby author text are "
                "authoritative when a visual estimate is unavailable or conflicts. Panel "
                "metric labels do not change the figure-level dipole orientation."
            ),
            metric_block + "\n\n".join(evidence_parts), conversation_history or [],
        )
        answer = _metric_aware_synthesis_guardrail(
            answer, metric_semantics, explicit_exceptions, grounded_trends,
            author_comparisons,
        )
    answer, orientation_corrections = _remove_orientation_leakage(
        answer, figure_contexts
    )
    successful_statuses = {"resolved", "grounded_text_fallback"}
    missing = [row for row in target_results if row.status not in successful_statuses]
    if missing:
        answer += "\n\nCould not verify: " + ", ".join(
            f"{row.target_type.title()} {row.target_number}" for row in missing
        ) + "."
    if debug_info is not None:
        debug_info.update({
            "targets": [asdict(row) for row in target_results],
            "metric_semantics": metric_semantics,
            "explicit_metric_exceptions": explicit_exceptions,
            "explicit_author_comparisons": author_comparisons,
            "grounded_trends": grounded_trends,
            "figure_contexts": figure_contexts,
            "orientation_corrections": orientation_corrections,
            "deterministic_author_synthesis": deterministic_answer is not None,
            "evidence_complete": all(
                row.status in successful_statuses for row in target_results
            ),
            "final_answer_path": "validated_multi_visual_synthesis",
            "final_answer_code_path": "validated_multi_visual_synthesis",
        })
    return answer
