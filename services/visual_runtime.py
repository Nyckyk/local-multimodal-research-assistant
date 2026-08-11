"""Shared resolved-visual runtime used by Streamlit, tests and diagnostics."""

from __future__ import annotations

import json
from pathlib import Path

from services.ollama_service import generate_answer
from services.scientific_evidence import (
    apply_grounded_slot_fallback,
    authoritative_panel_role_map,
    caption_results_fallback,
    contradiction_check_condition_prose,
    enforce_authoritative_panel_prose,
    figure_local_evidence,
    merge_mixed_figure_with_caption,
    remove_unsupported_acronym_expansions,
)
from services.structured_vision import format_structured_result, validate_typed_response
from services.visual_locator import VisualResolution, resolved_analysis_question
from services.vision_service import analyse_pdf_page


def analyse_resolved_visual(
    question: str,
    resolution: VisualResolution,
    *,
    text_evidence: str = "",
    debug_info: dict | None = None,
) -> str:
    """Run the complete analyse/validate/repair/render path after resolution."""
    if resolution.status != "resolved":
        raise ValueError("Visual analysis requires a resolved target.")
    if not resolution.pdf_path or not resolution.page_number:
        raise ValueError("Resolved visual target is missing its PDF path or page.")

    analysis_question = resolved_analysis_question(question, resolution)
    local_evidence = figure_local_evidence(
        Path(resolution.pdf_path),
        int(resolution.page_number),
        resolution.caption_page_number,
        resolution.target_number,
        resolution.full_caption or resolution.caption,
        question,
    )
    composed_evidence = (
        f"{local_evidence['evidence_text']}\n\nRETRIEVED RAG EVIDENCE:\n{text_evidence}"
    ).strip()
    try:
        answer = analyse_pdf_page(
            pdf_path=Path(resolution.pdf_path),
            page_number=int(resolution.page_number),
            question=analysis_question,
            debug_info=debug_info,
            text_evidence=composed_evidence,
            visual_type_override=resolution.visual_type,
        )
    except Exception as error:
        if debug_info is not None:
            debug_info.update({
                "resolved_visual_target": resolution.to_dict(),
                "runtime_analysis_question": analysis_question,
                "runtime_error": f"{type(error).__name__}: {error}",
            })
        raise

    if debug_info is not None:
        debug_info["figure_local_evidence"] = {
            key: value for key, value in local_evidence.items()
            if key != "evidence_text"
        }
        debug_info["caption_page_number"] = resolution.caption_page_number
        debug_info["visual_page_number"] = resolution.page_number
        path = str(debug_info.get("final_answer_path", ""))
        structured = debug_info.get("validated_json")
        if resolution.visual_type == "mixed_figure" and isinstance(structured, dict):
            merged, used_caption = merge_mixed_figure_with_caption(
                structured, local_evidence["panel_map"]
            )
            merged = validate_typed_response("mixed_figure", merged, composed_evidence)
            debug_info["validated_json"] = merged
            debug_info["final_structured_output"] = merged
            debug_info["rendered_structured_object"] = merged
            debug_info["final_answer_path"] = (
                "validated_partial_vision_with_text_fallback"
                if used_caption else "validated_multimodal_figure"
            )
            debug_info["final_answer_code_path"] = debug_info["final_answer_path"]
            answer = format_structured_result("mixed_figure", merged)
        elif resolution.visual_type == "mixed_figure" and not path.startswith("validated"):
            answer = caption_results_fallback(
                resolution.target_number,
                local_evidence["panel_map"],
                resolution.full_caption or resolution.caption,
            )
            debug_info["final_answer_path"] = "grounded_caption_results_fallback"
            debug_info["final_answer_code_path"] = "grounded_caption_results_fallback"
        answer = remove_unsupported_acronym_expansions(answer, local_evidence["glossary"])
        debug_info.update({
            "resolved_visual_target": resolution.to_dict(),
            "runtime_analysis_question": analysis_question,
            "runtime_error": "",
        })
    else:
        answer = remove_unsupported_acronym_expansions(answer, local_evidence["glossary"])

    slots = local_evidence["requested_answer_slots"]
    if slots:
        authoritative_path = str(
            (debug_info or {}).get("final_answer_code_path")
            or (debug_info or {}).get("final_answer_path")
            or "validated_visual"
        )
        synthesis_context = (
            "[FIGURE QUESTION COVERAGE]\n"
            f"Requested answer slots: {json.dumps(slots)}\n"
            "Resolve every slot from the supplied visual, caption, Results, or Methods evidence. "
            "For each slot, give grounded evidence, say explicitly that it was not found, or label "
            "a necessary interpretation as inference. Explicit condition-level Results statements "
            "override generalized visual summaries. Preserve quantitative qualifiers such as "
            "'less than'; do not replace them with zero. Avoid only, never, or excludes unless an "
            "author statement explicitly supports that scope. Panel summaries are intermediate "
            "evidence and must not replace the higher-level answer requested.\n"
            f"Provisional validated visual path: {authoritative_path}\n"
            "[AUTHORITATIVE PANEL ROLE MAP]\n"
            f"{json.dumps(authoritative_panel_role_map(local_evidence['panel_map']), ensure_ascii=False)}\n\n"
            "[RESOLVED ANSWER SLOT EVIDENCE]\n"
            f"{json.dumps(local_evidence['slot_evidence'], ensure_ascii=False)}\n\n"
            f"[VALIDATED VISUAL EVIDENCE]\n{answer}\n\n"
            "[EXPLICIT CONDITION OUTCOMES]\n"
            f"{json.dumps(local_evidence['explicit_condition_outcomes'], ensure_ascii=False)}\n\n"
            "[AUTHORITATIVE CONDITION TUPLES]\n"
            f"{json.dumps(local_evidence['condition_tuples'], ensure_ascii=False)}\n\n"
            "[EXPLICIT CLASSIFIER TAXONOMY]\n"
            f"{json.dumps(local_evidence['explicit_classifier_taxonomy'], ensure_ascii=False)}\n\n"
            f"{composed_evidence}"
        )
        coverage_debug = {}
        answer = generate_answer(question, synthesis_context, [], coverage_debug)
        answer, grounded_slots_appended = apply_grounded_slot_fallback(
            answer, local_evidence["slot_evidence"],
        )
        answer, panel_claims_removed = enforce_authoritative_panel_prose(
            answer, local_evidence["panel_map"],
        )
        removed_claims = []
        if "condition-specific outcomes" in slots:
            answer, removed_claims = contradiction_check_condition_prose(
                answer, local_evidence["explicit_condition_outcomes"],
                local_evidence["condition_tuples"],
            )
            if removed_claims:
                statements = list(dict.fromkeys(
                    statement
                    for row in local_evidence["condition_tuples"]
                    for statement in row.get("author_statements", [])
                ))
                answer = (
                    f"{answer.rstrip()}\n\n**Explicit author-reported condition outcomes**\n\n"
                    + "\n".join(f"- {statement}" for statement in statements)
                )
        answer = remove_unsupported_acronym_expansions(answer, local_evidence["glossary"])
        if debug_info is not None:
            debug_info.update({
                "requested_answer_slots": slots,
                "coverage_synthesis_applied": True,
                "coverage_synthesis_debug": coverage_debug,
                "grounded_slots_appended": grounded_slots_appended,
                "panel_claims_removed": panel_claims_removed,
                "condition_claims_removed": removed_claims,
                "final_answer_code_path": authoritative_path,
            })
    elif debug_info is not None:
        debug_info["requested_answer_slots"] = []
        debug_info["coverage_synthesis_applied"] = False
    return answer


def build_visual_evidence(
    resolution: VisualResolution,
    visual_answer: str,
    debug_info: dict | None = None,
) -> dict:
    """Build conversation evidence from the same final object used to render."""
    debug = debug_info if isinstance(debug_info, dict) else {}
    final_structured = debug.get("final_structured_output")
    return {
        "summary": "Local visual analysis of the selected page.",
        "analysis_kind": "Visual analysis",
        "pdf": resolution.pdf_name,
        "page": int(resolution.page_number),
        "vision_result": visual_answer,
        "structured_result": final_structured,
        "visual_target": resolution.to_dict(),
    }
