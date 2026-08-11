"""Shared resolved-visual runtime used by Streamlit, tests and diagnostics."""

from __future__ import annotations

from pathlib import Path

from services.scientific_evidence import (
    caption_results_fallback,
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
