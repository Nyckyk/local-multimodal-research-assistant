"""Shared resolved-visual runtime used by Streamlit, tests and diagnostics."""

from __future__ import annotations

from pathlib import Path

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
    try:
        answer = analyse_pdf_page(
            pdf_path=Path(resolution.pdf_path),
            page_number=int(resolution.page_number),
            question=analysis_question,
            debug_info=debug_info,
            text_evidence=text_evidence,
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
        debug_info.update({
            "resolved_visual_target": resolution.to_dict(),
            "runtime_analysis_question": analysis_question,
            "runtime_error": "",
        })
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
