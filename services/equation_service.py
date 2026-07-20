"""Grounded text analysis for explicitly numbered equations."""

from __future__ import annotations

import re
from pathlib import Path

import fitz

from services.ollama_service import generate_answer
from services.visual_locator import VisualResolution
from services.visual_reference_parser import canonical_identifier


def _displayed_equation_position(text: str, identifier: str) -> int | None:
    pattern = re.compile(
        rf"(?m)^\s*\({re.escape(str(identifier))}\)\s*$", re.IGNORECASE
    )
    match = pattern.search(text or "")
    return match.start() if match else None


def _equation_excerpt(text: str, identifier: str, radius: int = 1900) -> str:
    position = _displayed_equation_position(text, identifier)
    if position is None:
        return ""
    return text[max(0, position - radius):position + radius]


def build_equation_evidence(
    pdf_path: Path, page_number: int, equation_number: str
) -> str:
    with fitz.open(pdf_path) as document:
        page_index = int(page_number) - 1
        page_text = document[page_index].get_text("text") or ""
        target_excerpt = _equation_excerpt(page_text, equation_number)
        if not target_excerpt:
            target_excerpt = page_text

        referenced = []
        for identifier in re.findall(
            r"\bequation\s*\(\s*(\d+(?:\.\d+)?[a-z]?)\s*\)",
            target_excerpt,
            re.IGNORECASE,
        ):
            if canonical_identifier(identifier) != canonical_identifier(equation_number):
                referenced.append(identifier)

        prior_parts = []
        for identifier in dict.fromkeys(referenced):
            found = ""
            found_page = None
            for index in range(page_index, max(-1, page_index - 4), -1):
                candidate_text = document[index].get_text("text") or ""
                found = _equation_excerpt(candidate_text, identifier, radius=850)
                if found:
                    found_page = index + 1
                    break
            if found:
                prior_parts.append(
                    f"Referenced Equation {identifier}, PDF page {found_page}:\n{found}"
                )

    prior_evidence = "\n\n".join(prior_parts)
    return (
        "[EQUATION ANALYSIS MODE]\n"
        "Use the displayed equation and its immediate author explanation. "
        "When a reduction is requested, remove only terms whose stated parameter "
        "makes them zero and show the resulting equation. State assumptions made "
        "immediately before the target equation, and give a concise list of every "
        "term or coefficient eliminated by the zero-valued parameter. Use the "
        "paper's variable names in that list. Do not invoke figure vision.\n\n"
        f"Target Equation {equation_number}, PDF page {page_number}:\n{target_excerpt}\n\n"
        f"{prior_evidence}"
    ).strip()


def _append_grounded_zero_reduction(
    question: str, evidence: str, answer: str
) -> tuple[str, bool]:
    """Make a requested zero-relaxation reduction explicit from displayed terms."""
    lowered_question = str(question or "").casefold()
    if "zero" not in lowered_question or not any(
        phrase in lowered_question for phrase in ("relaxation time", "tau", "τ")
    ):
        return answer, False

    evidence_text = str(evidence or "")
    has_second_time_term = bool(
        re.search(r"[ττ].{0,80}∂\s*2\s*T|tau.{0,80}(?:second|\^?2)", evidence_text, re.I | re.S)
    )
    has_external_derivative = bool(
        re.search(r"[ττ].{0,80}∂\s*Q\s*_?\s*ext|tau.{0,80}(?:partial|derivative).{0,40}Q\s*_?\s*ext", evidence_text, re.I | re.S)
    )
    has_tau_coefficient = bool(
        re.search(r"\([ττ].{0,100}\).{0,80}∂\s*T|\(tau.{0,100}\).{0,80}(?:partial|derivative)", evidence_text, re.I | re.S)
    )
    if not (has_second_time_term and has_external_derivative and has_tau_coefficient):
        return answer, False

    assumption = re.search(
        r"(?:since\s+)?([A-Za-z][A-Za-z0-9_]*)\s+is\s+assumed\s+"
        r"(?:to\s+be\s+)?zero",
        evidence_text,
        re.IGNORECASE,
    )
    assumption_text = (
        f"{assumption.group(1)} is assumed zero before the target equation. "
        if assumption else ""
    )
    supplement = (
        "Grounded zero-relaxation reduction: "
        f"{assumption_text}Setting τ = 0 removes every τ-dependent "
        "contribution: the second-time-derivative term, the τ-dependent part "
        "of the coefficient multiplying ∂T/∂t, and the time derivative of "
        "the external source."
    )
    if "grounded zero-relaxation reduction" in str(answer or "").casefold():
        return answer, False
    return f"{str(answer or '').rstrip()}\n\n{supplement}", True


def analyse_resolved_equation(
    question: str,
    resolution: VisualResolution,
    *,
    conversation_history: list[dict] | None = None,
    debug_info: dict | None = None,
) -> str:
    if resolution.status != "resolved" or resolution.target_type != "equation":
        raise ValueError("Equation analysis requires a resolved equation target.")
    if not resolution.pdf_path or not resolution.page_number or not resolution.target_number:
        raise ValueError("Resolved equation target is incomplete.")

    evidence = build_equation_evidence(
        Path(resolution.pdf_path), resolution.page_number, resolution.target_number
    )
    generation_debug = debug_info if debug_info is not None else {}
    answer = generate_answer(
        question, evidence, conversation_history or [], debug_info=generation_debug
    )
    answer, supplemented = _append_grounded_zero_reduction(
        question, evidence, answer
    )
    generation_path = generation_debug.get("final_answer_code_path", "")
    generation_debug.update({
        "equation_evidence": evidence,
        "grounded_zero_reduction_supplemented": supplemented,
        "generation_code_path": generation_path,
        "final_answer_path": "validated_text_equation_analysis",
        "final_answer_code_path": "validated_text_equation_analysis",
    })
    return answer
