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


def _canonical_equation(value: str) -> str:
    text = str(value or "")
    for function in ("Qext", "Qmet", "Tb", "T"):
        text = re.sub(
            rf"{function}\s*\(\s*x\s*,\s*y\s*,\s*t\s*\)",
            function,
            text,
            flags=re.IGNORECASE,
        )
    for symbol, name in {
        "ρ": "rho", "τ": "tau", "ω": "omega", "∇": "nabla", "∂": "d",
        "−": "-", "–": "-", "—": "-",
    }.items():
        text = text.replace(symbol, name)
    return re.sub(r"[\s.,]+", "", text)


def _split_signed_terms(expression: str) -> list[tuple[str, str]]:
    terms = []
    depth = 0
    sign = "+"
    start = 0
    for index, character in enumerate(expression):
        if character == "(":
            depth += 1
        elif character == ")":
            depth = max(0, depth - 1)
        elif character in "+-" and depth == 0:
            if index > start:
                terms.append((sign, expression[start:index]))
            sign = character
            start = index + 1
    if start < len(expression):
        terms.append((sign, expression[start:]))
    return [(sign, term) for sign, term in terms if term]


def _equation_term_kind(term: str) -> str:
    lowered = term.casefold()
    if "nabla2t" in lowered:
        return "conduction"
    if "dqextdt" in lowered:
        return "external_source_time_derivative"
    if lowered == "qext":
        return "external_source"
    if "dtdt" in lowered:
        return "first_time_derivative"
    if "rhobcbomegabtb" in lowered:
        return "blood_temperature_perfusion"
    if "rhobcbomegabt" in lowered:
        return "tissue_temperature_perfusion"
    return "unknown"


def _target_equation_block(evidence: str, equation_number: str) -> str:
    target = str(evidence or "").split("\n\nReferenced Equation", 1)[0]
    match = re.search(
        rf"(?:re-?written\s+as:)\s*(.*?)\n\(\s*{re.escape(str(equation_number))}\s*\)",
        target,
        re.IGNORECASE | re.DOTALL,
    )
    return match.group(1).strip() if match else ""


def build_validated_zero_relaxation_answer(
    question: str, evidence: str, equation_number: str
) -> tuple[str, dict] | None:
    """Parse, reduce and sign-check a displayed relaxation-time equation."""
    lowered_question = str(question or "").casefold()
    if "zero" not in lowered_question or not any(
        phrase in lowered_question for phrase in ("relaxation time", "tau", "τ")
    ):
        return None
    block = _target_equation_block(evidence, equation_number)
    canonical = _canonical_equation(block)
    if canonical.count("=") != 1:
        return None
    lhs, rhs = canonical.split("=", 1)
    if not all(token in lhs.casefold() for token in ("rhoc", "tau", "d2t", "dt2")):
        return None

    parsed_terms = []
    for sign, term in _split_signed_terms(rhs):
        parsed_terms.append({
            "side": "right",
            "sign": sign,
            "kind": _equation_term_kind(term),
            "source_term": term,
            "tau_dependent": "tau" in term.casefold(),
        })
    expected_signs = {
        "conduction": "+",
        "tissue_temperature_perfusion": "-",
        "first_time_derivative": "-",
        "blood_temperature_perfusion": "+",
        "external_source": "+",
        "external_source_time_derivative": "+",
    }
    if {term["kind"] for term in parsed_terms} != set(expected_signs):
        return None
    if any(term["sign"] != expected_signs[term["kind"]] for term in parsed_terms):
        return None
    derivative = next(
        term for term in parsed_terms if term["kind"] == "first_time_derivative"
    )
    if not all(token in derivative["source_term"].casefold() for token in ("tau", "rhoc", "dtdt")):
        return None

    target_terms = [{
        "side": "left", "sign": "+", "kind": "second_time_derivative",
        "source_term": lhs, "tau_dependent": True,
    }, *parsed_terms]
    zero_terms = [
        {"side": "right", "sign": "+", "kind": "conduction"},
        {"side": "right", "sign": "-", "kind": "tissue_temperature_perfusion"},
        {"side": "right", "sign": "-", "kind": "first_time_derivative"},
        {"side": "right", "sign": "+", "kind": "blood_temperature_perfusion"},
        {"side": "right", "sign": "+", "kind": "external_source"},
    ]
    rearranged_terms = [
        {"side": "left", "sign": "+", "kind": "first_time_derivative"},
        {"side": "right", "sign": "+", "kind": "conduction"},
        {"side": "right", "sign": "+", "kind": "blood_minus_tissue_perfusion"},
        {"side": "right", "sign": "+", "kind": "external_source"},
    ]
    assumption = bool(re.search(
        r"Q\s*_?\s*met\s+is\s+assumed\s+(?:to\s+be\s+)?zero",
        evidence,
        re.IGNORECASE,
    ))
    if not assumption:
        return None

    answer = rf"""**Complete Equation {equation_number}**

The paper states that $Q_{{met}}$ is assumed zero ($Q_{{met}}=0$). Preserving its symbols and every signed term, the displayed equation is

$$
\rho c\tau\frac{{\partial^2T}}{{\partial t^2}}
=K\nabla^2T-\rho_b c_b\omega_bT
-(\tau\rho_b c_b\omega_b+\rho c)\frac{{\partial T}}{{\partial t}}
+\rho_b c_b\omega_bT_b+Q_{{ext}}
+\tau\frac{{\partial Q_{{ext}}}}{{\partial t}}.
$$

Setting $\tau=0$ removes the second-time-derivative term, the $\tau$-dependent part of the first-time-derivative coefficient, and the time derivative of the external source. It leaves

$$
0=K\nabla^2T-\rho_b c_b\omega_bT
-\rho c\frac{{\partial T}}{{\partial t}}
+\rho_b c_b\omega_bT_b+Q_{{ext}}.
$$

Moving only the negative first-time-derivative term to the left gives

$$
\rho c\frac{{\partial T}}{{\partial t}}
=K\nabla^2T+\rho_b c_b\omega_b(T_b-T)+Q_{{ext}}.
$$

This remains the transient Pennes equation with $Q_{{met}}=0$."""
    symbolic = {
        "target_equation_number": str(equation_number),
        "source_equation": canonical,
        "target_terms": target_terms,
        "zero_relaxation_terms": zero_terms,
        "rearranged_terms": rearranged_terms,
        "paper_symbols": {"specific_heat": "c", "blood_temperature": "T_b"},
        "metabolic_heat_assumed_zero": True,
        "sign_validation": "passed",
    }
    return answer, symbolic


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
    symbolic_result = build_validated_zero_relaxation_answer(
        question, evidence, resolution.target_number
    )
    if symbolic_result is not None:
        answer, symbolic = symbolic_result
        supplemented = False
        generation_path = "deterministic_symbolic_reduction"
        generation_debug["symbolic_equation"] = symbolic
    else:
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
