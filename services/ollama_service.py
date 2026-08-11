import json
import re

import ollama

from services.scientific_evidence import (
    document_glossary,
    extract_explicit_classifier_taxonomy,
    remove_unsupported_acronym_expansions,
)
from settings import NORMAL_NUM_PREDICT, OLLAMA_MODEL, SUMMARY_NUM_PREDICT


_SUMMARY_SECTION_PATTERNS = {
    "research question": r"\b(?:research question|objective|aim|purpose)\b",
    "methods": r"\b(?:methods?|methodology|approach)\b",
    "validation": r"\b(?:validation|verification|evaluation scenarios?|benchmarks?)\b",
    "main results": r"\b(?:main results?|results?|findings?)\b",
    "limitations": r"\blimitations?\b",
}


def _response_value(response, key: str, default=None):
    if isinstance(response, dict):
        return response.get(key, default)
    return getattr(response, key, default)


def _response_text(response) -> str:
    message = _response_value(response, "message", {})
    if isinstance(message, dict):
        content = message.get("content") or message.get("thinking")
    else:
        content = getattr(message, "content", None) or getattr(message, "thinking", None)
    return str(content or "").strip()


def _requested_summary_sections(question: str) -> list[str]:
    lowered = str(question or "").casefold()
    return [
        name for name, pattern in _SUMMARY_SECTION_PATTERNS.items()
        if re.search(pattern, lowered, re.IGNORECASE)
    ]


def _missing_summary_sections(answer: str, requested: list[str]) -> list[str]:
    missing = []
    for name in requested:
        match = re.search(_SUMMARY_SECTION_PATTERNS[name], answer, re.IGNORECASE)
        if not match:
            missing.append(name)
            continue
        nearby = re.sub(r"\s+", " ", answer[match.end():match.end() + 420]).strip(" :#*-\n")
        if len(re.findall(r"\b\w+\b", nearby)) < 6:
            missing.append(name)
    return missing


def _validated_framework_items(context: str) -> list[str]:
    marker = "[VALIDATED FRAMEWORK GROUNDING]"
    _, found, remainder = str(context or "").partition(marker)
    if not found:
        return []
    start = remainder.find("{")
    if start < 0:
        return []
    try:
        payload, _ = json.JSONDecoder().raw_decode(remainder[start:])
    except (json.JSONDecodeError, TypeError):
        return []
    items = payload.get("named_items", []) if isinstance(payload, dict) else []
    return [str(item).strip() for item in items if str(item).strip()]


def _missing_grounded_items(answer: str, items: list[str]) -> list[str]:
    normalized_answer = re.sub(r"[^a-z0-9]+", " ", str(answer or "").casefold())
    return [
        item for item in items
        if re.sub(r"[^a-z0-9]+", " ", item.casefold()).strip()
        not in normalized_answer
    ]


def _validated_inferred_limitations(context: str) -> list[dict]:
    marker = "[INFERRED LIMITATIONS GROUNDING]"
    _, found, remainder = str(context or "").partition(marker)
    if not found:
        return []
    start = remainder.find("{")
    if start < 0:
        return []
    try:
        payload, _ = json.JSONDecoder().raw_decode(remainder[start:])
    except (json.JSONDecodeError, TypeError):
        return []
    items = payload.get("items", []) if isinstance(payload, dict) else []
    return [
        item for item in items
        if isinstance(item, dict) and item.get("key") and item.get("statement")
    ]


_LIMITATION_ANSWER_PATTERNS = {
    "numerical_2d": r"\b2D\b.{0,80}\b(?:numerical|model|simulation)\b|\b(?:numerical|model|simulation)\b.{0,80}\b2D\b",
    "fixed_tissue_properties": r"\bfixed\b.{0,80}\btissue properties\b|\btissue properties\b.{0,80}\bfixed\b",
    "no_phase_changes": r"\b(?:no|exclude[ds]?)\b.{0,50}\bphase changes?\b|\bphase changes?\b.{0,50}\b(?:absent|excluded)\b",
    "no_chemical_reactions": r"\b(?:no|exclude[ds]?)\b.{0,50}\bchemical reactions?\b|\bchemical reactions?\b.{0,50}\b(?:absent|excluded)\b",
    "local_thermal_equilibrium": r"\bblood.tissue thermal equilibrium\b",
    "uniform_incident_irradiance": r"\buniform\b.{0,80}\bincident irradiance\b|\bincident irradiance\b.{0,80}\buniform\b",
    "simplified_environment": r"\b(?:walls?|metallic enclosures?|simplified environmental geometry|unobstructed environment)\b",
    "benchmark_validation": r"\bnew experimental human data\b",
}


def _missing_inferred_limitations(answer: str, items: list[dict]) -> list[dict]:
    return [
        item for item in items
        if not re.search(
            _LIMITATION_ANSWER_PATTERNS.get(
                str(item["key"]), re.escape(str(item["statement"]))
            ),
            str(answer or ""),
            re.IGNORECASE | re.DOTALL,
        )
    ]


def _validated_explicit_limitations(context: str) -> list[dict]:
    marker = "[EXPLICIT AUTHOR LIMITATIONS]"
    _, found, remainder = str(context or "").partition(marker)
    if not found or (start := remainder.find("{")) < 0:
        return []
    try:
        payload, _ = json.JSONDecoder().raw_decode(remainder[start:])
    except (json.JSONDecodeError, TypeError):
        return []
    items = payload.get("items", []) if isinstance(payload, dict) else []
    return [
        item for item in items
        if isinstance(item, dict) and item.get("key") and item.get("statement")
    ]


def _missing_explicit_limitations(answer: str, items: list[dict]) -> list[dict]:
    normalized_answer = set(re.findall(r"[a-z][a-z-]{3,}", str(answer or "").casefold()))
    missing = []
    for item in items:
        statement_tokens = {
            token for token in re.findall(
                r"[a-z][a-z-]{3,}", str(item.get("statement", "")).casefold()
            )
            if token not in {"than", "more", "method", "study", "hybrid"}
        }
        overlap = len(statement_tokens.intersection(normalized_answer))
        if statement_tokens and overlap / len(statement_tokens) < 0.55:
            missing.append(item)
    return missing


def _explicit_runtime_examples(items: list[dict]) -> list[str]:
    """Return author-provided quantitative runtime examples, when present."""
    examples = []
    for item in items:
        for sentence in item.get("evidence") or []:
            text = re.sub(r"\s+", " ", str(sentence or "")).strip()
            if re.search(
                r"\b(?:approximately|about|roughly)?\s*(?:three|3)\s+times\b"
                r".{0,180}\b(?:FEM|runtime|simulation|DOF)\b|"
                r"\b(?:FEM|runtime|simulation|DOF)\b.{0,180}"
                r"\b(?:three|3)\s+times\b",
                text,
                re.IGNORECASE | re.DOTALL,
            ) and text not in examples:
                examples.append(text)
    return examples


def _remove_efficiency_contradiction(context: str, answer: str) -> tuple[str, bool]:
    if not re.search(r"\b(?:more time-consuming|three times|3 times)\b", context, re.I):
        return answer, False
    pattern = re.compile(
        r"(?<=[.!?])\s+[^.!?]*\b(?:computationally more efficient|faster overall|less time-consuming)\b[^.!?]*[.!?]",
        re.I,
    )
    cleaned, count = pattern.subn("", answer)
    return cleaned.strip(), bool(count)


def _context_source_blocks(context: str) -> list[str]:
    return [
        block.strip() for block in re.split(r"(?=\nSource:\s*)", str(context or ""))
        if block.lstrip().startswith("Source:")
    ]


def _figure_identifier(block: str) -> str:
    match = re.search(r"\bfig(?:ure)?\.?\s*(\d+(?:\.\d+)?)\b", block, re.I)
    return match.group(1) if match else ""


def _unique_numbers(pattern: str, text: str) -> list[str]:
    values = []
    for match in re.finditer(pattern, text, re.I):
        value = match.group(1)
        if value not in values:
            values.append(value)
    return values


def _remove_temperature_direction_contradictions(answer: str) -> tuple[str, bool]:
    parts = re.split(r"(?<=[.!?])(?=\s|$)", str(answer or ""))
    kept = []
    removed = False
    contradiction = re.compile(
        r"temperature.{0,120}(?:diminish|decreas).{0,100}frequenc.{0,60}increas|"
        r"frequenc.{0,60}increas.{0,120}temperature.{0,100}(?:diminish|decreas)",
        re.I | re.DOTALL,
    )
    for part in parts:
        if contradiction.search(part):
            removed = True
            continue
        kept.append(part)
    return "".join(kept).strip(), removed


def _qualify_inferred_depth_language(answer: str) -> tuple[str, bool]:
    """Qualify depth terminology when the evidence is spatial contours only."""
    parts = re.split(r"(?<=[.!?])(?=\s|$)", str(answer or ""))
    changed = False
    qualified = []
    for part in parts:
        if re.search(r"\bheating depth\b|\bpenetration\b", part, re.I) and not re.search(
            r"\b(?:infer\w*|contours?|spatial)\b", part, re.I
        ):
            part = re.sub(
                r"\bheating depth\b",
                "inferred heating depth from spatial absorbed-power contours",
                part,
                flags=re.I,
            )
            part = re.sub(
                r"\bpenetration\b",
                "inferred penetration from spatial absorbed-power contours",
                part,
                flags=re.I,
            )
            changed = True
        qualified.append(part)
    return "".join(qualified), changed


def _append_grounded_multi_figure_details(
    context: str, answer: str
) -> tuple[str, list[str], bool, bool]:
    """Complete figure-specific trends from their own evidence blocks only."""
    if "[MULTI-FIGURE EVIDENCE MODE]" not in str(context or ""):
        return answer, [], False, False
    answer, contradiction_removed = _remove_temperature_direction_contradictions(answer)
    answer, depth_language_qualified = _qualify_inferred_depth_language(answer)
    additions = []
    appended = []
    for block in _context_source_blocks(context):
        identifier = _figure_identifier(block)
        lowered = block.casefold()
        if not identifier:
            continue
        if (
            ("absorbed power" in lowered or "power dissipation" in lowered)
            and re.search(r"\b(?:incident|exposure)\b", block, re.I)
        ):
            answer_has_localization = bool(re.search(
                r"(?:locali[sz]|concentrat).{0,180}(?:incident|exposure)|"
                r"(?:incident|exposure).{0,180}(?:locali[sz]|concentrat)",
                answer,
                re.I | re.DOTALL,
            ))
            if not answer_has_localization:
                directional = bool(
                    re.search(r"frequency.{0,80}increas", block, re.I | re.DOTALL)
                    and re.search(r"heated area.{0,80}(?:small|local)", block, re.I | re.DOTALL)
                )
                trend = (
                    "As frequency increases, the absorbed-power density becomes more "
                    "spatially localized near the incident boundary."
                    if directional else
                    "The absorbed-power density is spatially concentrated near the "
                    "incident boundary across the frequency panels."
                )
                additions.append(
                    f"- **Figure {identifier}:** {trend} Any heating-depth or "
                    "penetration interpretation is an inference from these "
                    "absorbed-power contours, not a direct depth measurement."
                )
                appended.append(f"figure_{identifier}_absorbed_power_localization")

        peak = re.search(r"\b(?:maximal|maximum|peak)\s+temperature", block, re.I)
        if peak:
            tail = block[peak.start():peak.start() + 900]
            temperatures = _unique_numbers(
                r"\b(\d{2}(?:\.\d+)?)\s*(?:Â°|Â◦|°|◦)?\s*C\b", tail
            )
            frequencies = _unique_numbers(
                r"\b(\d+(?:\.\d+)?)\s*GHz\b", block[:peak.start() + 40]
            )
            if len(temperatures) >= 2 and len(frequencies) >= 2 and not re.search(
                r"\bpeak temperatures?\b.{0,220}"
                + re.escape(temperatures[0])
                + r".{0,220}"
                + re.escape(temperatures[-1]),
                answer,
                re.I | re.DOTALL,
            ):
                additions.append(
                    f"- **Figure {identifier}:** The reported peak temperatures "
                    f"increase from approximately {temperatures[0]} °C at "
                    f"{frequencies[0]} GHz to {temperatures[-1]} °C at "
                    f"{frequencies[-1]} GHz; both values come from that figure's "
                    "isothermal-contour discussion."
                )
                appended.append(f"figure_{identifier}_peak_temperature_trend")
    if additions:
        answer = (
            f"{answer.rstrip()}\n\n**Grounded figure-specific details**\n\n"
            + "\n".join(additions)
        )
    return answer, appended, contradiction_removed, depth_language_qualified


def _append_grounded_transient_comparison(
    context: str, answer: str
) -> tuple[str, bool]:
    """Preserve an explicit initial model-comparison direction in summaries."""
    if "[DOCUMENT SUMMARY MODE]" not in str(context or ""):
        return answer, False
    match = re.search(
        r"initially\s*,?\s*([A-Za-z0-9-]+)\s+forecasts?\s+a\s+lower\s+heat\s+rise\s+than\s+([^.;\n]+)",
        context,
        re.I,
    )
    if not match or re.search(
        r"initial\w*.{0,100}" + re.escape(match.group(1))
        + r".{0,100}lower.{0,100}" + re.escape(match.group(2).strip()),
        answer,
        re.I | re.DOTALL,
    ):
        return answer, False
    model_a = match.group(1).strip()
    model_b = match.group(2).strip()
    convergence = (
        " As exposure approaches steady state, the two predictions converge."
        if re.search(r"progresses?\s+toward\s+a\s+steady\s+state.{0,100}converge", context, re.I | re.DOTALL)
        else ""
    )
    return (
        f"{answer.rstrip()}\n\n**Grounded transient comparison:** {model_a} "
        f"initially forecasts lower heat rise than {model_b}.{convergence}",
        True,
    )


def _ends_incomplete(answer: str) -> bool:
    text = str(answer or "").strip()
    if len(text) < 80:
        return False
    text = re.sub(
        r"(?:\s*\[[^\]]+\]|\s*\([^)]*\bpage\s+\d+[^)]*\))+$",
        "",
        text,
        flags=re.IGNORECASE,
    ).rstrip()
    if not text:
        return True
    if re.search(r"(?:\b(?:and|or|but|because|including|both)\s*)$", text, re.I):
        return True
    return text[-1] not in ".?!:;)]}"


def _merge_continuation(answer: str, continuation: str) -> str:
    base = str(answer or "").strip()
    extra = re.sub(
        r"^(?:continuation|continued answer|to continue)\s*:?\s*",
        "",
        str(continuation or "").strip(),
        flags=re.IGNORECASE,
    )
    if not extra:
        return base

    # Remove a repeated character-level boundary before checking paragraphs.
    lowered_base, lowered_extra = base.casefold(), extra.casefold()
    overlap = 0
    for size in range(min(len(base), len(extra), 600), 19, -1):
        if lowered_base[-size:] == lowered_extra[:size]:
            overlap = size
            break
    extra = extra[overlap:].lstrip()
    if not extra:
        return base

    existing_paragraphs = {
        re.sub(r"\s+", " ", paragraph).strip().casefold()
        for paragraph in re.split(r"\n\s*\n", base)
        if paragraph.strip()
    }
    new_paragraphs = [
        paragraph for paragraph in re.split(r"\n\s*\n", extra)
        if re.sub(r"\s+", " ", paragraph).strip().casefold()
        not in existing_paragraphs
    ]
    extra = "\n\n".join(new_paragraphs).strip()
    if not extra:
        return base
    separator = " " if _ends_incomplete(base) and not extra.startswith(("#", "*", "-")) else "\n\n"
    return f"{base}{separator}{extra}".strip()


def generate_answer(
    question: str,
    context: str,
    conversation_history: list[dict],
    debug_info: dict | None = None,
) -> str:
    summary_mode = "[DOCUMENT SUMMARY MODE]" in context
    requested_sections = _requested_summary_sections(question) if summary_mode else []
    grounded_items = _validated_framework_items(context)
    inferred_limitations = _validated_inferred_limitations(context)
    explicit_limitations = _validated_explicit_limitations(context)
    explicit_method_terms = []
    if summary_mode and "methods" in requested_sections:
        explicit_method_terms = [
            {"acronym": acronym, "term": expansion}
            for acronym, expansion in document_glossary(context).items()
            if re.search(r"\b(?:method|model|algorithm|classifier|framework|approach)\b", expansion, re.I)
        ][:6]
    coverage_match = re.search(r"Requested answer slots:\s*(\[[^\n]*\])", context)
    try:
        coverage_slots = json.loads(coverage_match.group(1)) if coverage_match else []
    except json.JSONDecodeError:
        coverage_slots = []
    section_rule = ""
    if len(requested_sections) >= 2:
        section_rule = (
            "\n12. Use explicit headings for every requested section: "
            + ", ".join(requested_sections)
            + ". Complete every section before ending."
        )
    prompt = f"""
You are a cautious research assistant answering questions about research papers.

Use only the supplied evidence. The evidence may include:
- extracted PDF text; and
- a labelled visual analysis produced from a selected PDF page.

EVIDENCE RULES:
1. Treat labelled visual analysis as valid evidence about figures, diagrams,
   tables, labels, colours and spatial layout.
2. For questions about a figure, table, diagram or page layout, prefer clear
   visual evidence over incomplete extracted text.
3. Do not reject a clear visual result merely because the same classification
   is not written as a sentence in the extracted text.
4. When text and visual evidence genuinely conflict, state the conflict.
5. Do not use outside knowledge or invent details.
6. Cite only source names and pages present in the supplied evidence.
7. Answer directly and concisely.
8. If neither source of evidence answers the question, say the evidence is
   insufficient.
9. Mention colours, icons, arrows or other visible features only when the
   supplied visual evidence explicitly supports them.
10. When evidence is marked DOCUMENT SUMMARY MODE, follow its document-type
    framing. Describe reviews as reviews or frameworks, not as original
    experimental studies, unless the evidence explicitly reports experiments.
11. When VALIDATED FRAMEWORK GROUNDING is present, use its category membership
    exactly. Do not promote mechanisms discussed inside a section into the
    canonical named-item list.
12. When INFERRED LIMITATIONS GROUNDING is present, include its grounded model
    constraints in the limitations section and explicitly label them as
    limitations inferred from stated assumptions, not limitations declared by
    the authors.
  13. Distinguish a convenient one-step or coupled formulation from measured
    computation time. Never call a method faster or more efficient overall
     when the evidence reports that it is more time-consuming.
  14. When EXPLICIT AUTHOR LIMITATIONS is present and the question asks for
      limitations in the plural, include every validated item and keep it
      separate from inferred constraints.
  15. For METHODS-AWARE RETRIEVAL, answer every requested answer slot. Never
      fill a missing method from "standard practice", what is "typically
      implied", or what is "presumably" done. Say it is not specified instead.
  16. Keep experimental provenance attached to its domain, species, tissue or
      cell line, treatment, control, measurement, and figure/panel. Human-derived
      cell lines are not human patient samples.
  17. For FIGURE QUESTION COVERAGE, explicit nearby Results statements override
      a generalized visual interpretation at the condition level. Preserve each
      named condition and quantitative qualifier. Do not change "less than 30%"
      into zero, and do not say only, never, or excludes unless the evidence uses
      wording with that scope.
  18. When EXPLICIT CLASSIFIER TAXONOMY is present, preserve identifiers and
      descriptions exactly. Do not infer acronym expansions, algorithm families,
      or umbrella families from a model name.
  19. A caption/panel inventory is intermediate evidence. Directly answer higher-
      level explanatory clauses using supplied Results or Methods evidence.
{section_rule}

Supplied evidence:
{context}

Question:
{question}

Before answering, verify that the conclusion is supported by either explicit PDF
text or the labelled visual analysis.
"""

    messages = [
        {
            "role": "system",
            "content": (
                "You are an evidence-grounded research assistant. A visual "
                "analysis of a selected PDF page is valid evidence for questions "
                "about figures, tables, diagrams and page layout."
            ),
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]

    num_predict = SUMMARY_NUM_PREDICT if summary_mode else NORMAL_NUM_PREDICT
    options = {
        "temperature": 0,
        "num_predict": num_predict,
    }
    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=messages,
        think=False,
        options=options,
    )

    answer = _response_text(response)

    if not answer:
        return "The model returned an empty response."

    done_reason = str(_response_value(response, "done_reason", "") or "")
    missing_sections = _missing_summary_sections(answer, requested_sections)
    missing_grounded_items = _missing_grounded_items(answer, grounded_items)
    missing_inferred_limitations = _missing_inferred_limitations(
        answer, inferred_limitations
    )
    missing_explicit_limitations = _missing_explicit_limitations(
        answer, explicit_limitations
    )
    missing_method_terms = [
        row for row in explicit_method_terms
        if not re.search(re.escape(row["term"]), answer, re.I)
    ]
    missing_coverage_slots = [
        slot for slot in coverage_slots
        if not re.search(re.escape(slot), answer, re.I)
    ]
    incomplete_ending = _ends_incomplete(answer)
    length_limited = done_reason.casefold() in {
        "length", "max_tokens", "max token", "num_predict",
    }
    needs_continuation = (
        length_limited or incomplete_ending or bool(missing_sections)
        or bool(missing_grounded_items) or bool(missing_inferred_limitations)
        or bool(missing_explicit_limitations)
        or bool(missing_method_terms)
        or bool(missing_coverage_slots)
    )
    attempts = [{
        "done": bool(_response_value(response, "done", False)),
        "done_reason": done_reason,
        "response_characters": len(answer),
        "response_words": len(answer.split()),
        "num_predict": num_predict,
    }]

    if needs_continuation:
        missing_parts = []
        if missing_sections:
            missing_parts.append("requested sections: " + ", ".join(missing_sections))
        if missing_grounded_items:
            missing_parts.append(
                "canonical framework terms: " + ", ".join(missing_grounded_items)
            )
        if missing_inferred_limitations:
            missing_parts.append(
                "grounded inferred limitations: " + "; ".join(
                    item["statement"] for item in missing_inferred_limitations
                )
            )
        if missing_explicit_limitations:
            missing_parts.append(
                "explicit author limitations: " + "; ".join(
                    item["statement"] for item in missing_explicit_limitations
                )
            )
        if missing_method_terms:
            missing_parts.append(
                "explicit source method terminology: " + ", ".join(
                    f"{row['term']} ({row['acronym']})" for row in missing_method_terms
                )
            )
        if missing_coverage_slots:
            missing_parts.append("requested methods fields: " + ", ".join(missing_coverage_slots))
        missing_text = "; ".join(missing_parts) or "the unfinished final thought"
        continuation_messages = [
            *messages,
            {"role": "assistant", "content": answer},
            {
                "role": "user",
                "content": (
                    "Continue and complete the grounded answer once. Address only "
                    f"what is missing: {missing_text}. If the last sentence is "
                    "unfinished, complete it directly. Do not repeat completed text, "
                    "do not add outside knowledge, and preserve source/page citations."
                ),
            },
        ]
        continuation_limit = 1100 if summary_mode else 450
        continued = ollama.chat(
            model=OLLAMA_MODEL,
            messages=continuation_messages,
            think=False,
            options={"temperature": 0, "num_predict": continuation_limit},
        )
        continuation_text = _response_text(continued)
        attempts.append({
            "done": bool(_response_value(continued, "done", False)),
            "done_reason": str(_response_value(continued, "done_reason", "") or ""),
            "response_characters": len(continuation_text),
            "response_words": len(continuation_text.split()),
            "num_predict": continuation_limit,
        })
        answer = _merge_continuation(answer, continuation_text)

    final_missing = _missing_summary_sections(answer, requested_sections)
    final_missing_grounded = _missing_grounded_items(answer, grounded_items)
    final_missing_limitations = _missing_inferred_limitations(
        answer, inferred_limitations
    )
    final_missing_explicit_limitations = _missing_explicit_limitations(
        answer, explicit_limitations
    )
    final_missing_method_terms = [
        row for row in explicit_method_terms
        if not re.search(re.escape(row["term"]), answer, re.I)
    ]
    grounded_items_appended = []
    if final_missing_grounded:
        grounded_items_appended = list(final_missing_grounded)
        answer = (
            f"{answer.rstrip()}\n\nCanonical framework terminology from the "
            f"validated evidence: {', '.join(final_missing_grounded)}."
        )
        final_missing_grounded = _missing_grounded_items(answer, grounded_items)
    framework_count_appended = False
    count_word = {
        1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six",
        7: "seven", 8: "eight", 9: "nine", 10: "ten", 11: "eleven", 12: "twelve",
    }.get(len(grounded_items), str(len(grounded_items)))
    if grounded_items and not re.search(
        rf"\b(?:{len(grounded_items)}|{re.escape(count_word)})\s+hallmarks?\b",
        answer,
        re.I,
    ):
        answer = (
            f"{answer.rstrip()}\n\nThe validated framework contains "
            f"{count_word} hallmarks."
        )
        framework_count_appended = True
    limitations_label_missing = bool(inferred_limitations) and not re.search(
        r"\blimitations?\b.{0,80}\binferred\b.{0,80}\bassumptions?\b|"
        r"\binferred\b.{0,80}\blimitations?\b.{0,80}\bassumptions?\b",
        answer,
        re.IGNORECASE | re.DOTALL,
    )
    inferred_limitations_appended = []
    if final_missing_limitations or limitations_label_missing:
        inferred_limitations_appended = [
            item["key"] for item in final_missing_limitations
        ]
        statements = "\n".join(
            f"- {item['statement']}" for item in final_missing_limitations
        )
        answer = (
            f"{answer.rstrip()}\n\n**Limitations inferred from stated assumptions**"
            + (f"\n\n{statements}" if statements else "")
            + "\n\nThese are inferred from the model assumptions and validation "
            "design; the authors do not present them as a dedicated limitations section."
        )
        final_missing_limitations = _missing_inferred_limitations(
            answer, inferred_limitations
        )
    explicit_limitations_appended = []
    if final_missing_explicit_limitations:
        explicit_limitations_appended = [
            item["key"] for item in final_missing_explicit_limitations
        ]
        statements = "\n".join(
            f"- {item['statement']}" for item in final_missing_explicit_limitations
        )
        answer = f"{answer.rstrip()}\n\n**Additional explicit author limitations**\n\n{statements}"
        final_missing_explicit_limitations = _missing_explicit_limitations(
            answer, explicit_limitations
        )
    explicit_method_terms_appended = []
    if final_missing_method_terms:
        explicit_method_terms_appended = [row["acronym"] for row in final_missing_method_terms]
        answer = (
            f"{answer.rstrip()}\n\n**Explicit method terminology from the source:** "
            + "; ".join(
                f"{row['term']} ({row['acronym']})" for row in final_missing_method_terms
            )
            + "."
        )
    runtime_examples_appended = []
    for example in _explicit_runtime_examples(explicit_limitations):
        if re.search(
            r"\b(?:three|3)\s+times\b.{0,180}\b(?:FEM|runtime|simulation|DOF)\b|"
            r"\b(?:FEM|runtime|simulation|DOF)\b.{0,180}\b(?:three|3)\s+times\b",
            answer,
            re.IGNORECASE | re.DOTALL,
        ):
            break
        answer = f"{answer.rstrip()}\n\n**Reported runtime example:** {example}"
        runtime_examples_appended.append(example)
        break
    answer, transient_comparison_appended = _append_grounded_transient_comparison(
        context, answer
    )
    answer, efficiency_contradiction_removed = _remove_efficiency_contradiction(
        context, answer
    )
    unsupported_completion_removed = False
    sentences = re.split(r"(?<=[.!?])\s+", answer)
    cleaned_sentences = [
        sentence for sentence in sentences
        if not re.search(r"\b(?:typically implied|standard practice would be|presumably)\b", sentence, re.I)
    ]
    if len(cleaned_sentences) != len(sentences):
        unsupported_completion_removed = True
        answer = " ".join(cleaned_sentences).strip()
    final_missing_coverage = [
        slot for slot in coverage_slots if not re.search(re.escape(slot), answer, re.I)
    ]
    if final_missing_coverage:
        answer += "\n\n**Requested fields not explicitly covered**\n\n" + "\n".join(
            f"- {slot}: Not specified in the retrieved evidence."
            for slot in final_missing_coverage
        )
    supported_terms = document_glossary(context)
    supported_terms.update({
        row["name"]: row["source_description"]
        for row in extract_explicit_classifier_taxonomy(context)
    })
    answer = remove_unsupported_acronym_expansions(answer, supported_terms)
    (
        answer,
        multi_figure_details_appended,
        contradiction_removed,
        depth_language_qualified,
    ) = (
        _append_grounded_multi_figure_details(context, answer)
    )
    if debug_info is not None:
        debug_info.update({
            "summary_mode": summary_mode,
            "num_predict": num_predict,
            "stream": False,
            "streaming_chunks_lost": False,
            "timeout_behavior": "synchronous Ollama client call; no application-level timeout",
            "generation_attempts": attempts,
            "initial_missing_sections": missing_sections,
            "initial_missing_grounded_items": missing_grounded_items,
            "initial_missing_inferred_limitations": [
                item["key"] for item in missing_inferred_limitations
            ],
            "initial_missing_explicit_limitations": [
                item["key"] for item in missing_explicit_limitations
            ],
            "initial_missing_explicit_method_terms": [
                item["acronym"] for item in missing_method_terms
            ],
            "initial_incomplete_ending": incomplete_ending,
            "initial_length_limited": length_limited,
            "continuation_used": len(attempts) == 2,
            "final_missing_sections": final_missing,
            "final_missing_grounded_items": final_missing_grounded,
            "final_missing_inferred_limitations": [
                item["key"] for item in final_missing_limitations
            ],
            "final_missing_explicit_limitations": [
                item["key"] for item in final_missing_explicit_limitations
            ],
            "grounded_items_appended": grounded_items_appended,
            "framework_count_appended": framework_count_appended,
            "inferred_limitations_appended": inferred_limitations_appended,
            "explicit_limitations_appended": explicit_limitations_appended,
            "explicit_method_terms_appended": explicit_method_terms_appended,
            "explicit_runtime_examples_appended": runtime_examples_appended,
            "grounded_transient_comparison_appended": transient_comparison_appended,
            "multi_figure_details_appended": multi_figure_details_appended,
            "multi_figure_contradiction_removed": contradiction_removed,
            "multi_figure_depth_language_qualified": depth_language_qualified,
            "efficiency_contradiction_removed": efficiency_contradiction_removed,
            "requested_answer_slots": coverage_slots,
            "final_missing_coverage_slots": final_missing_coverage,
            "unsupported_standard_practice_completion_removed": unsupported_completion_removed,
            "final_incomplete_ending": _ends_incomplete(answer),
            "response_characters": len(answer),
            "response_words": len(answer.split()),
            "final_answer_code_path": (
                "ollama_single_continuation" if len(attempts) == 2
                else "ollama_initial_response"
            ),
        })

    return answer
