import json
import re

import ollama

from settings import NORMAL_NUM_PREDICT, OLLAMA_MODEL, SUMMARY_NUM_PREDICT


_SUMMARY_SECTION_PATTERNS = {
    "research question": r"\b(?:research question|objective|aim|purpose)\b",
    "methods": r"\b(?:methods?|methodology|approach)\b",
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
    return [
        name for name in requested
        if not re.search(_SUMMARY_SECTION_PATTERNS[name], answer, re.IGNORECASE)
    ]


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
    incomplete_ending = _ends_incomplete(answer)
    length_limited = done_reason.casefold() in {
        "length", "max_tokens", "max token", "num_predict",
    }
    needs_continuation = (
        length_limited or incomplete_ending or bool(missing_sections)
        or bool(missing_grounded_items)
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
    grounded_items_appended = []
    if final_missing_grounded:
        grounded_items_appended = list(final_missing_grounded)
        answer = (
            f"{answer.rstrip()}\n\nCanonical framework terminology from the "
            f"validated evidence: {', '.join(final_missing_grounded)}."
        )
        final_missing_grounded = _missing_grounded_items(answer, grounded_items)
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
            "initial_incomplete_ending": incomplete_ending,
            "initial_length_limited": length_limited,
            "continuation_used": len(attempts) == 2,
            "final_missing_sections": final_missing,
            "final_missing_grounded_items": final_missing_grounded,
            "grounded_items_appended": grounded_items_appended,
            "final_incomplete_ending": _ends_incomplete(answer),
            "response_characters": len(answer),
            "response_words": len(answer.split()),
            "final_answer_code_path": (
                "ollama_single_continuation" if len(attempts) == 2
                else "ollama_initial_response"
            ),
        })

    return answer
