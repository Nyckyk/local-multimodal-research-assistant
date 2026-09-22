"""Resolve parsed figure/table references to a local PDF and 1-based page."""

from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

from services.document_matching import explicit_document_matches
from services.scientific_evidence import extract_caption_panels
from services.structured_vision import detect_visual_type
from services.visual_index import flatten_visual_targets
from services.visual_reference_parser import (
    VisualReference,
    canonical_identifier,
    parse_visual_reference,
    parse_visual_references,
)
from settings import PAPERS_FOLDER, VISUAL_AMBIGUITY_MARGIN


SAFE_CONFIDENCE = 0.72
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "does", "explain",
    "fig", "figure", "for", "from", "how", "in", "is", "it", "of", "on",
    "show", "table", "the", "this", "to", "using", "what", "with",
}


@dataclass
class VisualResolution:
    status: str
    pdf_path: str | None = None
    pdf_name: str | None = None
    page_number: int | None = None
    target_type: str = "unknown"
    target_number: str | None = None
    panel: str | None = None
    caption: str = ""
    full_caption: str = ""
    short_caption: str = ""
    caption_page_number: int | None = None
    nearby_text: str = ""
    visual_type: str | None = None
    confidence: float = 0.0
    candidate_count: int = 0
    reason: str = ""
    candidates: list[dict] = field(default_factory=list)
    reference: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _tokens(text: str) -> set[str]:
    text = unicodedata.normalize("NFKC", text or "")
    return {
        token.casefold() for token in re.findall(r"[A-Za-z0-9]+", text or "")
        if token.casefold() not in _STOPWORDS and len(token) > 1
    }


def _lexical_relevance(question: str, caption: str) -> float:
    question_tokens, caption_tokens = _tokens(question), _tokens(caption)
    if not question_tokens or not caption_tokens:
        return 0.0
    exact = question_tokens & caption_tokens
    fuzzy = {
        token for token in question_tokens - exact
        if len(token) >= 5 and any(
            other.startswith(token[:5]) or token.startswith(other[:5])
            for other in caption_tokens if len(other) >= 5
        )
    }
    matched = len(exact) + len(fuzzy)
    # Caption length should not dilute a strong match to the user's distinctive
    # terms. Long scientific captions routinely contain the few decisive words.
    coverage = matched / len(question_tokens)
    cosine = matched / math.sqrt(len(question_tokens) * len(caption_tokens))
    return 0.7 * coverage + 0.3 * cosine


def _unique_caption_coverage(question: str, captions: list[str]) -> list[float]:
    """Measure question terms that distinguish one candidate caption."""
    question_tokens = _tokens(question)
    caption_tokens = [_tokens(caption) for caption in captions]
    if not question_tokens:
        return [0.0] * len(captions)
    frequencies = {
        token: sum(token in tokens for tokens in caption_tokens)
        for token in question_tokens
    }
    return [
        sum(token in tokens and frequencies[token] == 1 for token in question_tokens)
        / len(question_tokens)
        for tokens in caption_tokens
    ]


def _caption_phrase_hits(question: str, caption: str) -> int:
    words = [
        token.casefold() for token in re.findall(r"[A-Za-z0-9]+", question)
        if token.casefold() not in _STOPWORDS
    ]
    caption_key = " ".join(re.findall(r"[a-z0-9]+", str(caption or "").casefold()))
    return sum(
        f"{first} {second}" in caption_key
        for first, second in zip(words, words[1:])
    )


def _embedding_relevance(question: str, captions: list[str], embedder=None) -> list[float]:
    if not captions:
        return []
    if embedder is None:
        return [_lexical_relevance(question, caption) for caption in captions]
    try:
        vectors = embedder.encode(
            [question, *captions], normalize_embeddings=True, show_progress_bar=False
        )
        query = vectors[0]
        return [
            max(0.0, min(1.0, float(sum(float(a) * float(b) for a, b in zip(query, vector)))))
            for vector in vectors[1:]
        ]
    except Exception:
        return [_lexical_relevance(question, caption) for caption in captions]


def _question_names_pdf(question: str, pdf_name: str) -> bool:
    return pdf_name in explicit_document_matches(question, [pdf_name])


def _previous_target(messages: list[dict], target_type: str | None = None) -> dict | None:
    for message in reversed(messages or []):
        if not isinstance(message, dict):
            continue

        direct_target = message.get("visual_target")
        evidence = message.get("evidence")
        if not isinstance(evidence, dict):
            evidence = {}
        evidence_target = evidence.get("visual_target")

        target = None
        for candidate in (direct_target, evidence_target):
            if not isinstance(candidate, dict):
                continue
            status = candidate.get("status")
            candidate_type = candidate.get("target_type")
            candidate_number = candidate.get("target_number")
            pdf_name = candidate.get("pdf_name")
            page_number = candidate.get("page_number")
            if status not in (None, "resolved"):
                continue
            if candidate_type not in {"figure", "table", "equation"}:
                continue
            if not isinstance(candidate_number, str) or not candidate_number.strip():
                continue
            if not isinstance(pdf_name, str) or not pdf_name.strip():
                continue
            if (
                not isinstance(page_number, int)
                or isinstance(page_number, bool)
                or page_number < 1
            ):
                continue
            target = candidate
            break
        if target is None:
            continue
        return target if not target_type or target.get("target_type") == target_type else None
    return None


def apply_conversation_reference(
    reference: VisualReference, messages: list[dict]
) -> tuple[VisualReference, dict | None, str]:
    """Attach panel/that-table/next-figure references to prior visual state."""
    if reference.explicit_reference:
        return reference, _previous_target(messages), "explicit visual reference"
    if reference.followup_kind == "panel":
        previous = _previous_target(messages, "figure")
        if previous:
            return replace(
                reference,
                target_type="figure",
                target_number=previous.get("target_number"),
            ), previous, "panel attached to previous figure"
    if reference.followup_kind == "previous":
        previous = _previous_target(
            messages,
            reference.target_type if reference.target_type != "unknown" else None,
        )
        if previous:
            return replace(
                reference,
                target_type=previous.get("target_type", reference.target_type),
                target_number=previous.get("target_number"),
                panel=previous.get("panel"),
            ), previous, "conversational reference attached to previous visual target"
    if reference.followup_kind == "next":
        previous = _previous_target(messages, "figure")
        identifier = str(previous.get("target_number", "")) if previous else ""
        match = re.fullmatch(r"(\d+)", identifier)
        if match:
            return replace(
                reference,
                target_type="figure",
                target_number=str(int(match.group(1)) + 1),
            ), previous, "next figure after previous visual target"
    return reference, None, "no unambiguous conversation target"


def choose_visual_type(
    reference: VisualReference, question: str, caption: str
) -> str | None:
    if reference.target_type == "equation":
        return None
    if reference.target_type == "table":
        return "table"
    panel_types = {
        row["visual_type"] for row in extract_caption_panels(caption)
        if row.get("visual_type") not in {None, "other"}
    }
    if len(panel_types) >= 2:
        return "mixed_figure"
    detected = detect_visual_type(question, caption)
    return detected or "labelled_diagram"


def manual_visual_resolution(
    pdf_path: Path, page_number: int, question: str,
) -> VisualResolution:
    reference = parse_visual_reference(question)
    return VisualResolution(
        status="resolved",
        pdf_path=str(pdf_path),
        pdf_name=pdf_path.name,
        page_number=int(page_number),
        target_type=reference.target_type,
        target_number=reference.target_number,
        panel=reference.panel,
        visual_type=("table" if reference.target_type == "table" else None),
        confidence=1.0,
        candidate_count=1,
        reason="Manual PDF/page override",
        reference=reference.to_dict(),
    )


def resolve_visual_target(
    question: str,
    index: dict,
    selected_pdf: Path | str | None = None,
    conversation_messages: list[dict] | None = None,
    current_source_names: list[str] | None = None,
    embedder=None,
    papers_folder: Path = PAPERS_FOLDER,
    reference_override: VisualReference | None = None,
) -> VisualResolution:
    reference, previous, context_reason = apply_conversation_reference(
        reference_override or parse_visual_reference(question), conversation_messages or []
    )
    if not reference.target_number or reference.target_type == "unknown":
        return VisualResolution(
            status="not_found",
            target_type=reference.target_type,
            panel=reference.panel,
            reason=(
                "No explicit or unambiguous conversational figure/table reference"
                if not reference.followup_kind else context_reason
            ),
            reference=reference.to_dict(),
        )

    all_targets = flatten_visual_targets(index, papers_folder)
    candidates = [
        target for target in all_targets
        if target["target_type"] == reference.target_type
        and canonical_identifier(target["target_number"])
        == canonical_identifier(reference.target_number)
    ]
    if not candidates:
        return VisualResolution(
            status="not_found",
            target_type=reference.target_type,
            target_number=reference.target_number,
            panel=reference.panel,
            reason=f"No indexed {reference.target_type} {reference.target_number} match",
            reference=reference.to_dict(),
        )

    # Keep the caption occurrence for each PDF/page; inline references are only
    # fallbacks when no caption occurrence exists on that same page.
    deduplicated = {}
    for candidate in candidates:
        key = (candidate["pdf_name"], candidate["page_number"])
        existing = deduplicated.get(key)
        if existing is None or (
            candidate["match_kind"] == "caption" and existing["match_kind"] != "caption"
        ):
            deduplicated[key] = candidate
    candidates = list(deduplicated.values())
    explicit_names = explicit_document_matches(
        question, [candidate["pdf_name"] for candidate in candidates]
    )
    captions = [candidate.get("caption", "") for candidate in candidates]
    embedding_relevance = _embedding_relevance(question, captions, embedder)
    lexical_relevance = [
        _lexical_relevance(
            question,
            f"{caption} {candidate.get('nearby_text', '')[:3000]}",
        )
        for caption, candidate in zip(captions, candidates)
    ]
    relevance = [
        max(embedding_score, lexical_score)
        for embedding_score, lexical_score in zip(
            embedding_relevance, lexical_relevance
        )
    ]
    unique_coverage = _unique_caption_coverage(question, captions)
    selected_name = Path(selected_pdf).name if selected_pdf else None
    previous_name = previous.get("pdf_name") if previous else None
    source_stems = {Path(name).stem.casefold() for name in (current_source_names or [])}
    scored = []
    for candidate, caption_score, unique_score, lexical_score in zip(
        candidates, relevance, unique_coverage, lexical_relevance
    ):
        if candidate["match_kind"] == "caption":
            score = 0.78
            reasons = ["exact caption identifier"]
        elif candidate["match_kind"] == "equation":
            score = 0.82
            reasons = ["exact displayed equation identifier"]
        else:
            score = 0.46
            reasons = ["exact in-page reference"]
        explicitly_named = candidate["pdf_name"] in explicit_names
        if explicitly_named:
            score += 0.80
            reasons.append("document title explicitly identified in question")
        if (
            not explicit_names and selected_name
            and candidate["pdf_name"].casefold() == selected_name.casefold()
        ):
            score += 0.04
            reasons.append("selected PDF preference")
        if (
            not explicit_names and previous_name
            and candidate["pdf_name"].casefold() == str(previous_name).casefold()
        ):
            score += 0.04
            reasons.append("previous visual PDF")
        if (
            not explicit_names
            and Path(candidate["pdf_name"]).stem.casefold() in source_stems
        ):
            score += 0.02
            reasons.append("conversation source context")
        if reference.target_type != "equation":
            score += min(0.30, caption_score * 0.30)
            if caption_score:
                reasons.append("caption relevance")
            score += min(0.15, unique_score * 0.15)
            if unique_score:
                reasons.append("distinctive caption terms")
        scored.append({
            **candidate,
            "rank_score": score,
            "score": min(1.0, score),
            "score_reasons": reasons,
            "caption_relevance_score": caption_score,
            "lexical_relevance_score": lexical_score,
            "unique_caption_coverage": unique_score,
            "caption_phrase_hits": _caption_phrase_hits(question, candidate.get("caption", "")),
        })
    scored.sort(key=lambda row: (-row["rank_score"], row["pdf_name"].casefold(), row["page_number"]))
    top = scored[0]
    second = scored[1] if len(scored) > 1 else None
    public_candidates = [
        {
            "pdf_name": row["pdf_name"],
            "page_number": row["page_number"],
            "caption": row.get("caption", ""),
            "confidence": round(row["score"], 3),
            "reason": ", ".join(row["score_reasons"]),
        }
        for row in scored
    ]
    explicit_or_context_tie_break = any(
        reason in top["score_reasons"]
        for reason in (
            "document title explicitly identified in question",
            "selected PDF preference",
            "previous visual PDF",
            "conversation source context",
        )
    )
    strong_distinctive_terminology = (
        (
            "distinctive caption terms" in top.get("score_reasons", [])
            and (
                top.get("unique_caption_coverage", 0.0) >= 0.30
                or second is None
                or top["rank_score"] - second["rank_score"] >= 0.04
            )
        )
        or (
            top.get("lexical_relevance_score", 0.0) >= 0.20
            and (
                second is None
                or top.get("lexical_relevance_score", 0.0)
                > second.get("lexical_relevance_score", 0.0) + 0.10
            )
        )
        or (
            top.get("caption_phrase_hits", 0) > 0
            and (second is None or second.get("caption_phrase_hits", 0) == 0)
        )
    )
    if (
        second
        and top["pdf_name"] != second["pdf_name"]
        and top["rank_score"] - second["rank_score"] < VISUAL_AMBIGUITY_MARGIN
        and not explicit_or_context_tie_break
        and not strong_distinctive_terminology
    ):
        return VisualResolution(
            status="ambiguous",
            target_type=reference.target_type,
            target_number=reference.target_number,
            panel=reference.panel,
            confidence=round(top["score"], 3),
            candidate_count=len(scored),
            reason="Multiple similarly ranked exact matches",
            candidates=public_candidates,
            reference=reference.to_dict(),
        )
    if top["score"] < SAFE_CONFIDENCE:
        return VisualResolution(
            status="ambiguous",
            target_type=reference.target_type,
            target_number=reference.target_number,
            panel=reference.panel,
            confidence=round(top["score"], 3),
            candidate_count=len(scored),
            reason="Best match is below the safe confidence threshold",
            candidates=public_candidates,
            reference=reference.to_dict(),
        )
    visual_type = choose_visual_type(reference, question, top.get("full_caption", top.get("caption", "")))
    return VisualResolution(
        status="resolved",
        pdf_path=top["pdf_path"],
        pdf_name=top["pdf_name"],
        page_number=int(top["page_number"]),
        target_type=reference.target_type,
        target_number=reference.target_number,
        panel=reference.panel,
        caption=top.get("caption", ""),
        full_caption=top.get("full_caption", top.get("caption", "")),
        short_caption=top.get("short_caption", top.get("caption", "")),
        caption_page_number=top.get("caption_page_number", top.get("page_number")),
        nearby_text=top.get("nearby_text", ""),
        visual_type=visual_type,
        confidence=round(top["score"], 3),
        candidate_count=len(scored),
        reason="; ".join(top["score_reasons"]),
        candidates=public_candidates,
        reference=reference.to_dict(),
    )


def resolve_visual_targets(question: str, index: dict, **kwargs) -> list[VisualResolution]:
    """Resolve every explicit target independently in request order."""
    references = parse_visual_references(question)
    if len(references) <= 1:
        return [resolve_visual_target(question, index, **kwargs)]
    clean_kwargs = dict(kwargs)
    clean_kwargs.pop("reference_override", None)
    return [
        resolve_visual_target(
            question, index, reference_override=reference, **clean_kwargs
        )
        for reference in references
    ]


def resolved_analysis_question(question: str, resolution: VisualResolution) -> str:
    if resolution.status != "resolved" or not resolution.target_number:
        return question
    label = {
        "table": "Table", "equation": "Equation",
    }.get(resolution.target_type, "Figure")
    panel = f", panel {resolution.panel}" if resolution.panel else ""
    return f"{question}\nResolved visual target: {label} {resolution.target_number}{panel}."


def should_activate_automatic_vision(question: str, resolution: VisualResolution) -> bool:
    reference = parse_visual_reference(question)
    return resolution.target_type != "equation" and resolution.status == "resolved" and (
        reference.explicit_reference or reference.followup_kind is not None
    )


def format_resolution_problem(resolution: VisualResolution) -> str:
    label = {
        "table": "Table", "equation": "Equation",
    }.get(resolution.target_type, "Figure")
    identifier = f" {resolution.target_number}" if resolution.target_number else ""
    if resolution.status == "ambiguous":
        low_confidence = "confidence" in resolution.reason.casefold()
        lines = [
            (
                f"I found only a low-confidence match for {label}{identifier}:"
                if low_confidence else
                f"I found {label}{identifier} in multiple possible locations:"
            )
        ]
        seen_pdfs = set()
        for candidate in resolution.candidates:
            if candidate["pdf_name"] in seen_pdfs:
                continue
            seen_pdfs.add(candidate["pdf_name"])
            lines.append(
                f"- {candidate['pdf_name']}, PDF page {candidate['page_number']}"
            )
        lines.append("Please select the intended paper or use the manual PDF/page override.")
        return "\n".join(lines)
    return (
        f"I could not locate {label.lower()}{identifier} in the local visual index. "
        "You can select the PDF and page with the manual vision override."
    )


def clear_visual_target_state(state) -> None:
    for key in (
        "last_visual_target", "pending_visual_resolution",
        "pending_visual_question", "visual_candidate_choice",
    ):
        state.pop(key, None)


def clear_visual_conversation_state(state) -> None:
    """Clear chat-scoped visual evidence without touching the local library."""
    state["messages"] = []
    state["last_user_question"] = ""
    clear_visual_target_state(state)

    # These names cover both current state and older/debug builds so a stale
    # resolution cannot survive an application upgrade or a Streamlit rerun.
    for key in (
        "previous_pdf",
        "previous_page",
        "previous_figure",
        "previous_table",
        "previous_panel",
        "previous_visual_target",
        "previous_source_context",
        "target_resolution",
        "automatic_detection_candidates",
        "pending_ambiguity_selection",
        "rewritten_visual_query_context",
        "pending_preferred_pdf_name",
    ):
        state.pop(key, None)

    # Widget-backed values cannot safely be changed after their widgets have
    # been instantiated. Apply this automatic preference reset at the start of
    # the next rerun, before the preferred-PDF selectbox is created. Manual
    # override controls deliberately remain untouched.
    state["pending_preferred_pdf_name"] = "No preference"

    for key in list(state):
        if str(key).startswith((
            "vision_result_", "raw_vision_", "initial_raw_", "debug_chunk_",
        )):
            state.pop(key, None)
