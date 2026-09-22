"""Match explicit user document descriptions to indexed PDF titles."""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path


_TITLE_NOISE = {
    "a", "an", "and", "article", "in", "of", "on", "paper", "research",
    "study", "the", "to",
}


def normalize_document_title(text: str) -> str:
    """Return a stable comparison key for a filename or title mentioned in prose."""
    value = unicodedata.normalize("NFKC", str(text or "")).casefold()
    value = re.sub(r"[‐‑‒–—−]", "-", value)
    value = re.sub(r"\.pdf\b", "", value)
    return " ".join(re.findall(r"[a-z0-9]+", value))


def _normal_tokens(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", str(text or "")).casefold()
    return [
        token for token in re.findall(r"[a-z0-9]+", normalized)
        if token not in _TITLE_NOISE and len(token) >= 3
    ]


def _title_terms(pdf_name: str) -> set[str]:
    tokens = _normal_tokens(Path(pdf_name).stem)
    terms = set(tokens)
    # A filename may separate a lexical word at whitespace (for example,
    # "hall marks") while the user naturally writes it as one word. Derive
    # adjacent compounds from the title rather than maintaining aliases.
    terms.update(
        f"{first}{second}"
        for first, second in zip(tokens, tokens[1:])
        if len(first) + len(second) >= 7
    )
    return terms


def _paper_descriptors(question: str) -> set[str]:
    descriptors: set[str] = set()
    normalized = unicodedata.normalize("NFKC", str(question or "")).casefold()
    for match in re.finditer(
        r"\b(?P<description>(?:[a-z0-9]+(?:[-\s]+)){1,6})paper\b",
        normalized,
    ):
        tokens = _normal_tokens(match.group("description"))
        tokens = tokens[-2:]
        descriptors.update(tokens)
        descriptors.update(
            f"{first}{second}" for first, second in zip(tokens, tokens[1:])
        )
    return descriptors


def explicit_document_matches(question: str, pdf_names: list[str]) -> set[str]:
    """Return a unique PDF identified by filename or a ``... paper`` phrase.

    Short descriptors are accepted only when they distinguish one of the
    available titles. This lets title-derived phrases such as "thermal paper"
    outrank conversation state without turning generic scientific terms into a
    hidden document preference.
    """
    names = list(dict.fromkeys(str(name) for name in pdf_names if name))
    question_key = normalize_document_title(question)
    exact = {
        name for name in names
        if normalize_document_title(Path(name).stem) in question_key
    }
    if exact:
        return exact

    # A long, contiguous, unique title fragment is stronger evidence than
    # conversation state. Requiring several distinctive tokens prevents a
    # generic phrase such as "aging paper" from silently selecting a file.
    question_tokens = question_key.split()
    title_tokens = {
        name: normalize_document_title(Path(name).stem).split() for name in names
    }
    partial_scores = {}
    for name, tokens in title_tokens.items():
        best = 0
        for size in range(min(len(tokens), len(question_tokens)), 3, -1):
            if any(
                tokens[start:start + size]
                == question_tokens[offset:offset + size]
                for start in range(len(tokens) - size + 1)
                for offset in range(len(question_tokens) - size + 1)
            ):
                best = size
                break
        if best >= 4 and best / max(1, len(tokens)) >= 0.45:
            partial_scores[name] = best / len(tokens)
    if partial_scores:
        best = max(partial_scores.values())
        return {
            name for name, score in partial_scores.items()
            if abs(score - best) < 0.02
        }

    descriptors = _paper_descriptors(question)
    if not descriptors:
        return set()

    terms_by_name = {name: _title_terms(name) for name in names}
    term_frequency = {
        term: sum(term in terms for terms in terms_by_name.values())
        for term in descriptors
    }
    scores = {
        name: sum(
            1 for term in descriptors
            if term in terms and term_frequency.get(term) == 1
        )
        for name, terms in terms_by_name.items()
    }
    best = max(scores.values(), default=0)
    return {name for name, score in scores.items() if score == best and score > 0}
