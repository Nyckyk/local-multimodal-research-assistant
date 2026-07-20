"""Match explicit user document descriptions to indexed PDF titles."""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path


_TITLE_NOISE = {
    "a", "an", "and", "article", "in", "of", "on", "paper", "research",
    "study", "the", "to",
}


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
    question_key = unicodedata.normalize("NFKC", str(question or "")).casefold()
    exact = {
        name for name in names
        if Path(name).name.casefold() in question_key
    }
    if len(exact) == 1:
        return exact

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
    matches = {name for name, score in scores.items() if score == best and score > 0}
    return matches if len(matches) == 1 else set()
