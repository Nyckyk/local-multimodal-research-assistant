"""Parse explicit and conversational figure/table references."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass


_IDENTIFIER = r"(?:[A-Za-z]\.)?\d+(?:\.\d+)?|[A-Za-z]\d+"
_TARGET_WORD = r"fig(?:ure)?\.?|table"
_EQUATION_WORD = r"eq(?:uation)?\.?"


@dataclass(frozen=True)
class VisualReference:
    target_type: str = "unknown"
    target_number: str | None = None
    panel: str | None = None
    explicit_reference: bool = False
    raw_reference: str = ""
    remaining_query: str = ""
    followup_kind: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def canonical_identifier(identifier: str | None) -> str:
    """Return a comparison key while preserving the original elsewhere."""
    if identifier is None:
        return ""
    return re.sub(r"\s+", "", str(identifier)).rstrip(".").casefold()


def _target_type(word: str) -> str:
    lowered = word.casefold()
    if lowered.startswith("table"):
        return "table"
    if lowered.startswith("eq"):
        return "equation"
    return "figure"


def _remaining_query(question: str, start: int, end: int) -> str:
    remaining = f"{question[:start]} {question[end:]}"
    return re.sub(r"\s+", " ", remaining).strip(" ,;:-")


def parse_visual_reference(question: str) -> VisualReference:
    """Parse one visual target without treating unrelated numbers as targets."""
    text = str(question or "")

    equation = re.search(
        rf"\b(?P<kind>{_EQUATION_WORD})\s*\(?\s*"
        r"(?P<number>\d+(?:\.\d+)?[a-z]?)\s*\)?"
        r"(?![A-Za-z0-9])",
        text,
        re.IGNORECASE,
    )
    if equation:
        return VisualReference(
            target_type="equation",
            target_number=equation.group("number"),
            explicit_reference=True,
            raw_reference=equation.group(0),
            remaining_query=_remaining_query(text, *equation.span()),
        )

    panel_prefix = re.search(
        rf"\bpanel\s+(?P<panel>[A-Za-z])\s+(?:of|in)\s+"
        rf"(?:(?:supplementary|supp\.)\s+)?(?P<kind>{_TARGET_WORD})\s*"
        rf"(?P<number>{_IDENTIFIER})(?![A-Za-z0-9])",
        text,
        re.IGNORECASE,
    )
    if panel_prefix:
        return VisualReference(
            target_type=_target_type(panel_prefix.group("kind")),
            target_number=panel_prefix.group("number"),
            panel=panel_prefix.group("panel").casefold(),
            explicit_reference=True,
            raw_reference=panel_prefix.group(0),
            remaining_query=_remaining_query(text, *panel_prefix.span()),
        )

    explicit = re.search(
        rf"\b(?:(?:supplementary|supp\.)\s+)?(?P<kind>{_TARGET_WORD})\s*"
        rf"(?P<number>{_IDENTIFIER})(?P<suffix>[a-z])?"
        rf"(?:\s*\(\s*(?:panel\s*)?(?P<paren>[A-Za-z])\s*\))?"
        rf"(?![A-Za-z0-9])",
        text,
        re.IGNORECASE,
    )
    if explicit:
        kind = _target_type(explicit.group("kind"))
        panel = explicit.group("paren")
        if kind == "figure" and not panel:
            panel = explicit.group("suffix")
        number = explicit.group("number")
        # The suffix belongs to a figure panel (Fig. 13b), not its identifier.
        # Supplement identifiers such as S2 are captured wholly by _IDENTIFIER.
        return VisualReference(
            target_type=kind,
            target_number=number,
            panel=panel.casefold() if panel else None,
            explicit_reference=True,
            raw_reference=explicit.group(0),
            remaining_query=_remaining_query(text, *explicit.span()),
        )

    panel_followup = re.search(r"\b(?:what about\s+)?panel\s+([A-Za-z])\b", text, re.I)
    if panel_followup:
        return VisualReference(
            panel=panel_followup.group(1).casefold(),
            raw_reference=panel_followup.group(0),
            remaining_query=_remaining_query(text, *panel_followup.span()),
            followup_kind="panel",
        )

    table_followup = re.search(r"\b(?:that|this|the)\s+table\b", text, re.I)
    if table_followup:
        return VisualReference(
            target_type="table",
            raw_reference=table_followup.group(0),
            remaining_query=_remaining_query(text, *table_followup.span()),
            followup_kind="previous",
        )

    equation_followup = re.search(
        r"\b(?:that|this|the)\s+equation\b", text, re.I
    )
    if equation_followup:
        return VisualReference(
            target_type="equation",
            raw_reference=equation_followup.group(0),
            remaining_query=_remaining_query(text, *equation_followup.span()),
            followup_kind="previous",
        )

    figure_followup = re.search(r"\b(?:that|this|the)\s+figure\b", text, re.I)
    if figure_followup:
        return VisualReference(
            target_type="figure",
            raw_reference=figure_followup.group(0),
            remaining_query=_remaining_query(text, *figure_followup.span()),
            followup_kind="previous",
        )

    next_figure = re.search(r"\b(?:the\s+)?next\s+figure\b", text, re.I)
    if next_figure:
        return VisualReference(
            target_type="figure",
            raw_reference=next_figure.group(0),
            remaining_query=_remaining_query(text, *next_figure.span()),
            followup_kind="next",
        )

    if re.search(r"\b(?:there|it)\b", text, re.I) and re.search(
        r"\b(?:axis|curve|figure|group|highest|lowest|panel|plot|row|show|table)\b",
        text,
        re.I,
    ):
        return VisualReference(
            raw_reference="conversational visual reference",
            remaining_query=text.strip(),
            followup_kind="previous",
        )

    return VisualReference(remaining_query=text.strip())


def has_visual_reference(question: str) -> bool:
    reference = parse_visual_reference(question)
    return reference.explicit_reference or reference.followup_kind is not None


def parse_visual_references(question: str) -> list[VisualReference]:
    """Return all explicit targets, expanding shared lists and ranges."""
    text = str(question or "")
    identifier = r"(?:(?:[A-Za-z]\.)?\d+(?:\.\d+)?|[A-Za-z]\d+)"
    block_pattern = re.compile(
        rf"\b(?P<kind>fig(?:ure)?s?|tables?|eq(?:uation)?s?)\.?\s*"
        rf"(?P<body>\(?\s*{identifier}\s*\)?(?:\s*(?:,|and|&|to|through|[-\u2013\u2014])"
        rf"\s*\(?\s*{identifier}\s*\)?)+|\(?\s*{identifier}\s*\)?)",
        re.IGNORECASE,
    )
    positioned: list[tuple[int, VisualReference]] = []
    for block in block_pattern.finditer(text):
        kind = block.group("kind").casefold()
        target_type = "figure" if kind.startswith("fig") else "table" if kind.startswith("table") else "equation"
        body = block.group("body")
        raw_values = re.findall(identifier, body, re.IGNORECASE)
        if not raw_values:
            continue
        if len(raw_values) == 2 and re.search(r"\b(?:to|through)\b|[-\u2013\u2014]", body, re.I):
            first, last = raw_values
            values = (
                [str(value) for value in range(int(first), int(last) + 1)]
                if first.isdigit() and last.isdigit() and int(last) >= int(first)
                else raw_values
            )
        else:
            values = raw_values
        for offset, value in enumerate(values):
            panel = None
            if target_type == "figure" and (compact := re.fullmatch(r"(\d+)([A-Za-z])", value)):
                value, panel = compact.group(1), compact.group(2).casefold()
            positioned.append((block.start() + offset, VisualReference(
                target_type=target_type,
                target_number=value.strip("() "),
                panel=panel,
                explicit_reference=True,
                raw_reference=block.group(0),
                remaining_query=_remaining_query(text, *block.span()),
            )))
    positioned.sort(key=lambda row: row[0])
    values, seen = [], set()
    for _, reference in positioned:
        key = (reference.target_type, canonical_identifier(reference.target_number), reference.panel)
        if key not in seen:
            seen.add(key)
            values.append(reference)
    return values
