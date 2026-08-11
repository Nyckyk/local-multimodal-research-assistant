"""Document-grounded helpers for mixed figures and experimental evidence."""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from pathlib import Path

import fitz


PANEL_MARKER = re.compile(
    r"(?:^|(?<=[.!?])\s+)(?P<panels>[a-z](?:\s*[-–—,]\s*[a-z])*)\s+(?=[A-Z0-9])"
)


def extract_caption_panels(caption: str) -> list[dict]:
    """Split a complete caption into panel records, expanding ranges."""
    text = re.sub(r"\s+", " ", str(caption or "")).strip()
    markers = list(PANEL_MARKER.finditer(text))
    rows: list[dict] = []
    for index, marker in enumerate(markers):
        raw = re.sub(r"[–—]", "-", marker.group("panels").lower().replace(" ", ""))
        panels: list[str] = []
        if "-" in raw and len(raw.split("-")) == 2:
            start, end = raw.split("-")
            if len(start) == len(end) == 1 and ord(start) <= ord(end):
                panels = [chr(value) for value in range(ord(start), ord(end) + 1)]
        if not panels:
            panels = re.findall(r"[a-z]", raw)
        description = text[marker.end(): markers[index + 1].start() if index + 1 < len(markers) else len(text)].strip(" .;")
        visual_type = classify_panel_visual_type(description)
        for panel in panels:
            rows.append({
                "panel": panel,
                "visual_type": visual_type,
                "caption_description": description,
            })
    return rows


def classify_panel_visual_type(description: str) -> str:
    text = str(description or "").casefold()
    if re.search(r"\b(?:heatmap|heat map)\b", text):
        return "heatmap"
    if re.search(r"\b(?:representative images?|micrographs?|staining|histolog|microscop)\b", text):
        return "microscopy"
    if re.search(r"\b(?:experimental design|workflow|timeline|scheme|schematic)\b", text):
        return "workflow"
    if re.search(r"\b(?:quantification|correlation|distribution|percentage|score|curve|plot|analysis)\b", text):
        return "graph"
    if re.search(r"\btable\b", text):
        return "table"
    return "other"


def panel_coverage(question: str, panel_map: list[dict], result: dict) -> dict:
    expected = {str(row.get("panel", "")).casefold() for row in panel_map if row.get("panel")}
    if not expected:
        expected = set(re.findall(r"\bpanel\s+([a-z])\b", str(question or ""), re.I))
    present = set()
    informative = set()
    for row in result.get("panels", []) if isinstance(result, dict) else []:
        if not isinstance(row, dict):
            continue
        panel = str(row.get("panel", "")).casefold()
        if panel:
            present.add(panel)
        payload = row.get("structured_analysis", row)
        if any(
            payload.get(key)
            for key in ("summary", "observations", "components", "measurements", "series", "relationships")
        ):
            informative.add(panel)
    missing = sorted(expected - informative)
    return {
        "expected": sorted(expected),
        "present": sorted(present),
        "informative": sorted(informative),
        "missing": missing,
        "ratio": len(informative & expected) / len(expected) if expected else (1.0 if informative else 0.0),
    }


def structured_semantically_sufficient(
    visual_type: str, result: dict, question: str = "", panel_map: list[dict] | None = None,
) -> tuple[bool, dict]:
    if visual_type == "mixed_figure":
        coverage = panel_coverage(question, panel_map or [], result)
        return coverage["ratio"] >= 0.6 and bool(coverage["informative"]), coverage
    if visual_type in {"labelled_diagram", "circuit", "schematic", "diagram"}:
        informative = bool(
            result.get("components") or result.get("spatial_relationships")
            or result.get("connections") or str(result.get("explanation", "")).strip()
        )
        return informative, {"informative": informative}
    return True, {"informative": True}


def document_glossary(text: str) -> dict[str, str]:
    """Extract only acronym expansions explicitly present in document text."""
    glossary: dict[str, str] = {}
    for match in re.finditer(
        r"\b([A-Z][A-Za-z][A-Za-z -]{2,80}?)\s*\(([A-Z][A-Z0-9-]{1,12})\)",
        str(text or ""),
    ):
        expansion, acronym = re.sub(r"\s+", " ", match.group(1)).strip(), match.group(2)
        glossary.setdefault(acronym, expansion)
    for match in re.finditer(
        r"\b([A-Z][A-Z0-9-]{1,12})\s*\(([A-Za-z][A-Za-z0-9 -]{3,100})\)",
        str(text or ""),
    ):
        acronym, expansion = match.group(1), re.sub(r"\s+", " ", match.group(2)).strip()
        if len(expansion.split()) >= 2:
            glossary.setdefault(acronym, expansion)
    return glossary


def remove_unsupported_acronym_expansions(answer: str, glossary: dict[str, str]) -> str:
    """Preserve unexplained acronyms and remove unsupported parenthetical guesses."""
    def replace(match):
        acronym, expansion = match.group(1), match.group(2).strip()
        grounded = glossary.get(acronym)
        return match.group(0) if grounded and grounded.casefold() == expansion.casefold() else acronym
    return re.sub(r"\b([A-Z][A-Z0-9-]{1,12})\s*\(([^)]{3,100})\)", replace, str(answer or ""))


@dataclass
class ExperimentalProvenance:
    experimental_domain: str
    species: str | None = None
    cell_line_or_tissue: str | None = None
    treatment_or_stressor: str | None = None
    control: str | None = None
    measurement_or_marker: str | None = None
    figure_or_panel: str | None = None
    purpose: str | None = None


def classify_experimental_domain(text: str) -> str:
    raw = re.sub(r"\s+", " ", str(text or ""))
    value = raw.casefold()
    if re.search(r"\b(?:patients?|clinical|biops(?:y|ies)|human (?:liver|tissue|samples?))\b", value):
        return "human_clinical_patient_tissue"
    if re.search(r"\b(?:mice|mouse|murine|rat|in vivo|animal model)\b", value):
        return "animal_model"
    if re.search(r"\b(?:ex vivo|tissue sections?|organotypic)\b", value):
        return "ex_vivo_tissue"
    if re.search(r"\bprimary (?:cells?|fibroblasts?|hepatocytes?)\b", value):
        return "primary_cell"
    generic_cell_line_token = re.search(
        r"\b[A-Z][A-Z0-9-]{2,15}(?:\s+human)?\s+cells?\b", raw
    )
    if generic_cell_line_token or re.search(
        r"\b(?:cell lines?|cultured cells?|in vitro|cells? treated)\b", value
    ):
        return "in_vitro_cell_line"
    if re.search(r"\b(?:algorithm|classifier|computational|simulation|model training)\b", value):
        return "computational"
    return "other"


def experimental_provenance(text: str, figure_or_panel: str | None = None) -> dict:
    value = re.sub(r"\s+", " ", str(text or ""))
    species_match = re.search(r"\b(human|mouse|mice|murine|rat)\b", value, re.I)
    cell_match = re.search(
        r"\b([A-Z][A-Z0-9-]{2,15}(?:\s+(?:cells?|cell line))?|(?:liver|brain|adipose|tumou?r) tissue)\b",
        value,
    )
    treatment = re.search(r"\b(?:treated with|exposed to|induced by)\s+([^.;]{2,80})", value, re.I)
    control = re.search(r"\b(?:control|vehicle|DMSO)[^.;]{0,50}", value, re.I)
    marker = re.search(r"\b(?:measured|stained|marker|score|expression of)\s+([^.;]{2,80})", value, re.I)
    return asdict(ExperimentalProvenance(
        experimental_domain=classify_experimental_domain(value),
        species=species_match.group(1) if species_match else None,
        cell_line_or_tissue=cell_match.group(1) if cell_match else None,
        treatment_or_stressor=treatment.group(1).strip() if treatment else None,
        control=control.group(0).strip() if control else None,
        measurement_or_marker=marker.group(1).strip() if marker else None,
        figure_or_panel=figure_or_panel,
    ))


def requested_answer_slots(question: str) -> list[str]:
    patterns = (
        ("features", r"\b(?:features?|variables?|measurements?)\b"),
        ("library construction", r"\b(?:librar(?:y|ies)|training sets? assembled|constructed)\b"),
        ("training cell counts", r"\b(?:number of cells|cell counts?|how many cells)\b"),
        ("CT split", r"\b(?:CT|classification tree).{0,40}\b(?:split|test|proportion)\b"),
        ("RF split", r"\b(?:RF|random forest).{0,40}\b(?:split|test|proportion)\b"),
        ("CT overfitting method", r"\b(?:overfitting|over-fitting|pruning|alpha)\b"),
        ("RF threshold", r"\b(?:RF|random forest).{0,40}\bthreshold\b|\bthreshold.{0,40}(?:RF|random forest)\b"),
        ("inclusion criteria", r"\b(?:inclusion|exclusion|criteria|threshold for these samples)\b"),
    )
    return [name for name, pattern in patterns if re.search(pattern, str(question or ""), re.I)]


def unsupported_source_completion(text: str) -> bool:
    return bool(re.search(
        r"\b(?:typically implied|standard practice would be|presumably)\b",
        str(text or ""), re.I,
    ))


def figure_local_evidence(
    pdf_path: Path,
    visual_page: int,
    caption_page: int | None,
    figure_number: str | None,
    full_caption: str,
) -> dict:
    """Collect deterministic caption, adjacent-page and direct-reference evidence."""
    with fitz.open(pdf_path) as document:
        page_numbers = {
            page for anchor in (visual_page, caption_page or visual_page)
            for page in (anchor - 1, anchor, anchor + 1)
            if 1 <= page <= len(document)
        }
        local_pages = {
            page: document[page - 1].get_text("text") or ""
            for page in sorted(page_numbers)
        }
        direct = []
        if figure_number:
            pattern = re.compile(
                rf"\b(?:Fig\.?|Figure)\s*{re.escape(str(figure_number))}\b",
                re.I,
            )
            for page_index, page in enumerate(document, start=1):
                text = page.get_text("text") or ""
                for match in pattern.finditer(text):
                    start, end = max(0, match.start() - 650), min(len(text), match.end() + 1100)
                    direct.append({"page": page_index, "text": re.sub(r"\s+", " ", text[start:end]).strip()})
        whole_text = "\n".join(page.get_text("text") or "" for page in document)
    panel_map = extract_caption_panels(full_caption)
    evidence_text = (
        f"TARGET FIGURE CAPTION (complete):\n{full_caption}\n\n"
        + "\n\n".join(
            f"LOCAL PDF PAGE {page}:\n{text}" for page, text in local_pages.items()
        )
        + "\n\nDIRECT FIGURE REFERENCES:\n"
        + "\n".join(f"Page {row['page']}: {row['text']}" for row in direct)
    )
    return {
        "full_caption": full_caption,
        "panel_map": panel_map,
        "local_pages": sorted(local_pages),
        "direct_references": direct,
        "evidence_text": evidence_text,
        "glossary": document_glossary(whole_text),
    }


def merge_mixed_figure_with_caption(result: dict, panel_map: list[dict]) -> tuple[dict, bool]:
    """Retain validated panels and fill only missing panels from explicit caption text."""
    merged = dict(result)
    panels = [dict(row) for row in result.get("panels", []) if isinstance(row, dict)]
    present = {str(row.get("panel", "")).casefold() for row in panels}
    added = False
    for caption_panel in panel_map:
        panel = str(caption_panel.get("panel", "")).casefold()
        if not panel or panel in present or not caption_panel.get("caption_description"):
            continue
        panels.append({
            "panel": panel,
            "visual_type": caption_panel.get("visual_type", "other"),
            "structured_analysis": {
                "summary": caption_panel["caption_description"],
                "labels": [], "components": [], "measurements": [], "observations": [],
                "evidence": ["caption"],
            },
            "confidence": 0.9,
            "uncertain_items": [],
        })
        present.add(panel)
        added = True
    panels.sort(key=lambda row: str(row.get("panel", "")))
    merged["panels"] = panels
    return merged, added


def caption_results_fallback(figure_number: str | None, panel_map: list[dict], full_caption: str) -> str:
    label = f"Figure {figure_number}" if figure_number else "This figure"
    if panel_map:
        lines = [f"**{label} — caption- and Results-grounded panel summary**"]
        lines.extend(
            f"\n- **Panel {row['panel']} ({row['visual_type']}):** {row['caption_description']}"
            for row in panel_map if row.get("caption_description")
        )
        lines.append(
            "\nThe visual model could not verify every panel, so the statements above "
            "come from the complete author caption and nearby document text."
        )
        return "\n".join(lines)
    return (
        f"**{label} — caption-grounded summary:** {full_caption}\n\n"
        "The visual model could not verify the structured image reading."
    )
