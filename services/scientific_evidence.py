"""Document-grounded helpers for mixed figures and experimental evidence."""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from pathlib import Path

import fitz


PANEL_MARKER = re.compile(
    r"(?:^|(?<=[.!?])\s+)(?P<panels>[a-z](?:\s*[-–—,]\s*[a-z])*)\s+(?=[A-Z0-9])"
)


def _clean_pdf_text(text: str) -> str:
    value = str(text or "").replace("ﬁ", "fi").replace("ﬂ", "fl")
    value = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", value)
    return value


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
                "role": infer_panel_role(description),
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


def infer_panel_role(description: str) -> str:
    """Describe what a panel does without mistaking validation plots for images."""
    text = str(description or "").casefold()
    if re.search(r"\b(?:test|validation|held[ -]?out)\s+(?:data|dataset|set|result)", text):
        return "test_validation_result"
    if re.search(r"\b(?:experimental design|workflow|timeline|scheme|schematic)\b", text):
        return "experimental_design"
    if re.search(r"\b(?:representative images?|micrographs?|staining|histolog|microscop)\b", text):
        return "microscopy"
    if re.search(r"\b(?:screen|library)\b", text):
        return "screening_result"
    if re.search(r"\b(?:quantification|correlation|distribution|percentage|score|curve|plot|analysis)\b", text):
        return "quantitative_result"
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
        r"\b([A-Za-z][A-Za-z-]*(?:\s+[A-Za-z][A-Za-z-]*){1,9})\s*"
        r"\(([A-Z][A-Z0-9-]{1,12})\)",
        str(text or ""),
    ):
        candidate, acronym = re.sub(r"\s+", " ", match.group(1)).strip(), match.group(2)
        words = candidate.split()
        for start in range(len(words) - 1):
            suffix = words[start:]
            initials = "".join(
                word[0] for word in suffix
                if word.casefold() not in {"a", "an", "and", "for", "in", "of", "the", "to"}
            ).upper()
            if initials == re.sub(r"[^A-Z0-9]", "", acronym):
                glossary.setdefault(acronym, " ".join(suffix))
                break
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
        return "human_patient_tissue"
    if re.search(r"\b(?:mice|mouse|murine|rat|in vivo|animal model)\b", value):
        return "mouse_animal_tissue"
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
        return "in_vitro_human_cell_line"
    if re.search(r"\b(?:algorithm|classifier|computational|simulation|model training)\b", value):
        return "computational"
    return "other"


def classify_experimental_evidence(
    caption: str, results: str = "", methods: str = "",
) -> dict:
    """Classify one evidence object, giving its caption priority over page spillover."""
    parts = {
        "caption": classify_experimental_domain(caption),
        "results": classify_experimental_domain(results),
        "methods": classify_experimental_domain(methods),
    }
    decisive = {
        "in_vitro_human_cell_line", "mouse_animal_tissue", "human_patient_tissue",
    }
    domain = parts["caption"] if parts["caption"] in decisive else None
    if domain is None and parts["results"] in decisive:
        domain = parts["results"]
    if domain is None and parts["methods"] in decisive:
        domain = parts["methods"]
    conflicts = {
        value for key, value in parts.items()
        if key != "methods" and value in decisive and value != domain
    }
    if conflicts and parts["caption"] not in decisive:
        domain = "uncertain"
    return {
        "experimental_domain": domain or "uncertain",
        "evidence_sources": [key for key, value in parts.items() if value == domain],
        "component_classifications": parts,
    }


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
        ("compounds screened", r"\b(?:how many|number of).{0,35}\b(?:compounds?|drugs?)\b|\b(?:compounds?|drugs?).{0,35}\b(?:screened|tested)\b"),
        ("condition-specific hit counts", r"\b(?:specific|selective|unique|only).{0,45}\b(?:hits?|compounds?|drugs?|cell lines?)\b|\b\w+-specific\b.{0,45}\b(?:hits?|compounds?|drugs?)\b|\bhow many\b.{0,45}\b(?:specific|unique)\b.{0,35}\b(?:cell|condition|group)\b|\bhits?\b.{0,45}\b(?:cell|condition|group)"),
        ("shared hit count", r"\b(?:both|shared|overlap).{0,35}\b(?:hits?|compounds?|drugs?|active)\b|\b(?:hits?|compounds?|drugs?|active).{0,35}\b(?:both|shared|overlap)\b"),
        ("first experiment", r"\b(?:first|initial|earlier)\s+(?:experiment|stage|assay)\b|\b(?:two|both)\s+experiments?\b"),
        ("later screening experiment", r"\b(?:second|later|subsequent|screening|high-throughput)\s+(?:experiment|stage|assay|screen)?\b|\b(?:two|both)\s+experiments?\b"),
        ("changing nuclear features", r"\bwhich\s+(?:nuclear\s+)?features?\s+(?:change|differ)|\b(?:features?|measurements?)\b.{0,35}\b(?:senescence|changed?|differ)\b"),
        ("feature exceptions", r"\b(?:except|exception|did not change|not significant|unchanged)\b.{0,40}\bfeatures?\b|\bfeatures?\b.{0,40}\b(?:except|exception|unchanged)\b"),
        ("panel roles", r"\b(?:role|purpose|represents?)\b.{0,40}\bpanels?\b|\bpanels?\b.{0,40}\b(?:role|purpose|represents?)\b"),
        ("condition-specific outcomes", r"\b(?:conditions?|groups?)\b.{0,80}\b(?:cells?|treatments?|markers?|senescen\w*|result)\b|\b(?:cells?|treatments?)\b.{0,80}\b(?:conditions?|groups?)\b"),
        ("experimental domains", r"\b(?:cell culture|in vitro|cell line)\b.{0,120}\b(?:mouse|animal|patient|clinical|human tissue)\b|\b(?:mouse|animal)\b.{0,120}\b(?:patient|clinical|human tissue)\b"),
    )
    return [name for name, pattern in patterns if re.search(pattern, str(question or ""), re.I)]


def explicit_condition_outcomes(text: str) -> list[str]:
    """Return condition-level author statements without generalising their scope."""
    sentences = [
        re.sub(r"\s+", " ", value).strip()
        for value in re.split(r"(?<=[.!?])\s+", str(text or ""))
    ]
    condition = re.compile(
        r"\b(?:control|vehicle|growing|irradiat\w*|treated|induced|cells?|cohort|"
        r"DMSO|condition|group|young|old|patient|mice|mouse)\b", re.I,
    )
    outcome = re.compile(
        r"\b(?:significant|positive|negative|increase|decrease|higher|lower|"
        r"less than|more than|identified|predicted|differ|unchanged|excluded)\b|[<>]\s*\d", re.I,
    )
    rows = []
    for sentence in sentences:
        if 7 <= len(sentence.split()) <= 100 and condition.search(sentence) and outcome.search(sentence):
            if sentence not in rows:
                rows.append(sentence)
    return rows


def contradiction_check_condition_prose(
    answer: str, author_outcomes: list[str],
) -> tuple[str, list[str]]:
    """Remove categorical condition claims that lack a matching author statement."""
    evidence = " ".join(author_outcomes).casefold()
    sentences = re.split(r"(?<=[.!?])\s+", str(answer or ""))
    kept, removed = [], []
    for sentence in sentences:
        lower = sentence.casefold()
        categorical = re.search(r"\b(?:never|excludes?|solely)\b|\bonly\b|\bnot in\b", lower)
        if not categorical:
            kept.append(sentence)
            continue
        marker = categorical.group(0).strip()
        content = {
            token for token in re.findall(r"[a-z0-9][a-z0-9-]{3,}", lower)
            if token not in {"only", "never", "excludes", "excluded", "solely", "that", "with", "from"}
        }
        supported = False
        for outcome in author_outcomes:
            outcome_lower = outcome.casefold()
            outcome_tokens = set(re.findall(r"[a-z0-9][a-z0-9-]{3,}", outcome_lower))
            overlap = len(content & outcome_tokens) / max(1, min(len(content), len(outcome_tokens)))
            marker_supported = (
                marker in outcome_lower
                or (marker == "not in" and re.search(r"\b(?:not|no)\b", outcome_lower))
            )
            if marker_supported and overlap >= 0.45:
                supported = True
                break
        if supported or not evidence:
            kept.append(sentence)
        else:
            removed.append(sentence)
    return " ".join(kept).strip(), removed


def extract_explicit_classifier_taxonomy(text: str) -> list[dict]:
    """Extract classifier families only where the source states them explicitly."""
    value = str(text or "").replace("ﬁ", "fi").replace("ﬂ", "fl")
    value = re.sub(r"(?<=\w)\s*-\s+(?=\w)", "", value)
    value = re.sub(r"(?<=[a-z])\s+(?=(?:fi|fl)[a-z])", "", value)
    value = re.sub(r"\s+", " ", value)
    rows = []
    for match in re.finditer(
        r"\b([A-Z][A-Z0-9-]{1,15})\s*\(([^)]{3,80}?(?:based|classifier|model)[^)]*)\)",
        value,
    ):
        row = {"name": match.group(1), "source_description": match.group(2).strip()}
        if row not in rows:
            rows.append(row)
    return rows


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
    question: str = "",
) -> dict:
    """Collect deterministic caption, adjacent-page and direct-reference evidence."""
    with fitz.open(pdf_path) as document:
        page_numbers = {
            page for anchor in (visual_page, caption_page or visual_page)
            for page in (anchor - 1, anchor, anchor + 1)
            if 1 <= page <= len(document)
        }
        local_pages = {
            page: _clean_pdf_text(document[page - 1].get_text("text") or "")
            for page in sorted(page_numbers)
        }
        direct = []
        if figure_number:
            pattern = re.compile(
                rf"\b(?:Fig\.?|Figure)\s*{re.escape(str(figure_number))}\b",
                re.I,
            )
            for page_index, page in enumerate(document, start=1):
                text = _clean_pdf_text(page.get_text("text") or "")
                for match in pattern.finditer(text):
                    start, end = max(0, match.start() - 650), min(len(text), match.end() + 1100)
                    direct.append({"page": page_index, "text": re.sub(r"\s+", " ", text[start:end]).strip()})
        all_pages = [
            (index, _clean_pdf_text(page.get_text("text") or ""))
            for index, page in enumerate(document, start=1)
        ]
        whole_text = "\n".join(text for _, text in all_pages)
    slots = requested_answer_slots(question)
    query_terms = {
        token for token in re.findall(
            r"[a-z0-9][a-z0-9-]{2,}", f"{question} {' '.join(slots)}".casefold()
        )
        if token not in {
            "the", "and", "with", "from", "this", "that", "figure", "explain", "which",
        }
    }
    supporting = []
    for page, page_text in all_pages:
        if not slots:
            break
        compact = re.sub(r"\s+", " ", page_text)
        sentences = re.split(r"(?<=[.!?])\s+", compact)
        for index in range(0, len(sentences), 2):
            passage = " ".join(sentences[index:index + 4]).strip()
            lower = passage.casefold()
            score = sum(term in lower for term in query_terms)
            if figure_number and re.search(
                rf"\b(?:fig\.?|figure)\s*{re.escape(str(figure_number))}\b", passage, re.I,
            ):
                score += 4
            if re.search(r"\b(?:results?|methods?|screened|significantly|identified|treated)\b", lower):
                score += 1
            if "condition-specific hit counts" in slots and re.search(
                r"\b(?:amongst those|hits?|induced senescence only|only in|both (?:cells?|groups?))\b", lower,
            ):
                score += 7
            if "compounds screened" in slots and re.search(r"\bscreen(?:ed|ing)\b.{0,100}\b\d+\b", lower):
                score += 7
            if "condition-specific outcomes" in slots:
                conditions = re.findall(
                    r"\b(?:(?i:growing|irradiat\w*|[a-z0-9-]+-treated|control)|[A-Z][A-Z0-9-]{2,})\b",
                    passage,
                )
                if len({value.casefold() for value in conditions}) >= 2 and re.search(
                    r"\b(?:significant|positive|negative|less than|identified|predicted)\b", lower,
                ):
                    score += 7
            if score >= 3:
                supporting.append({"page": page, "score": score, "text": passage[:2200]})
    supporting = sorted(
        supporting, key=lambda row: (row["score"], -row["page"]), reverse=True,
    )[:24]
    panel_map = extract_caption_panels(full_caption)
    evidence_text = (
        f"TARGET FIGURE CAPTION (complete):\n{full_caption}\n\n"
        + "\n\n".join(
            f"LOCAL PDF PAGE {page}:\n{text}" for page, text in local_pages.items()
        )
        + "\n\nDIRECT FIGURE REFERENCES:\n"
        + "\n".join(f"Page {row['page']}: {row['text']}" for row in direct)
        + "\n\nQUESTION-TARGETED RESULTS/METHODS PASSAGES:\n"
        + "\n".join(f"Page {row['page']}: {row['text']}" for row in supporting)
    )
    condition_outcomes = explicit_condition_outcomes(evidence_text)
    taxonomy = extract_explicit_classifier_taxonomy(evidence_text)
    return {
        "full_caption": full_caption,
        "panel_map": panel_map,
        "local_pages": sorted(local_pages),
        "direct_references": direct,
        "supporting_passages": supporting,
        "requested_answer_slots": slots,
        "explicit_condition_outcomes": condition_outcomes,
        "explicit_classifier_taxonomy": taxonomy,
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
                "role": caption_panel.get("role", "other"),
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
