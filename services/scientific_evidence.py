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
    if re.search(r"\banalysis of training (?:data|datasets?|sets?)\b", text):
        return "training_result"
    if re.search(r"\b(?:development|construction) of training (?:data|datasets?|sets?)\b", text):
        return "training_workflow"
    if "nuclear feature" in text and "distribution" in text:
        return "nuclear_feature_distribution"
    if re.search(r"\b(?:quantification|percentage)\b.{0,100}\b(?:marker|positive|activity|brdu|p21|p53|gal)\b", text):
        return "marker_quantification"
    if re.search(r"\b(?:experimental design|workflow|timeline|scheme|schematic)\b", text):
        return "experimental_design"
    if re.search(r"\b(?:representative images?|micrographs?|staining|histolog|microscop)\b", text):
        return "microscopy"
    if re.search(r"\b(?:screen|library)\b", text):
        return "screening_result"
    if re.search(r"\b(?:quantification|correlation|distribution|percentage|score|curve|plot|analysis)\b", text):
        return "quantitative_result"
    return "other"


def authoritative_panel_role_map(panel_map: list[dict]) -> dict[str, dict]:
    """Return the caption-defined identity for every explicitly mapped panel."""
    return {
        str(row["panel"]).casefold(): {
            "panel": str(row["panel"]).casefold(),
            "role": row.get("role", "other"),
            "visual_type": row.get("visual_type", "other"),
            "caption_description": row.get("caption_description", ""),
            "authority": "full_caption",
        }
        for row in panel_map
        if row.get("panel") and row.get("caption_description")
    }


def render_authoritative_panel_roles(panel_map: list[dict]) -> str:
    rows = authoritative_panel_role_map(panel_map).values()
    if not rows:
        return ""
    lines = ["**Caption-defined panel roles**"]
    lines.extend(
        f"- **Panel {row['panel']}:** {row['caption_description']}"
        for row in rows
    )
    return "\n".join(lines)


def enforce_authoritative_panel_prose(answer: str, panel_map: list[dict]) -> tuple[str, list[str]]:
    """Replace generated panel assignments with the deterministic full-caption map."""
    if not panel_map:
        return str(answer or ""), []
    removed = []
    kept = []
    for block in re.split(r"\n+", str(answer or "")):
        if not re.search(r"\bpanels?\s+[a-z](?:\s*[-â€“â€”]\s*[a-z])?\b", block, re.I):
            kept.append(block)
            continue
        fragments = re.split(r"(?<=[.!?])\s+", block)
        retained_fragments = []
        for fragment in fragments:
            if re.search(r"\bpanels?\s+[a-z](?:\s*[-â€“â€”]\s*[a-z])?\b", fragment, re.I):
                removed.append(fragment)
            else:
                retained_fragments.append(fragment)
        if retained_fragments:
            kept.append(" ".join(retained_fragments))
    remainder = "\n".join(value for value in kept if value.strip()).strip()
    authoritative = render_authoritative_panel_roles(panel_map)
    return (
        f"{authoritative}\n\n{remainder}".strip() if remainder else authoritative,
        removed,
    )


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
        # Statistical annotations and other compact measurements can follow a
        # model name in parentheses (for example ``BAEM (r = ...; p < ...)``).
        # They are evidence, not an attempted expansion of the acronym.
        if re.search(r"(?:[=<>]|\b[rap]\s*[=:<>]|\bCI\b|\bn\s*=)", expansion, re.I):
            return match.group(0)
        grounded = glossary.get(acronym)
        return match.group(0) if grounded and grounded.casefold() == expansion.casefold() else acronym
    value = re.sub(r"\b([A-Z][A-Z0-9-]{1,12})\s*\(([^)]{3,100})\)", replace, str(answer or ""))

    # Also handle the inverse form "invented expansion (ABC)". If the source
    # does not explicitly define that wording, retain only the stable acronym.
    reverse = re.compile(
        r"(?P<prefix>^|[,;:]\s*)"
        r"(?P<expansion>[A-Z][A-Za-z-]*(?:\s+[A-Za-z][A-Za-z-]*){1,6})\s*"
        r"\((?P<acronyms>[A-Z][A-Z0-9-]{1,12}(?:\s*/\s*[A-Z][A-Z0-9-]{1,12})*)\)",
        re.MULTILINE,
    )

    def replace_reverse(match):
        expansion = re.sub(r"\s+", " ", match.group("expansion")).strip()
        acronyms = re.split(r"\s*/\s*", match.group("acronyms"))
        grounded = [glossary.get(acronym) for acronym in acronyms]
        if grounded and all(
            item and item.casefold() == expansion.casefold() for item in grounded
        ):
            return match.group(0)
        return f"{match.group('prefix')}{'/'.join(acronyms)}"

    return reverse.sub(replace_reverse, value)


def remove_unsupported_acronym_names(answer: str, source_text: str) -> tuple[str, list[str]]:
    """Remove classifier/model identifiers that never occur in supplied evidence."""
    supported = {
        token for token in re.findall(r"\b[A-Z][A-Z0-9-]{2,15}\b", str(source_text or ""))
    }
    removed = []
    value = str(answer or "")
    for match in list(re.finditer(r"\b[A-Z][A-Z0-9-]{2,15}\b", value)):
        token = match.group(0)
        if token in supported or token in {"PDF", "JSON", "RAG", "DOI", "CI", "ANOVA"}:
            continue
        value = re.sub(rf"\b{re.escape(token)}\b", "", value)
        removed.append(token)
    value = re.sub(r"\s+([,.;:])", r"\1", value)
    value = re.sub(r" {2,}", " ", value)
    return value.strip(), list(dict.fromkeys(removed))


def correct_unsupported_measurement_entities(answer: str, source_text: str) -> tuple[str, list[dict]]:
    """Correct a measurement entity only when source evidence gives one unique alternative."""
    entities = ("cells", "nuclei", "pixels", "patients", "samples")
    value = str(answer or "")
    source = str(source_text or "")
    corrections = []
    sentence_pattern = re.compile(r"[^.!?]*(?:percentage|proportion)\s+of\s+(?:" + "|".join(entities) + r")[^.!?]*[.!?]?", re.I)
    for sentence_match in list(sentence_pattern.finditer(value)):
        sentence = sentence_match.group(0)
        entity_match = re.search(r"\b(?:percentage|proportion)\s+of\s+(" + "|".join(entities) + r")\b", sentence, re.I)
        if not entity_match:
            continue
        observed = entity_match.group(1).casefold()
        acronyms = re.findall(r"\b[A-Z][A-Z0-9-]{1,12}\b", sentence)
        if not acronyms:
            continue
        grounded_entities = set()
        for acronym in acronyms:
            for anchor in re.finditer(rf"\b{re.escape(acronym)}\b", source):
                window = source[max(0, anchor.start() - 650):anchor.end() + 650]
                grounded_entities.update(
                    found.casefold() for found in re.findall(
                        r"\b(?:percentage|proportion)\s+(?:based on|of)\s+(?:the\s+)?(?:number of\s+)?(cells|nuclei|pixels|patients|samples)\b",
                        window, re.I,
                    )
                )
        if observed in grounded_entities or len(grounded_entities) != 1:
            continue
        replacement = next(iter(grounded_entities))
        corrected_sentence = re.sub(
            r"\b((?:percentage|proportion)\s+of\s+)" + re.escape(observed) + r"\b",
            rf"\1{replacement}", sentence, flags=re.I,
        )
        value = value[:sentence_match.start()] + corrected_sentence + value[sentence_match.end():]
        corrections.append({"from": observed, "to": replacement, "acronyms": acronyms})
        break
    return value, corrections


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


def experimental_evidence_object(
    document: str, figure_number: str | None, page,
    caption: str = "", results: str = "", methods: str = "", panel: str | None = None,
) -> dict:
    """Create provenance only from an explicitly resolved visual-index record."""
    classification = classify_experimental_evidence(caption, results, methods)
    source_text = caption or results or methods
    base = experimental_provenance(source_text)
    resolved_number = str(figure_number).strip() if figure_number not in (None, "") else None
    provenance_sources = []
    if caption:
        provenance_sources.append("full_caption")
    if results:
        provenance_sources.append("results_context")
    if methods:
        provenance_sources.append("methods_context")
    species = base.get("species")
    if not species and classification["experimental_domain"] == "human_patient_tissue" and re.search(
        r"\b(?:patients?|human)\b", source_text, re.I,
    ):
        species = "human"
    return {
        "document": document,
        "figure_number": resolved_number,
        "figure_label": f"Figure {resolved_number}" if resolved_number else "figure number not resolved",
        "figure_or_panel": f"Figure {resolved_number}" if resolved_number else "figure number not resolved",
        "panel": panel,
        "page": page,
        "experimental_domain": classification["experimental_domain"],
        "species": species,
        "sample_type": base.get("cell_line_or_tissue"),
        "source_text": source_text,
        "source_provenance": provenance_sources or ["document_text"],
        "classification_evidence": classification,
    }


def requested_answer_slots(question: str) -> list[str]:
    value = str(question or "")
    patterns = (
        # A paper title may itself contain "nuclear features". Only treat it as
        # an answer slot when the question grammatically asks for those fields.
        ("features", r"\b(?:which|what|list|name|exact|include|including|cover)\b.{0,60}\b(?:nuclear\s+)?features?\b|\b(?:nuclear\s+)?features?\b\s*(?:that\s+|were\s+|are\s+)?(?:used|included|measured|changed?|differ(?:ed)?)\b|\bmethods?\b.{0,220}\bnuclear features?\b"),
        ("library construction", r"\blibrar(?:y|ies)\b|\btraining sets?\s+(?:assembled|constructed)\b"),
        ("training cell counts", r"\b(?:number of cells|cell counts?|how many cells)\b"),
        ("CT split", r"\b(?:CT|classification tree).{0,40}\b(?:split|test|proportion)\b"),
        ("RF split", r"\b(?:RF|random forest).{0,40}\b(?:split|test|proportion)\b"),
        ("CT overfitting method", r"\b(?:overfitting|over-fitting|pruning|alpha)\b"),
        ("RF threshold", r"\b(?:RF|random forest).{0,40}\bthreshold\b|\bthreshold.{0,40}(?:RF|random forest)\b"),
        ("inclusion criteria", r"\b(?:inclusion|exclusion|criteria|threshold for these samples)\b|"
         r"\bcircularity\b.{0,40}\bthreshold\b|\bthreshold\b.{0,40}\bcircularity\b"),
        ("compounds screened", r"\b(?:how many|number of).{0,35}\b(?:compounds?|drugs?)\b|\b(?:compounds?|drugs?).{0,35}\b(?:screened|tested|screening)\b|\bcompound screening\b.{0,80}\b(?:all|counts?|results?)\b"),
        ("condition-specific hit counts", r"\bhit counts?\b|\b(?:specific|selective|unique|only).{0,45}\b(?:hits?|compounds?|drugs?|cell lines?)\b|\b\w+-specific\b.{0,45}\b(?:hits?|compounds?|drugs?)\b|\bhow many\b.{0,45}\b(?:specific|unique)\b.{0,35}\b(?:cell|condition|group)\b|\bhits?\b.{0,45}\b(?:cell|condition|group)"),
        ("shared hit count", r"\b(?:all\s+)?hit counts?\b|\b(?:both|shared|overlap).{0,35}\b(?:hits?|compounds?|drugs?|active)\b|\b(?:hits?|compounds?|drugs?|active).{0,35}\b(?:both|shared|overlap)\b"),
        ("first experiment", r"\b(?:first|initial|earlier|stage\s*1)\s+(?:experiment|stage|assay|senolytic)?\b|\b(?:two|both)\s+experiments?\b"),
        ("later screening experiment", r"\b(?:second|later|subsequent|screening|high-throughput|stage\s*2)\s+(?:experiment|stage|assay|screen|compound)?\b|\b(?:two|both)\s+experiments?\b"),
        ("changing nuclear features", r"\bwhich\s+(?:nuclear\s+)?features?\s+(?:change|differ)|\b(?:features?|measurements?)\b.{0,35}\b(?:changed?|differ(?:ed|ent)?)\b"),
        ("feature exceptions", r"\b(?:except|exception|did not change|not significant|unchanged)\b.{0,40}\bfeatures?\b|\bfeatures?\b.{0,40}\b(?:except|exception|unchanged)\b"),
        ("panel roles", r"\b(?:role|purpose|represents?)\b.{0,40}\bpanels?\b|\bpanels?\b.{0,40}\b(?:role|purpose|represents?)\b"),
        ("condition-specific outcomes", r"\b(?:conditions?|groups?)\b.{0,80}\b(?:cells?|treatments?|markers?|senescen\w*|result)\b|\b(?:cells?|treatments?)\b.{0,80}\b(?:conditions?|groups?)\b"),
        ("experimental domains", r"\b(?:cell culture|in vitro|cell line)\b.{0,120}\b(?:mouse|animal|patient|clinical|human tissue)\b|\b(?:mouse|animal)\b.{0,120}\b(?:patient|clinical|human tissue)\b"),
        ("correlation statistics", r"\b(?:exact\s+)?correlation\s+(?:statistics?|values?)\b|\breport\b.{0,60}\bcorrelation\b.{0,60}\b(?:r|p)\b"),
        ("performance metrics", r"\bperformance(?:\s+metrics?)?\b|\b(?:precision|accuracy|recall|f1)\b.{0,80}\b(?:metric|performance|compare)"),
        ("classifier identities", r"\bclassifier\s+(?:famil(?:y|ies)|identit(?:y|ies)|names?)\b|\b(?:all\s+)?classifiers?\b.{0,80}\b(?:include|including|compare|control)"),
        ("experimental controls", r"\b(?:all\s+)?controls?\b|\b(?:vehicle|control)\s+(?:group|condition|versus|vs\.?)\b"),
        ("candidate validation", r"\bcandidates?\b.{0,70}\bvalidat(?:e|ed|ion)\b|\bvalidat(?:e|ed|ion)\b.{0,70}\bcandidates?\b"),
        ("toxicity distinction", r"\b(?:simple\s+)?toxicit(?:y|ies)\b|\bdistinguish\b.{0,70}\btoxic"),
        ("downstream validation", r"\b(?:downstream|one[- ]two[- ]punch)\b.{0,80}\bvalidat(?:e|ed|ion)?\b|\bdownstream\s+(?:experiment|assay)\b"),
        ("score construction", r"\b(?:how\b.{0,30})?(?:score|index|metric)\b.{0,50}\b(?:construct(?:ed|ion)|calculat(?:ed|ion)|deriv(?:ed|ation)|built)\b|\bconstructed\b.{0,50}\b(?:score|index|metric)\b"),
        ("comparison outcomes", r"\bwhat\b.{0,50}\b(?:results?|score|comparison)\b.{0,40}\bshow\b|\bwhat (?:happened|changed)\b|\b(?:higher|lower|increase|decrease|direction)\b.{0,60}\b(?:comparison|result)"),
        ("sample size", r"\b(?:cohort|patients?|samples?)\b.{0,80}\b(?:size|number|n\s*=|included|inclusion)\b|\bn\s*=\s*\d+\b"),
    )
    return [name for name, pattern in patterns if re.search(pattern, value, re.I)]


def explicit_condition_outcomes(text: str) -> list[str]:
    """Return condition-level author statements without generalising their scope."""
    protected = re.sub(r"\b(Fig|fig)\.", r"\1", str(text or ""))
    sentences = [
        re.sub(r"\s+", " ", value).strip()
        for value in re.split(r"(?<=[.!?])\s+", protected)
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


def extract_condition_tuples(text: str) -> list[dict]:
    """Keep treatment-specific condition evidence separate before synthesis."""
    compact = re.sub(r"\s+", " ", str(text or ""))
    compact = re.sub(r"\b(Fig|fig)\.", r"\1", compact)
    sentences = re.split(r"(?<=[.!?])\s+", compact)
    candidates: dict[str, set[str]] = {}

    def add(name: str, alias: str | None = None):
        cleaned = re.sub(r"\s+", " ", name).strip(" ,;:()")
        if not cleaned or len(cleaned.split()) > 5:
            return
        if re.fullmatch(r"\d+(?:\.\d+)?%?", cleaned):
            return
        if cleaned.casefold() in {
            "and", "or", "the", "treated", "cells", "supplementary", "supplementary table", "fbs",
        }:
            return
        key = cleaned.casefold()
        candidates.setdefault(key, set()).add(cleaned)
        if alias:
            candidates[key].add(alias)

    for match in re.finditer(r"\b([A-Za-z0-9-]+)-(?:treated|induced)\b", compact):
        add(match.group(1))
    for match in re.finditer(r"\b(irradiated|growing|quiescent)\s+cells?\b", compact, re.I):
        add(match.group(1))
    for match in re.finditer(r"\b(growing|irradiated|quiescent)\s*\(([^)]{1,20})\)", compact, re.I):
        add(match.group(1), match.group(2).strip())
    for match in re.finditer(r"\bsenescent\s*\(([^)]{1,100})\)", compact, re.I):
        for part in re.split(r"[;,]", match.group(1)):
            tokens = part.strip().split()
            if not tokens:
                continue
            alias = tokens[-1] if len(tokens[-1]) <= 3 else None
            name = " ".join(tokens[:-1]) if alias and len(tokens) > 1 else tokens[0]
            add(name, alias)
    for match in re.finditer(
        r"\bsenescence (?:caused|induced) (?:by|with)\s+([^.;]{2,100})", compact, re.I,
    ):
        for token in re.findall(r"\b[A-Za-z][A-Za-z0-9-]{2,}\b", match.group(1)):
            if token.casefold() not in {"the", "aurora", "kinase", "inhibitor", "and", "cells"}:
                add(token)

    rows = []
    for key, aliases in candidates.items():
        statements = []
        for sentence in sentences:
            if any(re.search(rf"\b{re.escape(alias)}\b", sentence, re.I) for alias in aliases):
                if re.search(
                    r"\b(?:significant|positive|negative|increase|decrease|less than|"
                    r"identified|predicted|senescent|damage|arrested)\b", sentence, re.I,
                ):
                    statements.append(sentence.strip())
        if statements:
            rows.append({
                "condition": sorted(aliases, key=len, reverse=True)[0],
                "aliases": sorted(aliases),
                "author_statements": list(dict.fromkeys(statements))[:5],
                "authority": "explicit_document_text",
            })
    return rows


def contradiction_check_condition_prose(
    answer: str, author_outcomes: list[str], condition_tuples: list[dict] | None = None,
) -> tuple[str, list[str]]:
    """Remove categorical condition claims that lack a matching author statement."""
    evidence = " ".join(author_outcomes).casefold()
    sentences = re.split(r"(?<=[.!?])\s+", str(answer or ""))
    kept, removed = [], []
    for sentence in sentences:
        lower = sentence.casefold()
        categorical = re.search(
            r"\b(?:never|excludes?|solely|only|none|all|does not|do not|not in|no)\b",
            lower,
        )
        if not categorical:
            kept.append(sentence)
            continue
        marker = categorical.group(0).strip()
        named_conditions = [
            row for row in condition_tuples or []
            if any(
                re.search(rf"\b{re.escape(alias)}\b", sentence, re.I)
                for alias in row.get("aliases", [])
            )
        ]
        treatment_conditions = [
            row for row in condition_tuples or []
            if row.get("condition", "").casefold() not in {"growing", "quiescent", "irradiated"}
        ]
        generic_senescence_collapse = (
            bool(re.search(r"\bsenescen\w*\b", lower))
            and len(treatment_conditions) >= 2
            and not any(row in named_conditions for row in treatment_conditions)
        )
        if generic_senescence_collapse:
            removed.append(sentence)
            continue
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
                or (
                    marker in {"not in", "does not", "do not", "none", "no"}
                    and re.search(r"\b(?:not|no|without)\b", outcome_lower)
                )
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
    """Build a glossary from explicit definitions, never identifier similarity."""
    value = str(text or "").replace("ﬁ", "fi").replace("ﬂ", "fl")
    value = re.sub(r"(?<=\w)\s*-\s+(?=\w)", "", value)
    value = re.sub(r"(?<=[a-z])\s+(?=(?:fi|fl)[a-z])", "", value)
    value = re.sub(r"\s+", " ", value)
    definitions: dict[str, list[str]] = {}

    def add(name: str, description: str):
        cleaned_name = name.strip()
        cleaned_description = re.sub(r"\s+", " ", description).strip(" .;:,()")
        if not cleaned_name or not cleaned_description:
            return
        definitions.setdefault(cleaned_name, [])
        if cleaned_description.casefold() not in {
            item.casefold() for item in definitions[cleaned_name]
        }:
            definitions[cleaned_name].append(cleaned_description)

    for match in re.finditer(
        r"\b([A-Z][A-Z0-9-]{1,15})\s*\(([^)]{3,80}?(?:based|classifier|model)[^)]*)\)",
        value,
    ):
        add(match.group(1), match.group(2))
    for match in re.finditer(
        r"\b((?:(?i:[A-Za-z][A-Za-z-]*\s+)){1,7}(?i:algorithm|classifier|model))\s*"
        r"\(([A-Z][A-Z0-9-]{1,15})\)",
        value,
    ):
        add(match.group(2), match.group(1))
    rows = []
    for name, descriptions in definitions.items():
        semantic_definitions = {
            term for item in descriptions for term in (
                "consensus" if "consensus" in item.casefold() else "",
                "clustering" if "clustering" in item.casefold() else "",
                "classification tree" if "classification tree" in item.casefold() else "",
                "random forest" if "random forest" in item.casefold() else "",
            ) if term
        }
        rows.append({
            "name": name,
            "source_description": descriptions[0],
            "source_descriptions": descriptions,
            "ambiguous_source_wording": len(semantic_definitions) > 1,
            "relationships": [],
        })
    return rows


def validate_classifier_taxonomy_prose(answer: str, taxonomy: list[dict]) -> tuple[str, list[str]]:
    """Remove unsupported family claims and expose conflicting source definitions."""
    sentences = re.split(r"(?<=[.!?])\s+", str(answer or ""))
    removed, kept, ambiguity_notes = [], [], []
    by_name = {row.get("name", ""): row for row in taxonomy if row.get("name")}
    for sentence in sentences:
        matched = [name for name in by_name if re.search(rf"\b{re.escape(name)}\b", sentence)]
        unsupported = False
        for name in matched:
            row = by_name[name]
            explicit = " ".join(
                row.get("source_descriptions") or [row.get("source_description", "")]
            ).casefold()
            canonical = str(row.get("source_description", "")).casefold()
            acknowledges_ambiguity = bool(re.search(
                r"\b(?:ambiguous|inconsistent|wording)\b", sentence, re.I,
            ))
            for claim in ("umbrella", "family", "clustering", "consensus"):
                if re.search(rf"\b{claim}\b", sentence, re.I) and claim not in explicit:
                    unsupported = True
                if (
                    row.get("ambiguous_source_wording") and claim in sentence.casefold()
                    and claim not in canonical and not acknowledges_ambiguity
                ):
                    unsupported = True
            if row.get("ambiguous_source_wording") and not acknowledges_ambiguity:
                ambiguity_notes.append(
                    f"The source wording for {name} is inconsistent: "
                    + "; ".join(row.get("source_descriptions", [])) + "."
                )
        if unsupported:
            removed.append(sentence)
        else:
            kept.append(sentence)
    value = " ".join(kept).strip()
    for note in dict.fromkeys(ambiguity_notes):
        if note.casefold() not in value.casefold():
            value = f"{value.rstrip()}\n\n{note}".strip()
    return value, removed


def unsupported_source_completion(text: str) -> bool:
    return bool(re.search(
        r"\b(?:typically implied|standard practice would be|presumably)\b",
        str(text or ""), re.I,
    ))


_SLOT_EVIDENCE_PATTERNS = {
    "features": (
        r"\bnuclear features?\b.{0,260}\b(?:including|used|were)\b",
        r"\b(?:area|radius|compactness|ratio|displacement|elongation|form factor)\b(?:.{0,80},){2}",
    ),
    "changing nuclear features": (
        r"\bnuclear features?\b.{0,260}\b(?:including|different|differed|change)\b",
        r"\b(?:different|differed|change[sd]?)\b.{0,180}\bnuclear features?\b",
    ),
    "feature exceptions": (
        r"\b(?:except|exception|unchanged|not significantly different)\b.{0,180}\b(?:feature|factor|parameter)\b",
        r"\b(?:feature|factor|parameter)\b.{0,180}\b(?:except|exception|unchanged|not significantly different)\b",
    ),
    "compounds screened": (
        r"\bscreen(?:ed|ing)\b.{0,180}\b\d+\s+(?:compounds?|drugs?)\b",
        r"\b\d+\s+(?:compounds?|drugs?)\b.{0,180}\bscreen(?:ed|ing)\b",
    ),
    "condition-specific hit counts": (
        r"\b\d+\b.{0,180}\b(?:only|specific|selective)\b.{0,120}\b(?:cells?|cell line|condition)\b",
        r"\b(?:only|specific|selective)\b.{0,180}\b\d+\b.{0,120}\b(?:compounds?|drugs?|hits?)\b",
    ),
    "shared hit count": (
        r"\b\d+\b.{0,140}\b(?:both|shared|overlap)\b",
        r"\b(?:both|shared|overlap)\b.{0,140}\b\d+\b",
    ),
    "condition-specific outcomes": (
        r"\b(?:significant|positive|negative|less than|identified|predicted)\b.{0,220}\b(?:treated|irradiat|condition|cells?)\b",
        r"\b(?:treated|irradiat|condition|cells?)\b.{0,220}\b(?:significant|positive|negative|less than|identified|predicted)\b",
    ),
    "library construction": (
        r"\b(?:training|parameter)\s+librar(?:y|ies)\b.{0,400}\b(?:plates?|wells?|control|random)",
        r"\b(?:plates?|wells?)\b.{0,300}\b(?:training sets?|librar(?:y|ies))\b",
        r"\bindependent training sets?\b.{0,240}\brandomi[sz]ations?\b",
    ),
    "training cell counts": (
        r"\b(?:randomly selecting|selected)\b.{0,180}\b\d[\d,]*\s+(?:normal|treated)?\s*cells?\b",
        r"\b\d[\d,]*\s+(?:normal|treated)\s+cells?\b",
    ),
    "CT split": (
        r"\b(?:classification tree|CT)(?:-based)?\b.{0,240}\b(?:test size|split|training set)\b",
    ),
    "RF split": (
        r"\b(?:random forest|RF)(?:-based)?\b.{0,240}\b(?:test size|split|training set)\b",
    ),
    "CT overfitting method": (
        r"\b(?:cost complexity|prun(?:e|ed|ing)|optimal\s+alpha)\b.{0,180}\b(?:over\s*fitting|classification tree|CT)\b",
        r"\b(?:classification tree|CT)\b.{0,260}\b(?:cost complexity|prun(?:e|ed|ing)|alpha)\b",
    ),
    "RF threshold": (
        r"\b(?:random forest|RF)\b.{0,260}\b(?:probability|values?)\s*[><=]+\s*\d",
        r"\bsenescence probability\b.{0,120}[><=]+\s*\d",
    ),
    "inclusion criteria": (
        r"\b(?:included|excluded|inclusion|threshold)\b.{0,360}\b(?:samples?|cells?|nuclei|patients?)\b",
        r"\b(?:samples?|nuclei|patients?)\b.{0,360}\b(?:included|excluded|threshold)\b",
    ),
    "correlation statistics": (
        r"\bcorrelation\b.{0,360}\br\s*[=:]\s*\d",
        r"\br\s*[=:]\s*\d.{0,120}\bp\s*[<=>:]\s*\d",
    ),
    "performance metrics": (
        r"\b(?:precision|accuracy|recall|F\s*1)\b.{0,360}\b(?:classifier|performance|score|heatmap)",
    ),
    "classifier identities": (
        r"\b(?:classification|decision)\s+tree\b.{0,420}\brandom forest\b",
        r"\b(?:classifiers?|algorithms?)\b.{0,420}\b(?:voting|consensus|general model)\b",
    ),
    "experimental controls": (
        r"\b(?:control|vehicle|DMSO)\b.{0,320}\b(?:treated|experiment|comparison|cells?|mice)\b",
    ),
    "candidate validation": (
        r"\b(?:selected|candidate)\s+(?:drugs?|compounds?)\b.{0,520}\b(?:SA-?beta-?Gal|p21|BrdU|validat)",
        r"\b(?:SA-?beta-?Gal|p21|BrdU)\b.{0,520}\b(?:selected|candidate)\s+(?:drugs?|compounds?)\b",
        r"\bvalidat(?:e|ed|ion)\b.{0,420}\b(?:drugs?|compounds?|senescence)\b",
    ),
    "toxicity distinction": (
        r"\b(?:toxic|toxicity|viability)\b.{0,360}\b(?:senescen|cell count|excluded|filter)",
        r"\b(?:cell cycle arrest|SASP|p21|SA-?beta-?Gal)\b.{0,420}\b(?:senescen|induction|A549|IMR90)\b",
        r"\b(?:senescen|induction|A549|IMR90)\b.{0,420}\b(?:cell cycle arrest|SASP|p21|SA-?beta-?Gal)\b",
    ),
    "downstream validation": (
        r"\b(?:one[- ]two[- ]punch|senolytic|downstream)\b.{0,420}\b(?:validat|treated|reduced|activity)",
        r"\b(?:one[- ]two[- ]punch|combined with senolytics?)\b.{0,520}\b(?:ABT|sensiti[sz]|pre-?treat)",
        r"\b(?:pre-?treat|sensiti[sz])\w*\b.{0,520}\b(?:senolytic|ABT|one[- ]two[- ]punch)",
    ),
    "score construction": (
        r"\b(?:cell|tissue)\s+senescence score\b.{0,500}\b(?:percentage|range|values?|nuclear|construct|calculat)",
        r"\b(?:CSS|TSS)\b.{0,500}\b(?:percentage|range|values?|nuclear|construct|calculat)",
    ),
    "comparison outcomes": (
        r"\b(?:higher|lower|increase[sd]?|decrease[sd]?|reduced|induced)\b.{0,360}\b(?:score|TSS|marker|positive cells?)\b",
        r"\b(?:score|TSS|marker|positive cells?)\b.{0,360}\b(?:higher|lower|increase[sd]?|decrease[sd]?|reduced|induced)\b",
    ),
    "sample size": (
        r"\b(?:n\s*=\s*\d+|\d+\s+(?:patients?|samples?))\b.{0,300}\b(?:included|cohort|samples?|patients?)\b",
    ),
}


def ground_retrieved_answer_slots(question: str, selected_results: list[dict]) -> dict[str, dict]:
    """Resolve requested slots only against the chunks selected for display."""
    slots = requested_answer_slots(question)
    if not slots:
        return {}
    pages = [
        (int(row.get("page")) if str(row.get("page", "")).isdigit() else index,
         str(row.get("document", "")))
        for index, row in enumerate(selected_results, start=1)
        if str(row.get("document", "")).strip()
    ]
    return resolve_slot_evidence(slots, question, None, [], pages, [])


def _evidence_windows(all_pages: list[tuple[int, str]]) -> list[dict]:
    rows = []
    for page, page_text in all_pages:
        compact = re.sub(r"\s+", " ", page_text).strip()
        sentences = re.split(r"(?<=[.!?])\s+", compact)
        for index in range(len(sentences)):
            text = " ".join(sentences[index:index + 3]).strip()
            if text:
                rows.append({"page": page, "text": text[:2400]})
    return rows


def extract_experiment_stages(panel_map: list[dict], windows: list[dict]) -> list[dict]:
    """Build ordered experiment stages from explicit caption design panels."""
    starters = [
        index for index, row in enumerate(panel_map)
        if row.get("role") == "experimental_design"
    ]
    if not starters:
        return []
    stages = []
    for stage_index, start in enumerate(starters):
        end = starters[stage_index + 1] if stage_index + 1 < len(starters) else len(panel_map)
        rows = panel_map[start:end]
        caption_text = " ".join(
            row.get("caption_description", "") for row in rows
            if row.get("caption_description")
        )
        terms = {
            token for token in re.findall(r"[a-z0-9][a-z0-9-]{3,}", caption_text.casefold())
            if token not in {
                "experimental", "design", "representative", "images", "percentage",
                "cells", "cell", "figure", "panel", "analysis", "results", "after",
                "using", "with", "from", "different", "treatment",
            }
        }
        ranked = []
        for window in windows:
            lower = window["text"].casefold()
            score = sum(term in lower for term in terms)
            threshold = max(3, min(5, len(terms) // 5))
            if score < threshold:
                continue
            # Results paragraphs are more useful than repeated caption windows
            # for completing an experiment-stage answer slot.
            if re.search(
                r"\b(?:selectively|identified|predicted|reduced|increased|"
                r"significant|screened|only|both)\b", lower,
            ):
                score += 3
            ranked.append((score, window))
        ranked.sort(key=lambda value: (value[0], -value[1]["page"]), reverse=True)
        selected_windows = []
        stage_page_counts: dict[int, int] = {}
        for _, row in ranked:
            page = int(row["page"])
            if stage_page_counts.get(page, 0) >= 2:
                continue
            selected_windows.append(row)
            stage_page_counts[page] = stage_page_counts.get(page, 0) + 1
            if len(selected_windows) >= 4:
                break
        evidence = [{
            "page": row["page"], "source": "document_text", "text": row["text"],
        } for row in selected_windows]
        evidence.append({
            "page": None,
            "source": "full_caption",
            "text": caption_text,
        })
        stages.append({
            "stage": stage_index + 1,
            "start_panel": rows[0].get("panel") if rows else None,
            "end_panel": rows[-1].get("panel") if rows else None,
            "evidence": evidence,
        })
    return stages


def resolve_slot_evidence(
    slots: list[str], question: str, figure_number: str | None,
    panel_map: list[dict], all_pages: list[tuple[int, str]], direct: list[dict],
) -> dict[str, dict]:
    """Search local caption, exact figure references, Results, then Methods per slot."""
    windows = _evidence_windows(all_pages)
    stages = extract_experiment_stages(panel_map, windows)
    records: dict[str, dict] = {}
    question_terms = {
        token for token in re.findall(r"[a-z0-9][a-z0-9-]{3,}", str(question or "").casefold())
        if token not in {
            "figure", "explain", "which", "what", "with", "from", "that", "this",
            "detection", "senescence", "using", "machine", "learning", "algorithms",
            "based", "nuclear", "features", "paper",
        }
    }
    question_acronyms = list(dict.fromkeys(re.findall(
        r"\b[A-Z][A-Z0-9-]{1,14}\b", str(question or ""),
    )))
    for slot in slots:
        if slot in {"first experiment", "later screening experiment"} and stages:
            stage = stages[0] if slot == "first experiment" else stages[-1]
            records[slot] = {
                "slot": slot,
                "status": "grounded",
                "stage": stage["stage"],
                "evidence": stage["evidence"],
                "query_terms": sorted(question_terms),
                "query_acronyms": question_acronyms,
            }
            continue
        patterns = _SLOT_EVIDENCE_PATTERNS.get(slot, (re.escape(slot),))
        ranked = []
        for window in windows:
            lower = window["text"].casefold()
            pattern_hits = sum(bool(re.search(pattern, lower, re.I | re.DOTALL)) for pattern in patterns)
            if not pattern_hits:
                continue
            score = pattern_hits * 8 + sum(term in lower for term in question_terms)
            if figure_number and re.search(
                rf"\b(?:fig\.?|figure)\s*{re.escape(str(figure_number))}(?:[a-z])?\b",
                window["text"], re.I,
            ):
                score += 5
            if re.search(r"\bmethods?\b", lower):
                score += 1
            ranked.append((score, window))
        ranked.sort(key=lambda value: (value[0], -value[1]["page"]), reverse=True)
        evidence = []
        ranked_direct = []
        for row in direct:
            pattern_hits = sum(
                bool(re.search(pattern, row["text"], re.I | re.DOTALL))
                for pattern in patterns
            )
            if not pattern_hits:
                continue
            outcome_detail = len(re.findall(
                r"\b(?:significant|positive|negative|less than|identified|predicted|"
                r"only|both|different)\b", row["text"], re.I,
            ))
            ranked_direct.append((pattern_hits * 8 + outcome_detail, row))
        ranked_direct.sort(key=lambda value: (value[0], value[1]["page"]), reverse=True)
        # Figure-reference windows overlap heavily. Limit any one page so a
        # run of near-identical matches cannot crowd out a later Results page
        # that contains the requested quantitative outcome.
        direct_page_counts: dict[int, int] = {}
        selected_direct = []
        for _, row in ranked_direct:
            page = int(row["page"])
            if direct_page_counts.get(page, 0) >= 2:
                continue
            selected_direct.append(row)
            direct_page_counts[page] = direct_page_counts.get(page, 0) + 1
            if len(selected_direct) >= 6:
                break
        for row in selected_direct:
            evidence.append({
                "page": row["page"], "source": "exact_figure_reference",
                "text": row["text"],
            })
        # Construction questions need at least one mechanics-bearing passage,
        # even when caption and figure-reference windows already occupy the
        # normal evidence budget.
        preferred_ranked = []
        if slot == "score construction":
            mechanics = [
                row for _, row in ranked
                if re.search(r"\b(?:nuclear morphology|nuclear features?)\b", row["text"], re.I)
                and re.search(r"\b(?:CSS|cell senescence score|score assigned)\b", row["text"], re.I)
            ][:1]
            validation = [
                row for _, row in ranked
                if re.search(r"\b(?:higher|lower|increased|decreased)\b", row["text"], re.I)
                and re.search(r"\b(?:CSS|TSS|score)\b", row["text"], re.I)
                and re.search(r"\b(?:versus|vs\.?|compared|than)\b", row["text"], re.I)
            ][:1]
            preferred_ranked = [*mechanics, *validation]
        for row in preferred_ranked:
            if not any(
                item["page"] == row["page"] and item["text"] == row["text"]
                for item in evidence
            ):
                evidence.append({
                    "page": row["page"], "source": "slot_specific_document_text",
                    "text": row["text"],
                })
        for _, row in ranked:
            if any(
                item["page"] == row["page"] and item["text"] == row["text"]
                for item in evidence
            ):
                continue
            evidence.append({
                "page": row["page"], "source": "document_text", "text": row["text"],
            })
            if len(evidence) >= 6:
                break
        records[slot] = {
            "slot": slot,
            "status": "grounded" if evidence else "not_found_after_local_search",
            "evidence": evidence,
            "query_terms": sorted(question_terms),
            "query_acronyms": question_acronyms,
        }
    # Quantitative screen slots belong to the later screening stage, never to
    # a preceding validation/senolytic stage. Attach their locally retrieved
    # Results evidence to that stage so chronological synthesis has complete,
    # non-interchangeable evidence for both parts of the question.
    later = records.get("later screening experiment")
    if later:
        signatures = {
            (item.get("page"), item.get("text", ""))
            for item in later.get("evidence", [])
        }
        for related_slot in (
            "compounds screened", "condition-specific hit counts", "shared hit count",
        ):
            for item in records.get(related_slot, {}).get("evidence", []):
                signature = (item.get("page"), item.get("text", ""))
                if signature not in signatures:
                    later.setdefault("evidence", []).append(item)
                    signatures.add(signature)
    return records


def apply_grounded_slot_fallback(answer: str, slot_evidence: dict[str, dict]) -> tuple[str, list[str]]:
    """Replace premature not-found warnings with already retrieved local evidence."""
    value = str(answer or "")
    appended = []
    additions = []
    for slot, record in slot_evidence.items():
        if record.get("status") != "grounded" or not record.get("evidence"):
            continue
        evidence_text = re.sub(r"\s+", " ", record["evidence"][0]["text"]).strip()
        warning = re.compile(
            rf"(?im)^\s*[-*]?\s*{re.escape(slot)}\s*:\s*"
            r"(?:not specified|not found|insufficient evidence)[^.\n]*(?:\.|$)"
        )
        value, replaced = warning.subn(f"- {slot}: {evidence_text}", value)
        if replaced:
            appended.append(slot)
        elif not re.search(re.escape(slot), value, re.I):
            additions.append(f"- **{slot}:** {evidence_text}")
            appended.append(slot)
    if additions:
        value = f"{value.rstrip()}\n\n**Grounded requested details**\n\n" + "\n".join(additions)
    return value.strip(), appended


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
                rf"\b(?:Fig\.?|Figure)\s*{re.escape(str(figure_number))}(?:[a-z])?\b",
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
                rf"\b(?:fig\.?|figure)\s*{re.escape(str(figure_number))}(?:[a-z])?\b", passage, re.I,
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
    slot_evidence = resolve_slot_evidence(
        slots, question, figure_number, panel_map, all_pages, direct,
    )
    existing_support = {
        (row.get("page"), re.sub(r"\s+", " ", row.get("text", "")).casefold())
        for row in supporting
    }
    for record in slot_evidence.values():
        for item in record.get("evidence", []):
            if item.get("page") is None:
                continue
            key = (item.get("page"), re.sub(r"\s+", " ", item.get("text", "")).casefold())
            if key in existing_support:
                continue
            supporting.append({
                "page": item["page"], "score": 100, "text": item["text"],
                "slot": record["slot"],
            })
            existing_support.add(key)
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
    condition_source_text = full_caption
    condition_record = slot_evidence.get("condition-specific outcomes", {})
    if condition_record.get("evidence"):
        condition_source_text += "\n" + "\n".join(
            item.get("text", "") for item in condition_record["evidence"]
        )
    elif direct:
        condition_source_text += "\n" + "\n".join(row["text"] for row in direct)
    condition_outcomes = explicit_condition_outcomes(condition_source_text)
    condition_tuples = extract_condition_tuples(condition_source_text)
    taxonomy = extract_explicit_classifier_taxonomy(evidence_text)
    return {
        "full_caption": full_caption,
        "panel_map": panel_map,
        "local_pages": sorted(local_pages),
        "direct_references": direct,
        "supporting_passages": supporting,
        "requested_answer_slots": slots,
        "slot_evidence": slot_evidence,
        "explicit_condition_outcomes": condition_outcomes,
        "condition_tuples": condition_tuples,
        "explicit_classifier_taxonomy": taxonomy,
        "evidence_text": evidence_text,
        "glossary": document_glossary(whole_text),
    }


def merge_mixed_figure_with_caption(result: dict, panel_map: list[dict]) -> tuple[dict, bool]:
    """Make explicit caption identity authoritative and fill missing panels."""
    merged = dict(result)
    panels = [dict(row) for row in result.get("panels", []) if isinstance(row, dict)]
    caption_by_panel = authoritative_panel_role_map(panel_map)
    present = set()
    for row in panels:
        panel = str(row.get("panel", "")).casefold()
        if not panel:
            continue
        present.add(panel)
        caption = caption_by_panel.get(panel)
        if not caption:
            continue
        payload = dict(row.get("structured_analysis") or {})
        old_summary = str(payload.get("summary", "")).strip()
        if old_summary and old_summary != caption["caption_description"]:
            payload["vision_summary"] = old_summary
        payload["summary"] = caption["caption_description"]
        payload["role"] = caption["role"]
        evidence = list(payload.get("evidence") or [])
        if "caption" not in evidence:
            evidence.append("caption")
        payload["evidence"] = evidence
        row["visual_type"] = caption["visual_type"]
        row["structured_analysis"] = payload
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
