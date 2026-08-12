"""Authoritative evidence objects and final-answer consistency checks.

This module sits after retrieval/vision validation.  It deliberately does not
rank evidence or reinterpret images: it converts already-resolved evidence
into one compact object, validates generated prose against that object, and
provides a deterministic last-resort rendering.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict


FINAL_EVIDENCE_MARKER = "[FINAL ANSWER EVIDENCE]"
AUTHORITY_ORDER = [
    "explicit_author_results_methods",
    "explicit_full_caption",
    "validated_structured_visual",
    "model_interpretation",
]

_PLACEHOLDER_RE = re.compile(
    r"\b(?:article\s+https?://\S+|nature communications\|?\s*\(\d{4}\)\s*\S+|"
    r"source data are provided|---\s*page\s*\d+\s*---)\b",
    re.I,
)


def _normal(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").casefold()).strip()


def _clean_text(value: str) -> str:
    text = str(value or "").replace("ﬁ", "fi").replace("ﬂ", "fl")
    text = re.sub(
        r"Article\s+https?://\S+\s+Nature Communications\|?\s*"
        r"\(\d{4}\)\s*\d+:\d+\s*\d*",
        " ", text, flags=re.I,
    )
    text = re.sub(r"\b(Fig|fig)\.", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip(" -;:")
    text = re.sub(
        r"(?<=[a-z)])\s+(?=(?:To this end|However|Finally|In contrast|As expected)\b)",
        ". ", text,
    )
    return text


def _clean_caption_description(value: str) -> str:
    """Remove publication boilerplate while retaining the caption claim."""
    text = _clean_text(value)
    text = re.split(
        r"\b(?:Source Data are provided|Data are presented as|Article\s+https?://)",
        text, maxsplit=1, flags=re.I,
    )[0]
    return text.strip(" -;:")


def _sentences(values) -> list[str]:
    if isinstance(values, str):
        values = [values]
    rows = []
    for value in values or []:
        cleaned = _clean_text(value)
        cleaned = re.sub(r"(?<=[.)])(?=[A-Z])", " ", cleaned)
        for sentence in re.split(r"(?<=[.!?])\s+", cleaned):
            sentence = sentence.strip()
            if len(sentence.split()) < 5 or _PLACEHOLDER_RE.search(sentence):
                continue
            if sentence not in rows:
                rows.append(sentence)
    return rows


def _all_local_text(local_evidence: dict) -> list[str]:
    rows = [local_evidence.get("full_caption", "")]
    rows.extend(local_evidence.get("explicit_condition_outcomes", []))
    rows.extend(row.get("text", "") for row in local_evidence.get("direct_references", []))
    for record in local_evidence.get("slot_evidence", {}).values():
        rows.extend(item.get("text", "") for item in record.get("evidence", []))
    for record in local_evidence.get("condition_tuples", []):
        rows.extend(record.get("author_statements", []))
    return rows


def _extract_feature_set(texts: list[str]) -> dict | None:
    combined = " ".join(_clean_text(value) for value in texts)
    candidates = []
    for match in re.finditer(
        r"\bnuclear features?\s*,?\s*(?:including|were|:)?\s*"
        r"([^.;]{15,260})", combined, re.I,
    ):
        fragment = re.split(
            r"\b(?:all of these|were significantly|could distinguish|fig(?:ure)?\s*\d)",
            match.group(1), maxsplit=1, flags=re.I,
        )[0]
        items = [
            re.sub(r"^(?:and|the)\s+", "", item.strip(" ,;:"), flags=re.I)
            for item in re.split(r",|\band\b", fragment, flags=re.I)
        ]
        items = [
            item for item in items
            if 1 <= len(item.split()) <= 4
            and not re.search(r"\b(?:cells?|treated|percentage|figure|software)\b", item, re.I)
        ]
        if len(items) >= 3:
            candidates.append(items)
    if not candidates:
        return None
    items = max(candidates, key=len)
    unique = []
    for item in items:
        if _normal(item) not in {_normal(existing) for existing in unique}:
            unique.append(item)
    exception = None
    comparison = None
    exception_matches = list(re.finditer(
        r"all of these nuclear features?\s*,?\s*except\s+([^,.;]+)\s*,?\s*"
        r"were significantly different between\s+(.{5,180}?)(?=\s*\(Fig|[.;])",
        combined, re.I,
    ))
    if exception_matches:
        exception_match = min(exception_matches, key=lambda item: len(item.group(2)))
        exception = exception_match.group(1).strip()
        comparison = exception_match.group(2).strip()
    return {
        "items": unique,
        "exception": exception,
        "comparison": comparison,
        "authority": "explicit_author_results_methods",
    }


def _condition_design_text(panel_map: list[dict]) -> str:
    rows = []
    include_next = False
    for row in panel_map:
        if row.get("role") == "experimental_design":
            include_next = True
            rows.append(row.get("caption_description", ""))
            continue
        if include_next:
            rows.append(row.get("caption_description", ""))
            include_next = False
    return _clean_text(" ".join(rows))


def _repair_condition_name(name: str, all_text: str) -> str:
    value = _clean_text(name).strip(" ,;:()")
    # PDF references can be fused to an identifier (for example a citation
    # number immediately after a drug name). Prefer the shorter token when it
    # is also explicitly present elsewhere in the resolved evidence.
    for cut in (4, 2, 1):
        if len(value) <= cut or not value[-cut:].isdigit():
            continue
        shorter = value[:-cut]
        if len(shorter) >= 3 and re.search(rf"\b{re.escape(shorter)}\b", all_text, re.I):
            value = shorter
            break
    return value


def _condition_id(name: str, aliases: list[str], statements: list[str]) -> str:
    joined = " ".join([name, *aliases]).casefold()
    if "irradiat" in joined or re.search(r"\bdd\b", joined):
        return "irradiated_dd"
    if "quiescent" in joined:
        return "quiescent"
    if "growing" in joined:
        return "growing"
    base = re.sub(r"[^a-z0-9]+", "_", name.casefold()).strip("_")
    base = re.sub(r"_(?:treated|induced)$", "", base)
    statement_text = " ".join(statements)
    if base and re.search(
        rf"\b(?:senescen\w*\b.{{0,60}}{re.escape(name)}|"
        rf"{re.escape(name)}.{{0,60}}\bsenescen\w*)",
        statement_text, re.I,
    ):
        return f"{base}_senescence"
    return base


def _measurement_names(statement: str) -> list[str]:
    names = []
    patterns = (
        r"\bBrdU\b", r"\bSA\s*-?\s*[βB]-?Gal(?:actosidase)?\b",
        r"\b\d+[A-Z][A-Za-z0-9-]*\b", r"\b[A-Z]{2,}[A-Za-z0-9-]*\b",
        r"\bDNA damage\b", r"\bsenescence\b",
    )
    for index, pattern in enumerate(patterns):
        flags = re.I if index in {0, 1, 4, 5} else 0
        for match in re.finditer(pattern, statement, flags):
            value = re.sub(r"\s+", " ", match.group(0)).strip()
            if value.casefold() in {"fig", "a549", "fbs", "dmso", "gfp"} or re.search(
                r"treated$", value, re.I,
            ):
                continue
            if _normal(value) not in {_normal(item) for item in names}:
                names.append(value)
    return names


def _qualifiers(statement: str) -> list[str]:
    lower = statement.casefold()
    rows = []
    patterns = (
        (r"\b(?:no|not|without)\b.{0,50}\bsignificant\b", "no_significant_change"),
        (r"\bsignificant(?:ly)?\s+(?:increase|higher|positive)", "significant_increase"),
        (r"\bless than\s+\d+(?:\.\d+)?\s*%", "bounded_value"),
        (r"\bidentified most\b|\bmost\b.{0,50}\bidentified\b", "most_identified"),
        (r"\bselectively (?:reduced|decreased|killed)\b", "selective_decrease"),
        (r"\bnot\b.{0,40}\bsenescen", "not_established_senescent"),
        (r"\bwere senescent\b|\bas senescent\b", "senescent"),
        (r"\bwere dividing\b|\bincorporat(?:e|ed|ing)\s+BrdU\b", "dividing"),
        (r"\bwere arrested\b|\bcell cycle arrest\b", "arrested"),
    )
    for pattern, label in patterns:
        if re.search(pattern, lower):
            rows.append(label)
    return rows


def _condition_qualifiers(statement: str, aliases: list[str]) -> list[str]:
    """Resolve qualifiers relative to one condition in a multi-condition sentence."""
    qualifiers = _qualifiers(statement)

    def mentions(text: str) -> bool:
        return any(
            re.search(
                rf"\b{re.escape(re.sub(r'[-_ ]treated$', '', alias, flags=re.I))}"
                r"(?:-?treated)?\b",
                text, re.I,
            )
            for alias in aliases
        )

    only_parts = re.split(r"\bonly observed in\b", statement, maxsplit=1, flags=re.I)
    if len(only_parts) == 2 and "significant_increase" in qualifiers:
        if not mentions(only_parts[1]):
            qualifiers.remove("significant_increase")
            qualifiers.append("no_significant_change")
    bounded_subject = re.search(
        r"\bless than\s+\d+(?:\.\d+)?\s*%[\s)]*of\s+(?:the\s+)?"
        r"([A-Za-z0-9-]+)", statement, re.I,
    )
    if bounded_subject and not mentions(bounded_subject.group(1)):
        qualifiers = [
            item for item in qualifiers
            if item not in {"bounded_value", "most_identified", "senescent"}
        ]
    elif bounded_subject and "bounded_value" in qualifiers:
        qualifiers = [
            item for item in qualifiers
            if item not in {"most_identified", "senescent"}
        ]
    not_parts = re.split(r"\bbut not\b", statement, maxsplit=1, flags=re.I)
    if len(not_parts) == 2 and re.search(r"\bsenescen", statement, re.I):
        if mentions(not_parts[1]):
            qualifiers = [item for item in qualifiers if item != "senescent"]
            if "not_established_senescent" not in qualifiers:
                qualifiers.append("not_established_senescent")
        elif mentions(not_parts[0]):
            qualifiers = [item for item in qualifiers if item != "not_established_senescent"]
            if "senescent" not in qualifiers:
                qualifiers.append("senescent")
    whereas_parts = re.split(r"\bwhereas\b", statement, maxsplit=1, flags=re.I)
    if len(whereas_parts) == 2:
        if mentions(whereas_parts[0]):
            qualifiers = [item for item in qualifiers if item != "arrested"]
        elif mentions(whereas_parts[1]):
            qualifiers = [item for item in qualifiers if item != "dividing"]
    return list(dict.fromkeys(qualifiers))


def _extract_condition_records(local_evidence: dict, texts: list[str]) -> list[dict]:
    all_text = _clean_text(" ".join(texts))
    design = _condition_design_text(local_evidence.get("panel_map", []))
    source_sentences = _sentences(texts)
    for index, sentence in enumerate(source_sentences):
        if not re.match(r"^(?:It|This) also\b", sentence, re.I):
            continue
        for previous in reversed(source_sentences[max(0, index - 5):index]):
            classifier = re.search(r"\b([A-Z][A-Z0-9-]{1,15})\s+classifier\b", previous)
            if classifier and re.search(r"\b(?:identified|predicted)\b", previous, re.I):
                source_sentences[index] = re.sub(
                    r"^(?:It|This)", f"The {classifier.group(1)} classifier",
                    sentence, count=1, flags=re.I,
                )
                break
    rows = []
    seen_ids = set()
    for source in local_evidence.get("condition_tuples", []):
        raw_name = str(source.get("condition", ""))
        name = _repair_condition_name(raw_name, all_text)
        aliases = [_repair_condition_name(str(item), all_text) for item in source.get("aliases", [])]
        aliases = list(dict.fromkeys([name, *aliases]))
        if not name or name.casefold() in {
            "fig", "figure", "fbs", "gfp", "mcherry", "supplementary", "dmso",
        }:
            continue
        if design and not any(re.search(rf"\b{re.escape(alias)}\b", design, re.I) for alias in aliases):
            continue
        def alias_pattern(alias: str) -> str:
            base = re.sub(r"[-_ ]treated$", "", alias, flags=re.I)
            if base != alias:
                return rf"\b{re.escape(base)}(?:-?treated)?\b"
            return rf"\b{re.escape(alias)}(?:-?treated)?\b"

        statements = [
            sentence for sentence in source_sentences
            if any(re.search(alias_pattern(alias), sentence, re.I) for alias in aliases)
            and re.search(
                r"\b(?:significant|positive|negative|senescen\w*|arrest|damage|"
                r"identified|predicted|less than|increase|decrease|BrdU|53BP1|AEM)\b",
                sentence, re.I,
            )
        ]
        statements.sort(
            key=lambda sentence: (
                sum(bool(re.search(pattern, sentence, re.I)) for pattern in (
                    r"\bsignificant increase\b", r"\bno significant\b",
                    r"\bstaining confirmed\b", r"\bidentified most\b",
                    r"\bless than\s+\d", r"\bnot just detecting\b",
                    r"\bonly observed\b",
                )),
                -len(sentence),
            ),
            reverse=True,
        )
        identifier = _condition_id(name, aliases, statements)
        if not identifier or identifier in seen_ids:
            continue
        seen_ids.add(identifier)
        if identifier == "growing":
            display_label = "growing"
        elif identifier == "quiescent":
            display_label = "quiescent"
        elif identifier == "irradiated_dd":
            display_label = "irradiated (DD)"
        elif identifier.endswith("_senescence"):
            display_label = f"{re.sub(r'[-_ ]treated$', '', name, flags=re.I)} senescence"
        else:
            display_label = name
        measurements = []
        for statement in statements[:10]:
            names = _measurement_names(statement)
            if names:
                measurements.append({
                    "names": names,
                    "qualifiers": _condition_qualifiers(statement, aliases),
                    "statement": statement,
                    "authority": "explicit_author_results_methods",
                })
        rows.append({
            "id": identifier,
            "condition": name,
            "label": display_label,
            "aliases": aliases,
            "measurements": measurements,
            "author_statements": statements[:10],
        })
    return rows


def _select_stage_facts(evidence: list[dict]) -> list[str]:
    statements = _sentences(item.get("text", "") for item in evidence)
    ranked = []
    for sentence in statements:
        score = sum(bool(re.search(pattern, sentence, re.I)) for pattern in (
            r"\b(?:experimental design|co-?culture|screen(?:ed|ing))\b",
            r"\b(?:control|treated|classifier|predicted)\b",
            r"\b(?:selectively|reduced|increased|identified|only|both|toxic)\b",
            r"\b\d+(?:\.\d+)?\s*%?\b",
        ))
        if score >= 2:
            ranked.append((score, sentence))
    ranked.sort(key=lambda item: (item[0], len(item[1])), reverse=True)
    return [sentence for _, sentence in ranked[:8]]


def _stage_required_terms(facts: list[str]) -> list[str]:
    terms = []
    for sentence in facts:
        if re.search(
            r"\b(?:examples?|exemplified|e\.g\.|such as|candidates?|including)\b",
            sentence, re.I,
        ):
            continue
        for match in re.finditer(
            r"\b(?:[A-Za-z]+-\d+|[A-Z][A-Za-z]*\d+[A-Za-z0-9-]*|"
            r"[A-Z]{2,}[A-Za-z0-9]*|m[A-Z][A-Za-z]+)\b",
            sentence,
        ):
            term = match.group(0)
            if term.casefold() in {
                "fig", "figure", "supplementary", "fbs", "parp1", "b-score",
                "ii", "iii", "iv",
            }:
                continue
            if term not in terms:
                terms.append(term)
        for phrase in (
            "selectively reduced", "non-senescent", "toxic",
        ):
            if phrase in sentence.casefold() and phrase not in terms:
                terms.append(phrase)
    return terms[:12]


def _extract_numeric_facts(statements: list[str]) -> list[dict]:
    rows = []
    for statement in statements:
        protected = re.sub(r"\b([ei])\.g\.", r"\1g", statement, flags=re.I)
        clauses = re.split(r"[.;]", protected)
        for clause in clauses:
            if not re.search(
                r"\b(?:screen(?:ed|ing)?|identified|hits?|only|both|shared|toxic|less than|"
                r"more than|threshold)\b", clause, re.I,
            ):
                continue
            matches = []
            for pattern in (
                r"\b(?:screened|screening)\s+(?:a\s+collection\s+of\s+)?(\d+)\b",
                r"\bidentified\s+(\d+)\s+(?:drugs?|compounds?|hits?)\b",
                r"\b(\d+)\s+(?:drugs?|compounds?|hits?)\b",
                r"(?<![-A-Za-z0-9])(\d+)(?![-A-Za-z0-9]).{0,140}?"
                r"\b(?:only\s+in|specific\s+to)\b",
                r"\b(?:less|more)\s+than\s+(\d+(?:\.\d+)?\s*%)",
            ):
                matches.extend(re.finditer(pattern, clause, re.I))
            seen_clause_values = set()
            for match in matches:
                value = re.sub(r"\s+", "", match.group(1))
                if value in seen_clause_values:
                    continue
                seen_clause_values.add(value)
                field = re.sub(r"\s+", " ", clause).strip(" ,:-")
                matched_text = match.group(0)
                tail = clause[match.end():]
                tail = re.split(r"(?<![-A-Za-z0-9])\d+(?![-A-Za-z0-9])", tail, maxsplit=1)[0]
                if re.search(r"\bscreen", matched_text, re.I):
                    key = "screened"
                elif re.search(r"\bidentified\b", matched_text, re.I) or re.search(
                    r"\bidentified\s*$", clause[:match.start()], re.I,
                ):
                    key = "identified_total"
                elif target := re.search(
                    r"\bonly in\s+([A-Za-z0-9-]+)", f"{matched_text} {tail}", re.I,
                ):
                    key = f"only_{_normal(target.group(1))}"
                elif re.search(r"\bboth\b", f"{matched_text} {tail}", re.I):
                    key = "both"
                elif re.search(r"\btoxic\b", field, re.I):
                    key = "toxicity_filter"
                elif re.search(r"\bless than\b", field, re.I):
                    key = "bounded_value"
                else:
                    key = _normal(field)
                signature = (key, value)
                if signature not in {(row["key"], row["value"]) for row in rows}:
                    rows.append({
                        "key": key,
                        "field": field,
                        "value": value,
                        "authority": "explicit_author_results_methods",
                    })
    return rows


def _extract_stage_records(local_evidence: dict) -> tuple[list[dict], list[dict]]:
    stages = []
    all_facts = []
    all_stage_sentences = []
    design_labels = [
        row.get("caption_description", "")
        for row in local_evidence.get("panel_map", [])
        if row.get("role") == "experimental_design"
    ]
    panel_rows = local_evidence.get("panel_map", [])
    design_starts = [
        index for index, row in enumerate(panel_rows)
        if row.get("role") == "experimental_design"
    ]
    stage_panel_sets = []
    for stage_index, start in enumerate(design_starts):
        end = design_starts[stage_index + 1] if stage_index + 1 < len(design_starts) else len(panel_rows)
        stage_panel_sets.append({
            str(row.get("panel", "")).casefold()
            for row in panel_rows[start:end] if row.get("panel")
        })
    target_match = re.search(
        r"\bFig(?:ure)?\.?\s*(\d+(?:\.\d+)?)(?:[a-z])?\b",
        str(local_evidence.get("full_caption", "")), re.I,
    )
    target_number = target_match.group(1) if target_match else None

    def belongs_to_stage(text: str, stage_number: int) -> bool:
        refs = re.findall(
            r"(?<!Supplementary\s)\bFig(?:ure)?\s*(\d+(?:\.\d+)?)([a-z])?\b",
            text, re.I,
        )
        if target_number and any(number != target_number for number, _ in refs):
            return False
        panels = stage_panel_sets[stage_number - 1] if stage_number <= len(stage_panel_sets) else set()
        target_panels = [panel.casefold() for number, panel in refs if number == target_number and panel]
        return not target_panels or not panels or any(panel in panels for panel in target_panels)

    for slot, record in local_evidence.get("slot_evidence", {}).items():
        stage_number = record.get("stage")
        if not stage_number:
            continue
        stage_sentences = _sentences(
            item.get("text", "") for item in record.get("evidence", [])
        )
        stage_sentences = [
            sentence for sentence in stage_sentences
            if belongs_to_stage(sentence, int(stage_number))
        ]
        anchor_tokens = {
            token for token in re.findall(
                r"[a-z0-9][a-z0-9-]{2,}", " ".join(stage_sentences).casefold(),
            )
            if token not in {
                "the", "and", "cells", "cell", "with", "from", "after", "before",
                "figure", "panel", "percentage", "different", "treatment",
            }
        }
        for outcome in _sentences(local_evidence.get("explicit_condition_outcomes", [])):
            if not belongs_to_stage(outcome, int(stage_number)):
                continue
            outcome_refs = re.findall(
                r"(?<!Supplementary\s)\bFig(?:ure)?\s*(\d+(?:\.\d+)?)([a-z])?\b",
                outcome, re.I,
            )
            outcome_tokens = set(re.findall(r"[a-z0-9][a-z0-9-]{2,}", outcome.casefold()))
            label_tokens = set(re.findall(
                r"[a-z0-9][a-z0-9-]{3,}",
                design_labels[int(stage_number) - 1].casefold()
                if int(stage_number) <= len(design_labels) else slot.casefold(),
            ))
            enough_overlap = (
                len(anchor_tokens & outcome_tokens) >= 2
                and (outcome_refs or len(label_tokens & outcome_tokens) >= 2)
            )
            if enough_overlap and re.search(
                r"\b(?:selectiv|reduc|increase|identified|screen|toxic|only|both)\w*\b",
                outcome, re.I,
            ) and outcome not in stage_sentences:
                stage_sentences.append(outcome)
        all_stage_sentences.extend(stage_sentences)
        facts = _select_stage_facts(record.get("evidence", []))
        facts = [
            fact for fact in facts
            if belongs_to_stage(fact, int(stage_number))
        ]
        for outcome in stage_sentences:
            if re.search(
                r"\b(?:selectiv|reduc|increase|identified|screen|toxic|only|both)\w*\b",
                outcome, re.I,
            ) and outcome not in facts:
                facts.append(outcome)
            if len(facts) >= 12:
                break
        numeric = _extract_numeric_facts(stage_sentences)
        for numeric_fact in numeric:
            supporting = next(
                (
                    sentence for sentence in stage_sentences
                    if numeric_fact["value"] in re.sub(r"\s+", "", sentence)
                ),
                None,
            )
            if supporting and supporting not in facts:
                facts.append(supporting)
        all_facts.extend(facts)
        stages.append({
            "stage": int(stage_number),
            "slot": slot,
            "label": (
                design_labels[int(stage_number) - 1]
                if int(stage_number) <= len(design_labels)
                else slot
            ),
            "facts": facts,
            "required_terms": _stage_required_terms(facts),
            "authority": "explicit_author_results_methods",
        })
    stages.sort(key=lambda row: row["stage"])
    return stages, _extract_numeric_facts([*all_stage_sentences, *all_facts])


def _extract_author_conclusions(
    texts: list[str], figure_number: str | None = None,
) -> list[dict]:
    rows = []
    for sentence in _sentences(texts):
        if not re.search(r"\bnot (?:just|merely) detect", sentence, re.I):
            continue
        refs = re.findall(
            r"(?<!Supplementary\s)\bFig(?:ure)?\s*(\d+(?:\.\d+)?)(?:[a-z])?\b",
            sentence, re.I,
        )
        if figure_number and refs and str(figure_number) not in refs:
            continue
        classifier = re.search(r"\b([A-Z][A-Z0-9-]{1,15})\s+classifier\b", sentence)
        object_match = re.search(r"\bnot (?:just|merely) detecting\s+([^,.;]+)", sentence, re.I)
        if classifier and object_match:
            rows.append({
                "subject": classifier.group(1),
                "relation": "not_merely_detector_of",
                "object": object_match.group(1).strip(),
                "authority": "explicit_author_results_methods",
            })
    return rows


def build_figure_final_answer_evidence(
    *, question: str, document: str, figure_number: str | None, page: int,
    local_evidence: dict, validated_visual=None,
) -> dict:
    """Build one authoritative object from already-resolved figure evidence."""
    texts = _all_local_text(local_evidence)
    stages, numeric_facts = _extract_stage_records(local_evidence)
    slot_numeric_facts = _extract_numeric_facts(
        _sentences(
            item.get("text", "")
            for record in local_evidence.get("slot_evidence", {}).values()
            for item in record.get("evidence", [])
        )
    )
    existing_numeric = {(row.get("key"), row.get("value")) for row in numeric_facts}
    for fact in slot_numeric_facts:
        if (fact.get("key"), fact.get("value")) not in existing_numeric:
            numeric_facts.append(fact)
            existing_numeric.add((fact.get("key"), fact.get("value")))
    design_labels = [
        row.get("caption_description", "")
        for row in local_evidence.get("panel_map", [])
        if row.get("role") == "experimental_design"
    ]
    screen_records = [
        record for slot, record in local_evidence.get("slot_evidence", {}).items()
        if re.search(r"\b(?:screen\w*|hit count|shared hit)\b", slot, re.I)
    ]
    if len(design_labels) >= 2 and screen_records and not any(
        row.get("stage") == 2 for row in stages
    ):
        screen_evidence = [
            item for record in screen_records for item in record.get("evidence", [])
        ]
        screen_facts = _select_stage_facts(screen_evidence)
        stages.append({
            "stage": 2,
            "slot": "screening experiment",
            "label": design_labels[1],
            "facts": screen_facts,
            "required_terms": _stage_required_terms(screen_facts),
            "authority": "explicit_author_results_methods",
        })
        stages.sort(key=lambda row: row["stage"])
    requested_slots = set(local_evidence.get("requested_answer_slots", []))
    feature_set = (
        _extract_feature_set(texts)
        if requested_slots.intersection({"features", "changing nuclear features", "feature exceptions"})
        else None
    )
    needs_conditions = bool(re.search(
        r"\b(?:conditions?|compare|distinguish|quiescen|irradiat|dna damage)\b",
        question, re.I,
    )) or "condition-specific outcomes" in requested_slots
    conditions = _extract_condition_records(local_evidence, texts) if needs_conditions else []
    quantitative_condition_facts = _extract_numeric_facts(
        statement
        for row in conditions for statement in row.get("author_statements", [])
    )
    for fact in quantitative_condition_facts:
        if (fact["field"], fact["value"]) not in {
            (row["field"], row["value"]) for row in numeric_facts
        }:
            numeric_facts.append(fact)
    return {
        "schema_version": 1,
        "answer_kind": "resolved_figure",
        "question": question,
        "authority_order": AUTHORITY_ORDER,
        "source": {
            "document": document,
            "figure_number": str(figure_number) if figure_number not in (None, "") else None,
            "page": int(page),
        },
        "panels": [
            {
                "panel": row.get("panel"),
                "role": row.get("role"),
                "description": _clean_caption_description(row.get("caption_description", "")),
                "authority": "explicit_full_caption",
            }
            for row in local_evidence.get("panel_map", [])
            if row.get("panel") and row.get("caption_description")
        ],
        "feature_set": feature_set,
        "condition_records": conditions,
        "experiment_stages": stages,
        "numeric_facts": numeric_facts,
        "author_conclusions": _extract_author_conclusions(texts, figure_number),
        "explicit_facts": _sentences(local_evidence.get("explicit_condition_outcomes", [])),
        "validated_visual": validated_visual,
    }


def build_provenance_final_answer_evidence(question: str, rows: list[dict]) -> dict:
    """Build the whole-document composer input from selected provenance rows."""
    return {
        "schema_version": 1,
        "answer_kind": "experimental_domain_synthesis",
        "question": question,
        "authority_order": AUTHORITY_ORDER,
        "provenance": [
            {
                "document": row.get("document"),
                "figure_number": row.get("figure_number"),
                "panel": row.get("panel"),
                "page": row.get("page"),
                "experimental_domain": row.get("experimental_domain"),
                "species": row.get("species"),
                "sample_type": row.get("sample_type"),
                "source_text": row.get("source_text"),
                "source_provenance": row.get("source_provenance", []),
            }
            for row in rows if isinstance(row, dict) and row.get("experimental_domain")
        ],
    }


def final_evidence_context(value: dict) -> str:
    return f"{FINAL_EVIDENCE_MARKER}\n{json.dumps(value, ensure_ascii=False)}"


def parse_final_evidence_context(context: str) -> dict | None:
    _, marker, remainder = str(context or "").partition(FINAL_EVIDENCE_MARKER)
    if not marker or (start := remainder.find("{")) < 0:
        return None
    try:
        value, _ = json.JSONDecoder().raw_decode(remainder[start:])
    except (json.JSONDecodeError, TypeError):
        return None
    return value if isinstance(value, dict) else None


def _panel_keywords(panel: dict) -> list[str]:
    description = str(panel.get("description", ""))
    tokens = re.findall(
        r"\b(?:SA-?[βB]-?Gal|BrdU|p21(?:Cip1)?|p53|DAPI|nuclear features?|"
        r"training|test|validation|microscopy|images?|workflow|distributions?)\b",
        description, re.I,
    )
    return list(dict.fromkeys(tokens))[:4]


def _field_terms(field: str) -> list[str]:
    return [
        token for token in re.findall(r"[a-z][a-z0-9-]{2,}", field.casefold())
        if token not in {
            "the", "and", "with", "from", "were", "that", "this", "cells",
            "cell", "figure", "after", "before", "using", "into", "their",
        }
    ]


def validate_answer_consistency(answer: str, evidence: dict) -> list[str]:
    """Return factual/display inconsistencies against the authoritative object."""
    value = str(answer or "").strip()
    lower = value.casefold()
    errors = []
    if not value:
        return ["empty answer"]
    if value.count("**") % 2:
        errors.append("unmatched markdown marker")
    if re.search(r"\b(?:finally|then|and)\s*,?\s*\*\*\s*\d", value, re.I):
        errors.append("malformed markdown before numeric clause")
    if re.search(r"\[(?:TARGET|VALIDATED|RESOLVED|EXPLICIT)[^\]]*\]", value, re.I):
        errors.append("raw evidence marker leaked into answer")
    if re.search(
        r"\b(?:RETRIEVED RAG EVIDENCE|Source Data are provided|doi:\s*10\.|"
        r"---\s*Page\s*\d+\s*---|Article\s+https?://)\b",
        value, re.I,
    ):
        errors.append("raw source-chunk fragment leaked into answer")

    if evidence.get("answer_kind") == "resolved_figure":
        for panel in evidence.get("panels", []):
            label = str(panel.get("panel", ""))
            if not re.search(rf"\bpanel\s+{re.escape(label)}\b", value, re.I):
                errors.append(f"missing panel {label}")
                continue
            keywords = _panel_keywords(panel)
            panel_match = re.search(
                rf"\bpanel\s+{re.escape(label)}\b"
                r"(.*?)(?=\bpanel\s+[a-z0-9]+\b|\Z)", value,
                re.I | re.DOTALL,
            )
            if keywords and panel_match and not any(
                re.search(re.escape(keyword), panel_match.group(0), re.I)
                for keyword in keywords
            ):
                errors.append(f"panel {label} description conflicts with full caption")
        if evidence.get("panels") and re.search(
            r"\bpanels?\s+[a-z](?:\s*[-–—]\s*[a-z])?\b.{0,100}\b(?:likely|probably)\b|"
            r"\b(?:likely|probably)\b.{0,100}\bpanels?\s+[a-z]",
            value, re.I,
        ):
            errors.append("speculative panel assignment despite explicit caption")

        feature_set = evidence.get("feature_set") or {}
        for item in feature_set.get("items", []):
            if _normal(item) not in _normal(value):
                errors.append(f"missing grounded feature: {item}")
        exception = feature_set.get("exception")
        if exception and not re.search(
            rf"(?:except\s+{re.escape(exception)}|{re.escape(exception)}.{{0,90}}"
            r"(?:not|unchanged|did not|exception))",
            value, re.I | re.DOTALL,
        ):
            errors.append(f"missing grounded feature exception: {exception}")

        stages = _renderer_stage_plan(evidence)
        for stage in stages:
            number = stage.get("stage")
            stage_match = re.search(
                rf"\bstage\s+{number}\b(.*?)(?=\bstage\s+\d+\b|\Z)",
                value, re.I | re.DOTALL,
            )
            if not stage_match:
                errors.append(f"missing Stage {number}")
                continue
            stage_text = stage_match.group(0)
            for term in stage.get("required_terms", []):
                if _normal(term) not in _normal(stage_text):
                    errors.append(f"Stage {number} missing grounded term: {term}")

        author_outcomes = [
            statement for row in evidence.get("condition_records", [])
            for statement in row.get("author_statements", [])
        ]
        answer_sentences = [
            *re.split(r"(?<=[.!?])\s+", value),
            *[line for line in value.splitlines() if line.strip()],
        ]
        for record in evidence.get("condition_records", []):
            aliases = list(dict.fromkeys([
                str(record.get("label", "")),
                *[str(item) for item in record.get("aliases", [])],
            ]))
            condition_sentences = [
                sentence for sentence in answer_sentences
                if any(
                    _normal(alias) and _normal(alias) in _normal(sentence)
                    for alias in aliases
                )
            ]
            if not condition_sentences:
                errors.append(f"missing condition record: {record.get('id')}")
                continue
            seen_requirements = set()
            for measurement in record.get("measurements", []):
                names = [
                    name for name in measurement.get("names", [])
                    if re.search(
                        r"(?:BrdU|SA.*Gal|53BP1|AEM|DNA damage|senescence)",
                        name, re.I,
                    )
                ]
                for qualifier in measurement.get("qualifiers", []):
                    requirement = (qualifier, tuple(_normal(name) for name in names))
                    if not names or requirement in seen_requirements:
                        continue
                    seen_requirements.add(requirement)
                    relevant = (
                        condition_sentences
                        if qualifier == "bounded_value"
                        else [
                            sentence for sentence in condition_sentences
                            if any(_normal(name) in _normal(sentence) for name in names)
                        ]
                    )
                    qualifier_patterns = {
                        "significant_increase": r"\bsignificant",
                        "no_significant_change": r"\b(?:no|not|without|little)\b.{0,80}\b(?:significant|damage|increase)",
                        "bounded_value": r"\bless than\s+\d|<\s*\d",
                        "most_identified": r"\bmost\b",
                        "not_established_senescent": r"\bnot\b.{0,80}\bsenescen",
                        "senescent": r"\bsenescen",
                        "selective_decrease": r"\bselectiv\w*\b.{0,40}\b(?:decreas|reduc|kill)",
                    }
                    pattern = qualifier_patterns.get(qualifier)
                    if pattern and not any(re.search(pattern, sentence, re.I) for sentence in relevant):
                        errors.append(
                            f"missing {qualifier} evidence for condition {record.get('id')}"
                        )
        if author_outcomes:
            records = evidence.get("condition_records", [])
            for sentence in re.split(r"(?<=[.!?])\s+|\n", value):
                if not re.search(r"\b(?:only|none|never|all|excludes?|does not|do not)\b", sentence, re.I):
                    continue
                mentioned = [
                    record for record in records
                    if any(
                        _normal(alias) and _normal(alias) in _normal(sentence)
                        for alias in [record.get("label", ""), *record.get("aliases", [])]
                    )
                ]
                measurement_names = [
                    name for record in records for measurement in record.get("measurements", [])
                    for name in measurement.get("names", [])
                    if re.search(r"(?:BrdU|SA.*Gal|53BP1|AEM|DNA damage|senescence)", name, re.I)
                    and _normal(name) in _normal(sentence)
                ]
                if measurement_names and not mentioned:
                    errors.append("categorical condition claim collapses distinct experimental groups")
                    continue
                if re.search(r"\bonly\b", sentence, re.I) and measurement_names:
                    for record in records:
                        if record in mentioned:
                            continue
                        if any(
                            any(_normal(name) in {_normal(item) for item in measurement.get("names", [])} for name in measurement_names)
                            and set(measurement.get("qualifiers", [])).intersection({
                                "significant_increase", "senescent", "most_identified",
                            })
                            for measurement in record.get("measurements", [])
                        ):
                            errors.append("exclusive condition claim contradicts another grounded condition")
                            break

        for conclusion in evidence.get("author_conclusions", []):
            subject = str(conclusion.get("subject", ""))
            object_value = str(conclusion.get("object", ""))
            if conclusion.get("relation") == "not_merely_detector_of" and not re.search(
                rf"\b{re.escape(subject)}\b.{{0,180}}\bnot\s+(?:just|merely)\b"
                rf".{{0,100}}{re.escape(object_value)}",
                value, re.I | re.DOTALL,
            ):
                errors.append(f"missing explicit author conclusion for {subject}")

    numeric_field_patterns = {
        "screened": r"(?:\bscreen(?:ed|ing)\b.{0,60}?\b(\d+)\b|\b(\d+)\b.{0,60}?\bscreened\b)",
        "identified_total": r"(?:\bidentified\b.{0,50}?\b(\d+)\b.{0,30}\b(?:total\s+)?(?:hits?|drugs?|compounds?)\b|\b(\d+)\b\s+total\s+(?:hits?|drugs?|compounds?)\b)",
        "both": r"\b(\d+)\b.{0,70}\b(?:both|shared)\b",
        "toxicity_filter": r"\b(\d+(?:\.\d+)?\s*%)\b.{0,70}\b(?:toxic|toxicity|filter|threshold)\b",
        "bounded_value": r"\b(?:less|more)\s+than\s+(\d+(?:\.\d+)?\s*%)",
    }
    for fact in evidence.get("numeric_facts", []):
        expected = str(fact.get("value", ""))
        if expected and expected not in re.sub(r"\s+", "", value):
            errors.append(f"missing grounded numeric value: {expected}")
        key = str(fact.get("key", ""))
        pattern = numeric_field_patterns.get(key)
        if key.startswith("only_"):
            group = re.escape(key[5:].replace("_", " "))
            pattern = rf"\b(\d+)\b.{{0,70}}\b(?:only|specific)\b.{{0,30}}\b{group}\b"
        if pattern:
            asserted = {
                re.sub(r"\s+", "", group)
                for match in re.finditer(pattern, value, re.I | re.DOTALL)
                for group in match.groups() if group
            }
            if asserted and expected not in asserted:
                errors.append(
                    f"numeric value for {key} contradicts grounded value {expected}: "
                    f"{sorted(asserted)}"
                )

    # Detect two different counts asserted for the same categorical field.
    numeric_groups = defaultdict(set)
    for sentence in re.split(r"(?<=[.;!?])\s+|\n", value):
        patterns = {
            "screened": r"\b(?:screened|screening)\b.{0,60}?\b(\d+)\b|\b(\d+)\b.{0,60}?\bscreened\b",
            "both": (
                r"\b(\d+)\s+(?:(?:drugs?|compounds?|hits?)\b.{0,80}|"
                r"(?:was|were)\b.{0,50})\b(?:both|shared)\b"
            ),
            "total_hits": r"\b(?:identified|total)\b.{0,40}\b(\d+)\s+(?:drugs?|compounds?|hits?)\b",
        }
        for label, pattern in patterns.items():
            for match in re.finditer(pattern, sentence, re.I):
                number = next((item for item in match.groups() if item), None)
                if number:
                    numeric_groups[label].add(number)
    for label, numbers in numeric_groups.items():
        if len(numbers) > 1:
            errors.append(f"contradictory numeric values for {label}: {sorted(numbers)}")

    if evidence.get("answer_kind") == "experimental_domain_synthesis":
        resolved = [
            row for row in evidence.get("provenance", [])
            if row.get("figure_number") not in (None, "")
        ]
        for row in resolved:
            number = str(row["figure_number"])
            if not re.search(rf"\bfig(?:ure)?\.?\s*{re.escape(number)}\b", value, re.I):
                errors.append(
                    f"missing resolved Figure {number} for {row.get('experimental_domain')}"
                )
                continue
            domain_patterns = {
                "in_vitro_human_cell_line": r"\b(?:in vitro|cell[- ]line|cell culture)\b",
                "mouse_animal_tissue": r"\b(?:mouse|mice|animal tissue)\b",
                "human_patient_tissue": r"\b(?:human patient|patient tissue|clinical tissue)\b",
            }
            domain_pattern = domain_patterns.get(str(row.get("experimental_domain")), "")
            paragraphs = re.split(r"\n\s*\n|(?<=[.!?])\s+", value)
            if domain_pattern and not any(
                re.search(domain_pattern, paragraph, re.I)
                and re.search(rf"\bfig(?:ure)?\.?\s*{re.escape(number)}\b", paragraph, re.I)
                for paragraph in paragraphs
            ):
                errors.append(
                    f"Figure {number} is not attached to {row.get('experimental_domain')}"
                )
        if resolved and "figure number not resolved" in lower:
            errors.append("reports unresolved figure despite resolved provenance")
        allowed = {str(row["figure_number"]).casefold() for row in resolved}
        for match in re.finditer(
            r"\b(?:Supplementary\s+)?Fig(?:ure)?\.?\s*(S?\d+(?:\.\d+)?)\b",
            value, re.I,
        ):
            identifier = match.group(1).casefold()
            if (identifier.startswith("s") or "supplementary" in match.group(0).casefold()) and identifier not in allowed:
                errors.append(f"unverified supplementary figure: {match.group(0)}")
            elif identifier not in allowed:
                errors.append(f"unselected figure number: {match.group(0)}")

    if evidence.get("answer_kind") == "resolved_figure":
        expected_figure = str(evidence.get("source", {}).get("figure_number") or "").casefold()
        if expected_figure:
            for match in re.finditer(
                r"\b(?:Supplementary\s+)?Fig(?:ure)?\.?\s*(S?\d+(?:\.\d+)?)\b",
                value, re.I,
            ):
                identifier = match.group(1).casefold()
                if identifier != expected_figure:
                    errors.append(f"changed figure number: {match.group(0)}")

    return list(dict.fromkeys(errors))


def _caption_title(source_text: str, figure_number: str | None) -> tuple[str, str]:
    """Return the explicit caption claim and the remaining caption text."""
    text = _clean_text(source_text)
    number = re.escape(str(figure_number or ""))
    match = re.search(
        rf"\bFig(?:ure)?\s*{number}\s*\|\s*(.+?)(?=\.\s+[a-z](?:\s*[-–]\s*[a-z]|\s*,\s*[a-z])?\s+[A-Z]|\Z)",
        text, re.I,
    )
    if not match:
        return "", text
    return match.group(1).strip(" ."), text[match.end():].strip()


def _joined_figures(rows: list[dict]) -> str:
    figures = [f"Figure {row['figure_number']}" for row in rows if row.get("figure_number")]
    if not figures:
        return "figure number not resolved"
    if len(figures) == 1:
        return figures[0]
    return ", ".join(figures[:-1]) + f" and {figures[-1]}"


def _clean_explanatory_sentence(value: str) -> str:
    text = _clean_text(value)
    text = re.sub(r"^[a-z](?:\s*[-–]\s*[a-z]|\s*,\s*[a-z])?\s+", "", text)
    text = re.split(
        r"\b(?:Scale bars?|Error bars|Source Data are provided|Data represent|"
        r"Statistical significance|Significance was calculated)\b",
        text, maxsplit=1, flags=re.I,
    )[0]
    text = re.sub(r"\s+", " ", text).strip(" .;:")
    return f"{text}." if text else ""


def _diverse_caption_sentences(source_text: str, figure_number: str | None) -> dict:
    """Select experiment and measurement/result clauses without interpreting them."""
    title, remainder = _caption_title(source_text, figure_number)
    sentences = []
    for sentence in _sentences(remainder):
        cleaned = _clean_explanatory_sentence(sentence)
        if not cleaned or _PLACEHOLDER_RE.search(cleaned):
            continue
        if cleaned not in sentences:
            sentences.append(cleaned)

    experiments = [
        sentence for sentence in sentences
        if re.search(
            r"\b(?:experimental design|schematic|design of the experiments?|"
            r"analysis of|co-?cultures?|treated with|comparison of)\b",
            sentence, re.I,
        )
    ][:3]
    measurements = [
        sentence for sentence in sentences
        if sentence not in experiments and re.search(
            r"\b(?:quantification|percentage|correlation|score|staining|"
            r"predicted|classifier|distribution|positive cells?|activity)\b",
            sentence, re.I,
        )
    ][:6]
    if not experiments and sentences:
        experiments = sentences[:1]
    if not measurements:
        measurements = [row for row in sentences if row not in experiments][:2]
    return {"title": title, "experiments": experiments, "measurements": measurements}


def _render_provenance_item(row: dict, domain_label: str) -> str:
    figure = f"Figure {row['figure_number']}" if row.get("figure_number") else "Selected evidence"
    details = _diverse_caption_sentences(row.get("source_text", ""), row.get("figure_number"))
    parts = []
    sample = str(row.get("sample_type") or "").strip()
    if sample and not re.search(r"[-/]$|^(?:IHC|IF|H&E|staining)$", sample, re.I):
        parts.append(f"It studies {sample}.")
    if details["experiments"]:
        parts.append(f"Experiment: {' '.join(details['experiments'])}")
    if details["measurements"]:
        parts.append(f"Measurements and reported results: {' '.join(details['measurements'])}")
    if details["title"]:
        parts.append(f"Direct contribution: the caption states that {details['title'].rstrip('.')}.")
    parts.append(f"This is selected evidence for the requested {domain_label.casefold()} comparison.")
    return f"- **{figure}:** {' '.join(parts)}"


def _measurement_semantics(name: str, statement: str) -> tuple[str, str]:
    """Type a displayed measurement so unrelated qualifiers cannot migrate to it."""
    value = str(name or "")
    combined = f"{value} {statement}"
    if re.search(r"\bBrdU\b", value, re.I):
        return "BrdU incorporation", "proliferation / DNA synthesis / cell-cycle activity"
    if re.search(r"SA\s*-?\s*[βB]-?Gal", value, re.I):
        return "SA-β-Gal staining", "senescence-associated marker"
    if re.search(r"\b53BP1\b", value, re.I):
        return "53BP1 foci", "DNA-damage marker"
    if re.search(r"\b(?:γH2AX|DNA damage)\b", value, re.I):
        return "detectable DNA damage", "DNA-damage marker"
    if re.search(r"\b(?:classifier|predicted|identified)\b", combined, re.I) and re.fullmatch(
        r"[A-Z][A-Z0-9-]{1,15}", value,
    ) and value not in {"DNA", "DD"}:
        return value, "classifier prediction"
    if re.search(r"\b(?:p21|p53)\b", value, re.I):
        return value, "cell-cycle / senescence marker"
    return value or "measured outcome", "measured outcome"


def _typed_measurement_records(record: dict) -> list[dict]:
    """Create renderer-local name/role/observation records from grounded tuples."""
    typed = []
    for measurement in record.get("measurements", []):
        names = [str(name) for name in measurement.get("names", [])]
        statement = str(measurement.get("statement", ""))
        typed_names = [_measurement_semantics(name, statement) for name in names]

        def first_role(role: str) -> tuple[str, str] | None:
            return next((item for item in typed_names if item[1] == role), None)

        classifier = first_role("classifier prediction")
        damage = first_role("DNA-damage marker")
        senescence_marker = first_role("senescence-associated marker")
        proliferation = first_role("proliferation / DNA synthesis / cell-cycle activity")
        qualifiers = list(dict.fromkeys(measurement.get("qualifiers", [])))
        for qualifier in qualifiers:
            item = None
            if qualifier == "significant_increase":
                name, role = damage or typed_names[0] if typed_names else ("measured outcome", "measured outcome")
                item = {"name": name, "role": role, "observation": "showed a significant increase"}
            elif qualifier == "no_significant_change":
                if classifier and re.search(r"DNA damage", statement, re.I):
                    item = {
                        "name": classifier[0], "role": classifier[1],
                        "observation": (
                            "identified the condition as senescent despite no significant "
                            "detectable DNA-damage increase"
                        ),
                    }
                else:
                    name, role = damage or typed_names[0] if typed_names else ("measured outcome", "measured outcome")
                    item = {"name": name, "role": role, "observation": "showed no significant increase"}
            elif qualifier == "bounded_value":
                bounded = re.search(r"\bless than\s+\d+(?:\.\d+)?\s*%", statement, re.I)
                name, role = classifier or ("stated classifier", "classifier prediction")
                observation = (
                    f"{bounded.group(0).capitalize()} of cells were identified as senescent by {name}"
                    if bounded else "retained the stated upper bound"
                )
                item = {
                    "name": "Classifier result" if bounded else name,
                    "role": role, "observation": observation,
                }
            elif qualifier == "most_identified":
                name, role = classifier or ("stated classifier", "classifier prediction")
                item = {"name": name, "role": role, "observation": "identified most cells as senescent"}
            elif qualifier in {"senescent", "not_established_senescent"}:
                if qualifier == "senescent" and "most_identified" in qualifiers:
                    continue
                name, role = senescence_marker or classifier or ("senescence status", "senescence-associated status")
                observation = (
                    "confirmed that the condition was senescent"
                    if qualifier == "senescent"
                    else "did not establish the condition as senescent"
                )
                item = {"name": name, "role": role, "observation": observation}
            elif qualifier == "dividing":
                name, role = proliferation or ("cell-cycle activity", "proliferation / DNA synthesis")
                item = {"name": name, "role": role, "observation": "showed that the cells were dividing"}
            elif qualifier == "arrested":
                item = {
                    "name": "Cell-cycle activity", "role": "cell-cycle status",
                    "observation": "showed that the cells were arrested",
                }
            elif qualifier == "selective_decrease":
                name, role = typed_names[0] if typed_names else ("measured outcome", "measured outcome")
                item = {"name": name, "role": role, "observation": "showed a selective decrease"}
            if item:
                typed.append(item)

    # Prefer one informative classifier conclusion over a duplicate generic one.
    if any(
        row["role"] == "classifier prediction" and re.search(r"\b(?:most|despite)\b", row["observation"])
        for row in typed
    ):
        typed = [
            row for row in typed
            if not (
                row["role"] == "classifier prediction"
                and row["observation"] == "confirmed that the condition was senescent"
            )
        ]
    specific_classifier_observations = {
        _normal(row["observation"])
        for row in typed
        if row["role"] == "classifier prediction"
        and _normal(row["name"]) not in {"stated classifier", "classifier result"}
    }
    if specific_classifier_observations:
        typed = [
            row for row in typed
            if not (
                row["role"] == "classifier prediction"
                and _normal(row["name"]) in {"stated classifier", "classifier result"}
                and _normal(row["observation"]) in specific_classifier_observations
            )
        ]
    has_named_bounded_classifier = any(
        row["role"] == "classifier prediction"
        and "less than" in _normal(row["observation"])
        and "stated classifier" not in _normal(row["observation"])
        for row in typed
    )
    if has_named_bounded_classifier:
        typed = [
            row for row in typed
            if not (
                row["role"] == "classifier prediction"
                and "less than" in _normal(row["observation"])
                and "stated classifier" in _normal(row["observation"])
            )
        ]
    unique = []
    seen = set()
    for row in typed:
        signature = (_normal(row["name"]), _normal(row["role"]), _normal(row["observation"]))
        if signature not in seen:
            unique.append(row)
            seen.add(signature)
    return unique


def _render_condition_facts(record: dict) -> list[str]:
    facts = []
    for row in _typed_measurement_records(record):
        if row["name"] == "Classifier result":
            facts.append(f"{row['observation']} (role: {row['role']}).")
        else:
            facts.append(f"{row['name']} {row['observation']} (role: {row['role']}).")
    return facts


def _classifier_model_types(facts: list[str]) -> list[tuple[str, str]]:
    combined = " ".join(facts)
    rows = []
    patterns = (
        r"\b([A-Z][A-Z0-9]{1,15})(?:CP)?\s*\(classification tree-based\)",
        r"\b([A-Z][A-Z0-9]{1,15})(?:CP)?\s*\(random forest-based\)",
    )
    for pattern, model_type in zip(patterns, ("classification-tree-based", "random-forest-based")):
        for match in re.finditer(pattern, combined, re.I):
            name = re.sub(r"CP$", "", match.group(1), flags=re.I)
            pair = (name, model_type)
            if pair not in rows:
                rows.append(pair)
    return rows


def _render_training_validation_logic(evidence: dict) -> list[str]:
    """Explain grounded train/test logic when caption roles explicitly define it."""
    panels = evidence.get("panels", [])
    by_role = {str(row.get("role")): row for row in panels}
    required_roles = {"training_workflow", "training_result", "test_validation_result"}
    if not required_roles.issubset(by_role):
        return []
    facts = [str(item) for item in evidence.get("explicit_facts", [])]
    model_types = _classifier_model_types(facts)
    training = str(by_role["training_result"].get("description", ""))
    validation = str(by_role["test_validation_result"].get("description", ""))
    combined_panels = f"{training} {validation}"
    treatments = re.findall(r"\b([A-Za-z0-9-]+)-treated\b", combined_panels, re.I)
    normal = re.search(r"([A-Za-z0-9-]+)(?:-treated)?\s*\(normal\)", combined_panels, re.I)
    senescent = re.search(r"([A-Za-z0-9-]+)-treated[^.]{0,120}\(senescent\)", combined_panels, re.I)

    lines = []
    if model_types:
        rendered_models = " and ".join(
            f"{name} is the {model_type} model" for name, model_type in model_types
        )
        lines.append(f"Model design: {rendered_models}.")
    if normal and senescent:
        lines.append(
            "Training labels followed the stated treatment assumption: "
            f"{senescent.group(1)}-treated cells were treated as senescent and "
            f"{normal.group(1)}-treated cells as normal/non-senescent."
        )
    elif len(dict.fromkeys(treatments)) >= 2:
        values = list(dict.fromkeys(treatments))[:2]
        lines.append(
            f"The training datasets compared {values[0]}-treated and {values[1]}-treated cells."
        )

    training_fact = next((
        fact for fact in facts
        if re.search(r"training sets?", fact, re.I)
        and re.search(r"both classifiers", fact, re.I)
        and re.search(r"similar extent.*SA-?β-Gal", fact, re.I)
    ), "")
    if training_fact:
        lines.append(
            "Training result: both classifiers identified senescence in the treated "
            "training cells to a similar extent as SA-β-Gal staining."
        )
    test_fact = next((
        fact for fact in facts
        if re.search(r"validated with test data from new samples", fact, re.I)
    ), "")
    if test_fact or re.search(r"\b(?:test|validation) datasets?\b", validation, re.I):
        lines.append(
            "Independent validation: the classifiers were then evaluated on new test "
            "samples, with predictions compared on the same cells as SA-β-Gal staining."
        )
    no_single_feature = any(
        re.search(r"none of these nuclear features alone could distinguish", fact, re.I)
        for fact in facts
    )
    if no_single_feature and model_types:
        lines.append(
            "Together, the training and independent test results support using the "
            "combined nuclear-morphology features, because the Results state that no "
            "single feature alone was sufficient to distinguish senescent cells."
        )
    return lines


def _render_stage(stage: dict, numeric_facts: list[dict]) -> str:
    facts = stage.get("facts", [])
    selected = []
    for preferred_pattern, fallback_pattern in (
        (r"\bco-?culture assay\b", r"\bco-?cultures? of senescent\b"),
        (r"\bwe treated the co-?cultures\b", r"\bafter treatment with\b"),
        (r"\bnon-senescent\b.{0,180}\bselectively reduced\b", None),
    ):
        matches = [fact for fact in facts if re.search(preferred_pattern, fact, re.I)]
        complete_matches = [
            fact for fact in matches
            if not re.match(r"^[a-z](?:\s|$)", fact)
            and not re.search(r"\bpredicted to be senescent by\s+To\b", fact, re.I)
        ]
        if not complete_matches and fallback_pattern:
            fallback_matches = [
                fact for fact in facts if re.search(fallback_pattern, fact, re.I)
            ]
            complete_matches = [
                fact for fact in fallback_matches
                if not re.match(r"^[a-z](?:\s|$)", fact)
                and not re.search(r"\bpredicted to be senescent by\s+To\b", fact, re.I)
            ]
            matches = fallback_matches
        sentence = max(complete_matches or matches, key=len) if matches else None
        if sentence and sentence not in selected:
            selected.append(sentence)
    if stage.get("stage") == 1 and selected:
        return " ".join(selected)

    stage_numbers = numeric_facts if stage.get("stage") != 1 else []
    by_key = {row.get("key"): row for row in stage_numbers}
    rendered = []
    if row := by_key.get("screened"):
        noun = "drugs" if re.search(r"\bdrugs?\b", row.get("field", ""), re.I) else "candidates"
        rendered.append(f"{row['value']} {noun} were screened.")
    if row := by_key.get("toxicity_filter"):
        rendered.append(f"Toxic candidates above the stated {row['value']} filter were excluded.")
    if row := by_key.get("identified_total"):
        classifier = re.search(r"\b([A-Z][A-Z0-9-]{1,15})\s+classifier\b", row.get("field", ""))
        subject = f"The {classifier.group(1)} classifier" if classifier else "The analysis"
        rendered.append(f"{subject} identified {row['value']} total hits.")
    for key, row in by_key.items():
        if str(key).startswith("only_"):
            group = str(key)[5:].upper()
            rendered.append(f"{row['value']} hits were specific to {group}.")
    if row := by_key.get("both"):
        rendered.append(f"{row['value']} hits were active in both groups.")
    if rendered:
        rendered_text = " ".join(rendered)
        missing_terms = [
            term for term in stage.get("required_terms", [])
            if _normal(term) not in _normal(rendered_text)
        ]
        if missing_terms:
            rendered.insert(0, f"The stage included {', '.join(missing_terms)}.")
        return " ".join(rendered)

    clean_facts = [
        fact for fact in facts
        if not re.search(r"\b(?:Scale bar|Data represent|Source Data)\b", fact, re.I)
    ]
    return " ".join(clean_facts[:3])


def _multi_stage_panel_groups(evidence: dict) -> list[dict]:
    """Derive renderer-local stage boundaries from caption panel ordering."""
    panels = evidence.get("panels", [])
    starts = [
        index for index, panel in enumerate(panels)
        if panel.get("role") == "experimental_design"
    ]
    if len(starts) < 2 or not evidence.get("experiment_stages"):
        return []
    groups = []
    for offset, start in enumerate(starts):
        end = starts[offset + 1] if offset + 1 < len(starts) else len(panels)
        stage_panels = panels[start:end]
        groups.append({
            "stage": offset + 1,
            "label": stage_panels[0].get("description", "") if stage_panels else "",
            "panels": {
                str(panel.get("panel", "")).casefold()
                for panel in stage_panels if panel.get("panel")
            },
            "descriptions": [
                str(panel.get("description", "")) for panel in stage_panels
                if panel.get("description")
            ],
        })
    return groups


def _stage_entity_tokens(values: list[str]) -> set[str]:
    combined = " ".join(values)
    return {
        _normal(match.group(0))
        for match in re.finditer(
            r"\b(?:[A-Z][A-Za-z]*\d+[A-Za-z0-9-]*|[A-Z]{2,}[A-Za-z0-9-]*|"
            r"m[A-Z][A-Za-z0-9-]+)\b",
            combined,
        )
        if _normal(match.group(0)) not in {"fig", "figure", "data"}
    }


def _referenced_stage_panels(text: str, figure_number: str) -> set[str]:
    panels = set()
    pattern = (
        rf"\bFig(?:ure)?\.?\s*{re.escape(figure_number)}\s*"
        r"([a-z])(?:\s*[-–—]\s*([a-z]))?(?:\s*,\s*([a-z]))?"
    )
    for match in re.finditer(pattern, text, re.I):
        first, last, extra = (value.casefold() if value else "" for value in match.groups())
        panels.add(first)
        if last:
            panels.update(chr(value) for value in range(ord(first), ord(last) + 1))
        if extra:
            panels.add(extra)
    return panels


def _fact_allowed_in_stage(
    fact: str, *, figure_number: str, group: dict,
    own_entities: set[str], other_entities: set[str], default: bool,
) -> bool:
    main_refs = re.findall(
        r"(?<!Supplementary\s)\bFig(?:ure)?\.?\s*(\d+(?:\.\d+)?)",
        fact, re.I,
    )
    if main_refs and any(number != figure_number for number in main_refs):
        return False
    if re.match(r"^[a-z](?:\s|$)", fact) and not re.search(
        rf"\bFig(?:ure)?\.?\s*{re.escape(figure_number)}\b",
        fact, re.I,
    ):
        return False
    panel_refs = _referenced_stage_panels(fact, figure_number)
    if panel_refs:
        return bool(panel_refs.intersection(group["panels"]))
    normal_fact = _normal(fact)
    own_hits = {token for token in own_entities if token and token in normal_fact}
    other_hits = {token for token in other_entities if token and token in normal_fact}
    # A fact containing an entity owned exclusively by another stage must not
    # cross the boundary merely because it also contains a shared/current-stage
    # entity. Explicit panel references above are the grounded exception.
    if other_hits:
        return False
    if own_hits and not other_hits:
        return True
    return default


def _renderer_stage_plan(evidence: dict) -> list[dict]:
    """Recover missing caption-defined stages without mutating authoritative evidence."""
    existing = {
        int(stage.get("stage")): stage
        for stage in evidence.get("experiment_stages", [])
        if str(stage.get("stage", "")).isdigit()
    }
    groups = _multi_stage_panel_groups(evidence)
    if not groups:
        return list(evidence.get("experiment_stages", []))

    figure_number = str(evidence.get("source", {}).get("figure_number") or "")
    entity_sets = [_stage_entity_tokens(group["descriptions"]) for group in groups]
    explicit = [str(fact) for fact in evidence.get("explicit_facts", [])]
    rendered = []
    for index, group in enumerate(groups):
        stage_number = int(group["stage"])
        original = existing.get(stage_number, {})
        own_entities = entity_sets[index]
        other_entities = set().union(*(
            entities for offset, entities in enumerate(entity_sets) if offset != index
        )).difference(own_entities)
        facts = list(group["descriptions"])
        for fact in original.get("facts", []):
            if _fact_allowed_in_stage(
                str(fact), figure_number=figure_number, group=group,
                own_entities=own_entities, other_entities=other_entities, default=True,
            ) and fact not in facts:
                facts.append(str(fact))
        for fact in explicit:
            if _fact_allowed_in_stage(
                fact, figure_number=figure_number, group=group,
                own_entities=own_entities, other_entities=other_entities, default=False,
            ) and fact not in facts:
                facts.append(fact)

        allowed_text = _normal(" ".join(facts))
        required_terms = [
            str(term) for term in original.get("required_terms", [])
            if _normal(term) and _normal(term) in allowed_text
        ]
        if not required_terms:
            required_terms = _stage_required_terms(facts)
        rendered.append({
            "stage": stage_number,
            "slot": original.get("slot") or f"caption stage {stage_number}",
            "label": group["label"] or original.get("label", ""),
            "facts": facts,
            "required_terms": required_terms,
            "authority": original.get("authority") or "explicit_full_caption",
        })
    return rendered


def _stage_heading(stage: dict) -> str:
    label = _clean_text(stage.get("label", "")).strip(" .")
    if re.search(r"\bsenolytic\b", label, re.I):
        return "Senolytic evaluation"
    if re.search(r"\b(?:screen|drugs? inducing senescence)\b", label, re.I):
        return "Senescence-inducing compound screen"
    return label or str(stage.get("slot") or "Grounded experiment")


def render_final_answer_evidence(evidence: dict) -> str:
    """Deterministic safe rendering used only after generation and repair fail."""
    if evidence.get("answer_kind") == "experimental_domain_synthesis":
        grouped = defaultdict(list)
        for row in evidence.get("provenance", []):
            grouped[row.get("experimental_domain", "other")].append(row)
        labels = {
            "in_vitro_human_cell_line": "In vitro human cell-line evidence",
            "mouse_animal_tissue": "Mouse animal/tissue evidence",
            "human_patient_tissue": "Human patient-tissue evidence",
        }
        blocks = []
        for domain in (
            "in_vitro_human_cell_line", "mouse_animal_tissue", "human_patient_tissue",
        ):
            rows = grouped.get(domain, [])
            if not rows:
                continue
            blocks.append(f"**{labels[domain]} — {_joined_figures(rows)}**")
            blocks.extend(_render_provenance_item(row, labels[domain]) for row in rows)
        return "\n\n".join(blocks)

    source = evidence.get("source", {})
    figure = source.get("figure_number")
    lines = [f"**Figure {figure} — grounded explanation**" if figure else "**Grounded figure explanation**"]
    panels = evidence.get("panels", [])
    if panels:
        lines.append("")
        lines.extend(
            f"- **Panel {row['panel']}:** {row['description']}"
            for row in panels
        )
    feature_set = evidence.get("feature_set") or {}
    if feature_set.get("items"):
        features = ", ".join(feature_set["items"])
        exception = feature_set.get("exception")
        comparison = feature_set.get("comparison")
        sentence = f"The measured nuclear features were {features}."
        if exception and comparison:
            sentence += f" All except {exception} differed significantly between {comparison}."
        lines.extend(["", sentence])
    training_logic = _render_training_validation_logic(evidence)
    if training_logic:
        lines.extend(["", "**Training and validation logic**", *training_logic])
    for stage in _renderer_stage_plan(evidence):
        rendered_stage = _render_stage(stage, evidence.get("numeric_facts", []))
        lines.extend(["", f"**Stage {stage['stage']} — {_stage_heading(stage)}:** {rendered_stage}"])
    if evidence.get("condition_records"):
        lines.extend(["", "**Condition-level Results**"])
        for row in evidence["condition_records"]:
            facts = _render_condition_facts(row)
            if facts:
                lines.append(f"- **{row['label']}:** {' '.join(facts)}")
    for conclusion in evidence.get("author_conclusions", []):
        if conclusion.get("relation") == "not_merely_detector_of":
            lines.extend([
                "",
                f"Together, these results show that {conclusion['subject']} is not merely "
                f"a detector of {conclusion['object']}.",
            ])
    return "\n".join(lines).strip()
