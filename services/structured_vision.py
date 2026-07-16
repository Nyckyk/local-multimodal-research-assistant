import json
import math
import re
import time
import unicodedata
from copy import deepcopy
from html import unescape
from pathlib import Path

import fitz
import ollama

from settings import VISION_MODEL

STRUCTURED_NUM_PREDICT = 1800
COMPACT_PANEL_NUM_PREDICT = 650
COMPACT_COMPARISON_NUM_PREDICT = 350
AXIS_LABEL_NUM_PREDICT = 220
NYQUIST_FIT_NUM_PREDICT = 260
VISUAL_TYPES = {"labelled_diagram", "graph", "table"}
VISUAL_IDENTIFIER_PATTERN = r"(?:[A-Za-z]\.)?\d+(?:\.\d+)?|[A-Za-z]\d+"


class StructuredOutputError(ValueError):
    pass


class TruncatedJSONError(StructuredOutputError):
    pass


def normalize_impedance_axis_label(label: str, evidence_text: str = "") -> str:
    """Canonicalise a fat-impedance subscript without inventing one."""
    original = unescape(str(label or "")).strip()
    if not original:
        return original

    normalized = original
    normalized = re.sub(
        r"Z\s*<sub>\s*fat\s*</sub>", "Zfat", normalized,
        flags=re.IGNORECASE,
    )
    normalized = re.sub(
        r"Z\s*_?\s*\{?\s*(?:\\(?:mathrm|text)\s*\{\s*)?fat\s*\}?\s*\}?",
        "Zfat", normalized, flags=re.IGNORECASE,
    )
    normalized = re.sub(
        r"Z\s*(?:_|₍|\()\s*fat\s*(?:₎|\))?",
        "Zfat", normalized, flags=re.IGNORECASE,
    )
    normalized = re.sub(r"Z\s*fat", "Zfat", normalized, flags=re.IGNORECASE)

    compact = re.sub(r"[^a-z0-9]+", "", normalized.lower())
    has_fat_subscript = "zfat" in compact
    generic_z = bool(re.search(r"\|?\s*Z\s*\|?", normalized, re.IGNORECASE))
    grounded_fat_impedance = bool(re.search(
        r"\b(?:adipose(?:[-\s]+tissue)?|fat(?:[-\s]+tissue)?)\b"
        r".{0,80}\bimpedance\b|\bimpedance\b.{0,80}"
        r"\b(?:adipose(?:[-\s]+tissue)?|fat(?:[-\s]+tissue)?)\b",
        evidence_text, re.IGNORECASE | re.DOTALL,
    ))
    if not has_fat_subscript and generic_z and grounded_fat_impedance:
        normalized = re.sub(
            r"Z(?!\s*fat)", "Zfat", normalized,
            count=1, flags=re.IGNORECASE,
        )
    return normalized


def _axis_label_needs_retry(label: str, confidence) -> bool:
    text = str(label or "")
    compact = re.sub(r"[^a-z0-9]+", "", unescape(text).lower())
    low_confidence = not isinstance(confidence, (int, float)) or confidence < 0.7
    base_without_fat_subscript = "z" in compact and "zfat" not in compact
    partially_readable = not text.strip() or bool(re.search(
        r"\b(?:unknown|unreadable|partial|uncertain)\b", text, re.IGNORECASE,
    ))
    return low_confidence or base_without_fat_subscript or partially_readable


def _axis_label_retry_prompt(evidence_text: str) -> str:
    return f"""
Read only the rotated magnitude y-axis label in these tight label-strip crops.
Preserve every visible mathematical subscript and nearby qualifier. Distinguish
subscripts attached to Z from subscripts attached to the enclosing magnitude.
Do not infer a missing subscript from general scientific knowledge. Return JSON
only, with the exact best reading and a confidence from 0 to 1:
{{"label": "...", "confidence": 0.0, "subscript_visible": true}}

Immediately associated caption/text (grounding only; never overrides pixels):
{evidence_text[:1200]}
"""


def reread_magnitude_axis_label(
    image_paths: list[Path], evidence_text: str
) -> tuple[str, str]:
    """Reread a shared y label from tight crops and return canonical label/raw."""
    raw = _call_model_images(
        image_paths, _axis_label_retry_prompt(evidence_text), AXIS_LABEL_NUM_PREDICT,
    )
    value = parse_json_response(raw)
    label = value.get("label")
    confidence = value.get("confidence")
    if not isinstance(label, str) or not label.strip():
        raise StructuredOutputError("Targeted axis-label retry returned no label.")
    if not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
        raise StructuredOutputError("Targeted axis-label retry confidence is invalid.")
    return normalize_impedance_axis_label(label, evidence_text), raw


def detect_visual_type(question: str, page_text: str = "") -> str | None:
    question_lower = question.lower()
    if re.search(
        rf"\btable\s*(?:{VISUAL_IDENTIFIER_PATTERN})\b", question_lower
    ):
        return "table"

    relevant_text = question_lower
    figure_match = re.search(
        rf"\bfig(?:ure)?\.?\s*({VISUAL_IDENTIFIER_PATTERN})", question_lower
    )
    if figure_match:
        number = re.escape(figure_match.group(1))
        caption = re.search(
            rf"\bfig(?:ure)?\.?\s*{number}\b(.{{0,500}})",
            page_text,
            re.IGNORECASE | re.DOTALL,
        )
        if caption:
            relevant_text += " " + caption.group(0).lower()

    if re.search(
        r"\b(graph|plot|chart|bode|nyquist|axis|axes|curve|trend|panel|spectrum)\b",
        relevant_text,
    ):
        return "graph"
    if re.search(
        r"\b(diagram|schematic|circuit|topology|component|connection|labelled|labeled)\b",
        relevant_text,
    ):
        return "labelled_diagram"
    if figure_match:
        return "labelled_diagram"
    return None


def _schema_text(visual_type: str) -> str:
    if visual_type == "labelled_diagram":
        return """{
  "diagram_kind": "circuit|other",
  "labels": ["..."],
  "components": [{"name": "...", "description": "..."}],
  "spatial_relationships": [
    {"subject": "...", "relationship": "...", "object": "..."}
  ],
  "connections": [
    {"from": "...", "to": "...", "relationship": "..."}
  ],
  "circuit_topology": {
    "nodes": [{"id": "...", "label": "wire junction or terminal"}],
    "edges": [{"from_node": "...", "to_node": "...", "component": "..."}],
    "branches": [
      {"id": "...", "start_node": "...", "end_node": "...", "components": ["..."]}
    ],
    "parallel_branch_sets": [["branch id", "branch id"]]
  },
  "explanation": "...",
  "uncertain_items": ["..."]
}"""
    if visual_type == "graph":
        return """{
  "figure_number": "...",
  "panels": [
    {
      "panel": "...",
      "graph_kind": "bode_magnitude|bode_phase|nyquist|other",
      "group": "... or null",
      "x_axis": {
        "label": "...", "unit": "...", "scale": "linear|log|unknown",
        "tick_labels": ["..."], "scientific_multiplier": "... or null"
      },
      "y_axis": {
        "label": "...", "unit": "...", "scale": "linear|log|unknown",
        "tick_labels": ["..."], "scientific_multiplier": "... or null"
      },
      "series": ["..."],
      "visible_range": {"min": 0.0, "max": 0.0, "unit": "...", "confidence": 0.0},
      "shape_features": ["..."],
      "complexity_score": 0.0,
      "visible_trends": ["..."]
    }
  ],
  "comparisons": [
    {
      "claim": "...", "subject": "...", "relation": "highest|lowest|most_complex|other",
      "metric": "...", "evidence": ["vision|text"], "confidence": 0.0,
      "uncertain": false, "evidence_conflict": false
    }
  ],
  "frequency_direction_evidence": ["..."],
  "uncertain_values": ["..."]
}"""
    if visual_type == "table":
        return """{
  "table_number": "...",
  "title": "...",
  "columns": ["..."],
  "rows": [["...", null]],
  "units": {"column name": "unit or null"},
  "comparisons": ["..."],
  "unreadable_cells": [{"row": 0, "column": "..."}]
}"""
    raise ValueError(f"Unsupported visual type: {visual_type}")


def build_structured_prompt(
    visual_type: str,
    question: str,
    evidence_text: str = "",
) -> str:
    evidence = evidence_text[:6000]
    type_rules = {
        "labelled_diagram": (
            "Identify visible labels/components and explicit spatial or connective "
            "relationships. In a circuit, every connection endpoint must name a listed "
            "component, node or branch. In a biological diagram, a relationship endpoint "
            "may instead name a structure explicitly present in the caption/text. "
            "Check scientific relationships against the supplied caption/text when it "
            "explicitly describes them; preserve uncertainty on conflicts. For a circuit, "
            "set diagram_kind to circuit and populate circuit_topology with nodes, edges, "
            "branches and parallel_branch_sets. Each branch must name its shared start/end "
            "nodes and list separate component IDs in traversal order; never combine two "
            "components into one edge string. Connections may reference component IDs, "
            "visible biological labels, node IDs or branch IDs. For a non-circuit, set "
            "circuit_topology to null. Ignore anything outside the cropped target figure."
        ),
        "graph": (
            "Treat each visible panel independently. Preserve exact readable axis variable "
            "labels. Record visible tick labels and any scientific-notation multiplier "
            "separately. Infer scale from tick values and spacing: a multiplier such as "
            "x10^5 does not make an axis logarithmic. Before comparisons, independently "
            "estimate each panel's visible range and shape/complexity. Build comparisons "
            "from those estimates, cross-check explicit caption/body statements, and set "
            "uncertain/evidence_conflict when visual and text evidence disagree. In a "
            "Nyquist plot describe left/right or low/high Re(Z); never infer frequency "
            "direction unless arrows, frequency labels, or grounded text establish it. "
            "For model-versus-experimental plots, describe the overall visual fit quality "
            "for every panel before comparing which fit is closer. Identify where each "
            "panel's largest visible deviation occurs using left/right or Re(Z), even when "
            "the overall fit is good. Use the exact panel sample/group as the subject of "
            "fit and deviation comparisons, not a generic phrase such as fit quality. "
            "Compare closeness using the largest visible model-data separation divided by "
            "that panel's displayed y-axis range, not raw separation between differently "
            "scaled panels. "
            "Do not call a fit excellent, perfect, or superimposed across the entire range "
            "when a visible deviation remains. Apply each scientific multiplier to the "
            "numeric y-axis tick range reported in visible_range."
        ),
        "table": (
            "Transcribe columns and rows in display order. Repeat visually merged group "
            "cells on each applicable row. Every row must contain exactly one value per "
            "column. Use null for unreadable cells and record their positions."
        ),
    }[visual_type]
    return f"""
Analyse this research-paper {visual_type.replace('_', ' ')}.

Question: {question}

{type_rules}

Use only visible content and the supplied document evidence. Never invent an
unreadable value; use null or "unreadable". Return exactly one JSON object with
no Markdown or prose, following this schema exactly:
{_schema_text(visual_type)}

Caption/retrieved text for validation:
{evidence}
"""


def strip_json_fences(raw: str) -> str:
    text = raw.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    return fenced.group(1).strip() if fenced else text


def parse_json_response(raw: str) -> dict:
    text = strip_json_fences(raw)
    try:
        value = json.loads(text)
    except (json.JSONDecodeError, TypeError) as error:
        # Extract a complete object from a short model preface without repairing
        # or otherwise changing its structured data.
        if isinstance(error, json.JSONDecodeError):
            decoder = json.JSONDecoder()
            for start in (match.start() for match in re.finditer(r"\{", text)):
                try:
                    value, _ = decoder.raw_decode(text[start:])
                    break
                except json.JSONDecodeError:
                    continue
            else:
                value = None
            if isinstance(value, dict):
                return value
        if isinstance(error, json.JSONDecodeError):
            truncated = (
                "unterminated string" in error.msg.lower()
                or error.pos >= max(0, len(text) - 3)
                or text.count("{") > text.count("}")
                or text.count("[") > text.count("]")
            )
            if truncated:
                raise TruncatedJSONError(
                    f"Response JSON was truncated: {error}"
                ) from error
        raise StructuredOutputError(f"Response was not valid JSON: {error}") from error
    if not isinstance(value, dict):
        raise StructuredOutputError("Structured response must be a JSON object.")
    return value


def _require_keys(value: dict, keys: set[str], name: str):
    missing = keys.difference(value)
    if missing:
        raise StructuredOutputError(f"{name} is missing fields: {sorted(missing)}")


def _normal_name(value) -> str:
    compatible = unicodedata.normalize("NFKC", str(value))
    return re.sub(r"[^a-z0-9]+", "", compatible.lower())


def _validate_circuit_topology(
    topology: dict,
    component_names: set[str],
    physiological_labels: set[str] | None = None,
) -> None:
    _require_keys(
        topology,
        {"nodes", "edges", "branches", "parallel_branch_sets"},
        "Circuit topology",
    )
    if not all(
        isinstance(topology[key], list)
        for key in ("nodes", "edges", "branches", "parallel_branch_sets")
    ):
        raise StructuredOutputError("Circuit topology fields must be lists.")

    physiological_labels = {
        label for label in (physiological_labels or set()) if len(label) >= 3
    }
    node_ids = []
    for node in topology["nodes"]:
        if (
            not isinstance(node, dict)
            or not isinstance(node.get("id"), str)
            or not node["id"].strip()
        ):
            raise StructuredOutputError("Each circuit node needs a string id.")
        node_text = _normal_name(
            f"{node['id']} {node.get('label', '') if isinstance(node.get('label'), str) else ''}"
        )
        for label in physiological_labels:
            concept_node_forms = {
                label,
                f"{label}node", f"node{label}",
                f"{label}terminal", f"terminal{label}",
                f"{label}junction", f"junction{label}",
            }
            if any(form in node_text for form in concept_node_forms):
                raise StructuredOutputError(
                    "Physiological labels cannot be used as electrical circuit nodes."
                )
        node_ids.append(node["id"])
    if len(set(node_ids)) != len(node_ids):
        raise StructuredOutputError("Circuit node ids must be unique.")
    node_set = set(node_ids)

    edges_by_component = {}
    for edge in topology["edges"]:
        if not isinstance(edge, dict):
            raise StructuredOutputError("Each circuit edge must be an object.")
        _require_keys(edge, {"from_node", "to_node", "component"}, "Circuit edge")
        if edge["from_node"] not in node_set or edge["to_node"] not in node_set:
            raise StructuredOutputError("A circuit edge references an unknown node.")
        if re.search(r"\b(?:and|series|parallel|branch)\b", str(edge["component"]), re.I):
            raise StructuredOutputError(
                "Circuit edges must reference one component ID, not a combined component string."
            )
        component_key = _normal_name(edge["component"])
        if component_names and component_key not in component_names:
            raise StructuredOutputError("A circuit edge references an unknown component.")
        if component_key in edges_by_component:
            raise StructuredOutputError("A circuit component appears on more than one edge.")
        edges_by_component[component_key] = edge

    branches = {}
    branch_component_keys = set()
    for branch in topology["branches"]:
        if not isinstance(branch, dict):
            raise StructuredOutputError("Each circuit branch must be an object.")
        _require_keys(
            branch,
            {"id", "start_node", "end_node", "components"},
            "Circuit branch",
        )
        if branch["id"] in branches:
            raise StructuredOutputError("Circuit branch ids must be unique.")
        if branch["start_node"] not in node_set or branch["end_node"] not in node_set:
            raise StructuredOutputError("A circuit branch references an unknown node.")
        if not isinstance(branch["components"], list) or not branch["components"]:
            raise StructuredOutputError("A circuit branch must contain components.")

        # Verify that the ordered component edges form one continuous path
        # between the branch's declared wire junctions.
        current = branch["start_node"]
        for component in branch["components"]:
            component_key = _normal_name(component)
            edge = edges_by_component.get(component_key)
            if edge is None:
                raise StructuredOutputError("A circuit branch references an unknown edge component.")
            branch_component_keys.add(component_key)
            endpoints = {edge["from_node"], edge["to_node"]}
            if current not in endpoints:
                raise StructuredOutputError("Circuit branch components do not form a continuous path.")
            current = next(iter(endpoints - {current}))
        if current != branch["end_node"]:
            raise StructuredOutputError("A circuit branch does not end at its declared junction.")
        branches[branch["id"]] = branch

    if branch_component_keys != set(edges_by_component):
        raise StructuredOutputError(
            "Every circuit edge component must belong to a declared branch."
        )
    if component_names and set(edges_by_component) != component_names:
        raise StructuredOutputError(
            "Every declared electrical component must have exactly one circuit edge."
        )

    for branch_set in topology["parallel_branch_sets"]:
        if not isinstance(branch_set, list) or len(branch_set) < 2:
            raise StructuredOutputError("A parallel branch set needs at least two branches.")
        try:
            members = [branches[branch_id] for branch_id in branch_set]
        except KeyError as error:
            raise StructuredOutputError("A parallel set references an unknown branch.") from error
        junctions = {
            frozenset((branch["start_node"], branch["end_node"]))
            for branch in members
        }
        if len(junctions) != 1:
            raise StructuredOutputError(
                "Parallel branches must share the same two wire junctions."
            )


def _explicit_series_pairs(evidence_text: str) -> list[tuple[str, str]]:
    evidence_text = unicodedata.normalize("NFKC", evidence_text)
    pairs = []
    patterns = (
        r"series\s+(?:\w+[\s-]+){0,3}([A-Za-z][A-Za-z0-9_]*)\s+and\s+"
        r"(?:\w+[\s-]+){0,3}([A-Za-z][A-Za-z0-9_]*)",
        r"series[^()]{0,80}\(([A-Za-z][A-Za-z0-9_]*),\s*"
        r"([A-Za-z][A-Za-z0-9_]*)\)",
        r"([A-Za-z][A-Za-z0-9_]*)\s+and\s+"
        r"([A-Za-z][A-Za-z0-9_]*)\s+(?:form|forms|are|make|comprise)"
        r"[^.]{0,40}\bseries\b",
    )
    for pattern in patterns:
        pairs.extend(re.findall(pattern, evidence_text, re.IGNORECASE))
    return pairs


def _explicit_parallel_pairs(evidence_text: str) -> list[tuple[str, str]]:
    formula = unicodedata.normalize("NFKC", evidence_text)
    layout_pattern = re.compile(
        r"([A-Za-z][A-Za-z0-9_]*)\s*[⋅·]\s*"
        r"([A-Za-z][A-Za-z0-9_]*)[ \t]*\r?\n[ \t]*\1\s*\+\s*\2",
        re.IGNORECASE,
    )
    pairs = layout_pattern.findall(formula)
    formula = re.sub(r"\s+", "", formula)
    for multiplication_sign in ("⋅", "·"):
        formula = formula.replace(multiplication_sign, "*")
    pattern = re.compile(
        r"\(?([A-Za-z][A-Za-z0-9_]*)\*([A-Za-z][A-Za-z0-9_]*)\)?"
        r"/\(?\1\+\2\)?",
        re.IGNORECASE,
    )
    pairs.extend(pattern.findall(formula))
    return pairs


def ground_circuit_topology_from_evidence(value: dict, evidence_text: str) -> dict:
    """Rebuild an unambiguous topology from explicit series/equation evidence."""
    component_display = {
        _normal_name(
            component if isinstance(component, str) else component.get("name", "")
        ): component if isinstance(component, str) else component.get("name", "")
        for component in value.get("components", [])
    }
    series_pairs = {
        tuple(_normal_name(item) for item in pair)
        for pair in _explicit_series_pairs(evidence_text)
    }
    parallel_pairs = {
        tuple(_normal_name(item) for item in pair)
        for pair in _explicit_parallel_pairs(evidence_text)
    }
    candidates = []
    for series_pair in series_pairs:
        series_set = set(series_pair)
        if len(series_set) != 2 or not series_set.issubset(component_display):
            continue
        for parallel_pair in parallel_pairs:
            parallel_set = set(parallel_pair)
            shared = series_set.intersection(parallel_set)
            direct = parallel_set.difference(series_set)
            if (
                len(shared) == 1
                and len(direct) == 1
                and direct.issubset(component_display)
            ):
                candidates.append((series_pair, next(iter(direct))))
    if len(candidates) != 1:
        raise StructuredOutputError(
            "Circuit evidence does not establish one unambiguous series/parallel topology."
        )

    series_pair, direct_key = candidates[0]
    series_components = [component_display[key] for key in series_pair]
    direct_component = component_display[direct_key]
    topology = {
        "nodes": [
            {"id": "junction_start", "label": "shared start junction"},
            {"id": "junction_series", "label": "series junction"},
            {"id": "junction_end", "label": "shared end junction"},
        ],
        "edges": [
            {
                "from_node": "junction_start", "to_node": "junction_end",
                "component": direct_component,
            },
            {
                "from_node": "junction_start", "to_node": "junction_series",
                "component": series_components[0],
            },
            {
                "from_node": "junction_series", "to_node": "junction_end",
                "component": series_components[1],
            },
        ],
        "branches": [
            {
                "id": "direct_branch", "start_node": "junction_start",
                "end_node": "junction_end", "components": [direct_component],
            },
            {
                "id": "series_branch", "start_node": "junction_start",
                "end_node": "junction_end", "components": series_components,
            },
        ],
        "parallel_branch_sets": [["direct_branch", "series_branch"]],
    }
    corrected = dict(value)
    corrected["components"] = [
        {
            **component,
            "description": re.sub(
                r"\s*\([A-Z][A-Z0-9_-]{1,7}\)\s*", " ",
                str(component.get("description", "")),
            ).strip(),
        }
        if isinstance(component, dict) else component
        for component in value.get("components", [])
    ]
    corrected["circuit_topology"] = topology
    stable_references = set(component_display)
    stable_references.update(
        _normal_name(
            label if isinstance(label, str)
            else label.get("text", label.get("name", ""))
        )
        for label in value.get("labels", [])
    )
    corrected["spatial_relationships"] = [
        relationship for relationship in value.get("spatial_relationships", [])
        if isinstance(relationship, dict)
        and all(
            isinstance(relationship.get(field), str)
            and relationship[field].strip()
            for field in ("subject", "relationship", "object")
        )
    ]
    preserved_connections = [
        connection for connection in value.get("connections", [])
        if isinstance(connection, dict)
        and {
            _normal_name(connection.get("from", "")),
            _normal_name(connection.get("to", "")),
        }.issubset(stable_references)
    ]
    corrected["connections"] = [
        *preserved_connections,
        {
            "from": series_components[0], "to": series_components[1],
            "relationship": "series",
        },
        {
            "from": "direct_branch", "to": "series_branch",
            "relationship": "parallel",
        },
    ]
    return corrected


_VIEW_QUALIFIER = re.compile(
    r"\s*\((?:microscopic(?:\s+image)?|schematic(?:\s+diagram)?|"
    r"histological(?:\s+image)?|diagram|image|left|right)\)\s*$",
    re.IGNORECASE,
)


def _diagram_reference_display(value: dict) -> dict[str, str]:
    references: dict[str, str] = {}
    for label in value.get("labels", []):
        display = (
            label if isinstance(label, str)
            else label.get("text", label.get("name", ""))
            if isinstance(label, dict) else ""
        )
        if isinstance(display, str) and _normal_name(display):
            references.setdefault(_normal_name(display), display.strip())
    for component in value.get("components", []):
        display = (
            component if isinstance(component, str)
            else component.get("name", "")
            if isinstance(component, dict) else ""
        )
        if isinstance(display, str) and _normal_name(display):
            references.setdefault(_normal_name(display), display.strip())
    return references


def _known_endpoint_parts(endpoint: str, references: dict[str, str]) -> list[str]:
    """Split only an explicit conjunction whose every member is already known."""
    stripped = _VIEW_QUALIFIER.sub("", endpoint).strip()
    direct = references.get(_normal_name(stripped))
    if direct:
        return [direct]
    parts = [
        part.strip(" ,")
        for part in re.split(r"\s*,?\s+(?:and|&)\s+", stripped, flags=re.IGNORECASE)
    ]
    if len(parts) < 2 or any(not part for part in parts):
        return [stripped]
    normalized = [_normal_name(part) for part in parts]
    if len(set(normalized)) != len(normalized) or any(
        name not in references for name in normalized
    ):
        return [stripped]
    return [references[name] for name in normalized]


def normalize_composite_diagram_endpoints(value: dict) -> dict:
    """Expand clear known-label conjunctions without inventing diagram entities."""
    corrected = deepcopy(value)
    if str(corrected.get("diagram_kind", "")).strip().lower() != "other":
        return corrected
    references = _diagram_reference_display(corrected)
    for field, start_key, end_key in (
        ("spatial_relationships", "subject", "object"),
        ("connections", "from", "to"),
    ):
        expanded = []
        for relationship in corrected.get(field, []):
            if not isinstance(relationship, dict):
                expanded.append(relationship)
                continue
            start = relationship.get(start_key)
            end = relationship.get(end_key)
            if not isinstance(start, str) or not isinstance(end, str):
                expanded.append(relationship)
                continue
            starts = _known_endpoint_parts(start, references)
            ends = _known_endpoint_parts(end, references)
            for start_part in starts:
                for end_part in ends:
                    expanded.append({
                        **relationship,
                        start_key: start_part,
                        end_key: end_part,
                    })
        corrected[field] = expanded
    return corrected


def validate_labelled_diagram(value: dict, evidence_text: str = "") -> dict:
    required = {
        "diagram_kind", "labels", "components", "spatial_relationships",
        "connections", "circuit_topology", "explanation", "uncertain_items",
    }
    _require_keys(value, required, "Labelled diagram")
    if not isinstance(value["labels"], list) or not isinstance(value["components"], list):
        raise StructuredOutputError("Diagram labels and components must be lists.")
    if not isinstance(value["spatial_relationships"], list) or not isinstance(value["connections"], list):
        raise StructuredOutputError("Diagram relationships and connections must be lists.")
    if not isinstance(value["explanation"], str) or not isinstance(value["uncertain_items"], list):
        raise StructuredOutputError("Diagram explanation/uncertainty fields are invalid.")

    value = normalize_composite_diagram_endpoints(value)
    diagram_kind = str(value["diagram_kind"]).strip().lower()
    if diagram_kind not in {"circuit", "other"}:
        raise StructuredOutputError("diagram_kind must be circuit or other.")
    label_names = [
        _normal_name(
            label if isinstance(label, str)
            else label.get("text", label.get("name", ""))
        )
        for label in value["labels"]
    ]
    if any(not name for name in label_names):
        raise StructuredOutputError("Each diagram label needs text.")
    if len(label_names) != len(set(label_names)):
        raise StructuredOutputError("Diagram labels must not contain duplicates.")
    component_names = set()
    for component in value["components"]:
        if isinstance(component, str):
            if diagram_kind == "circuit" and re.search(
                r"\b(?:and|series|parallel|branch)\b", component, re.I
            ):
                raise StructuredOutputError(
                    "Circuit components must be separate component IDs."
                )
            component_names.add(_normal_name(component))
        elif isinstance(component, dict) and isinstance(component.get("name"), str):
            if diagram_kind == "circuit" and re.search(
                r"\b(?:and|series|parallel|branch)\b",
                component["name"],
                re.I,
            ):
                raise StructuredOutputError(
                    "Circuit components must be separate component IDs."
                )
            component_names.add(_normal_name(component["name"]))
        else:
            raise StructuredOutputError("Each diagram component needs a name.")

    if diagram_kind == "circuit":
        if not isinstance(value["circuit_topology"], dict):
            raise StructuredOutputError("Circuit diagrams require circuit_topology.")
        _validate_circuit_topology(
            value["circuit_topology"],
            component_names,
            set(label_names).difference(component_names),
        )
        for first, second in _explicit_series_pairs(evidence_text):
            expected = {_normal_name(first), _normal_name(second)}
            if not expected.issubset(component_names):
                continue
            matching = [
                branch
                for branch in value["circuit_topology"]["branches"]
                if {_normal_name(item) for item in branch["components"]} == expected
            ]
            if not matching:
                raise StructuredOutputError(
                    "Circuit topology contradicts an explicit series branch in caption/text."
                )
        branches = value["circuit_topology"]["branches"]
        parallel_sets = value["circuit_topology"]["parallel_branch_sets"]
        by_id = {branch["id"]: branch for branch in branches}
        for first, second in _explicit_parallel_pairs(evidence_text):
            first_key, second_key = _normal_name(first), _normal_name(second)
            if not {first_key, second_key}.issubset(component_names):
                continue
            supported = False
            for branch_set in parallel_sets:
                member_components = [
                    {_normal_name(item) for item in by_id[branch_id]["components"]}
                    for branch_id in branch_set
                ]
                if any(first_key in items for items in member_components) and any(
                    second_key in items for items in member_components
                ):
                    first_members = {index for index, items in enumerate(member_components) if first_key in items}
                    second_members = {index for index, items in enumerate(member_components) if second_key in items}
                    supported = bool(first_members - second_members and second_members - first_members)
                if supported:
                    break
            if not supported:
                raise StructuredOutputError(
                    "Circuit topology contradicts an explicit parallel impedance equation."
                )
    elif value["circuit_topology"] is not None:
        raise StructuredOutputError("Non-circuit diagrams must set circuit_topology to null.")

    valid_references = set(component_names)
    for label in value["labels"]:
        if isinstance(label, str):
            valid_references.add(_normal_name(label))
        elif isinstance(label, dict):
            label_text = label.get("text", label.get("name"))
            if isinstance(label_text, str):
                valid_references.add(_normal_name(label_text))
    if isinstance(value["circuit_topology"], dict):
        for node in value["circuit_topology"]["nodes"]:
            valid_references.add(_normal_name(node["id"]))
            if isinstance(node.get("label"), str):
                valid_references.add(_normal_name(node["label"]))
        for branch in value["circuit_topology"]["branches"]:
            valid_references.add(_normal_name(branch["id"]))
    for relationship in value["spatial_relationships"]:
        if not isinstance(relationship, dict):
            raise StructuredOutputError("Each spatial relationship must be an object.")
        _require_keys(
            relationship,
            {"subject", "relationship", "object"},
            "Spatial relationship",
        )
        if not all(
            isinstance(relationship[field], str) and relationship[field].strip()
            for field in ("subject", "relationship", "object")
        ):
            raise StructuredOutputError(
                "Spatial relationship fields must be non-empty strings."
            )
    grounded_evidence = _normal_name(evidence_text)
    for connection in value["connections"]:
        if not isinstance(connection, dict):
            raise StructuredOutputError("Each diagram connection must be an object.")
        _require_keys(connection, {"from", "to", "relationship"}, "Diagram connection")
        endpoints = {_normal_name(connection["from"]), _normal_name(connection["to"])}
        unknown = endpoints.difference(valid_references)
        if diagram_kind != "circuit":
            unknown = {
                endpoint for endpoint in unknown
                if not endpoint or endpoint not in grounded_evidence
            }
        if valid_references and unknown:
            raise StructuredOutputError(
                "A diagram connection references an unknown component, label, node or branch."
            )

    series_sentences = [
        sentence.lower()
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", evidence_text)
        if "series" in sentence.lower()
    ]
    relevant_series_evidence = any(
        sum(
            bool(re.search(re.escape(name), _normal_name(sentence)))
            for name in component_names
        ) >= 2
        for sentence in series_sentences
    )
    if relevant_series_evidence and value["connections"] and not any(
        "series" in str(connection.get("relationship", "")).lower()
        for connection in value["connections"]
    ):
        raise StructuredOutputError(
            "Caption/text states a component-specific series relationship "
            "missing from the diagram result."
        )
    return value


_SUPERSCRIPT_TRANSLATION = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺", "0123456789-+")


def _numeric_tick(label) -> float | None:
    text = str(label).strip().replace(",", "").replace("−", "-")
    if re.fullmatch(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)", text):
        return float(text)
    return None


def _power_of_ten_tick(label) -> int | None:
    text = str(label).strip().replace(" ", "").replace("×", "x")
    translated = text.translate(_SUPERSCRIPT_TRANSLATION)
    match = re.fullmatch(r"10(?:\^|\*\*)?([-+]?\d+)", translated)
    return int(match.group(1)) if match else None


def _scientific_multiplier_factor(multiplier) -> float | None:
    if multiplier is None or str(multiplier).strip().lower() in {"", "none", "null"}:
        return 1.0
    text = unicodedata.normalize("NFKC", str(multiplier))
    text = text.replace("×", "x").replace("Ã—", "x").replace(" ", "")
    match = re.fullmatch(r"[x*]?10(?:\^|\*\*)?([-+]?\d+)", text, re.IGNORECASE)
    return 10.0 ** int(match.group(1)) if match else None


def normalize_nyquist_axis_label(label: str) -> str:
    """Use conventional impedance notation without changing the variable."""
    text = unicodedata.normalize("NFKC", str(label or ""))
    text = re.sub(
        r"(?:-|\u2212|\u2013)\s*Img\s*\(\s*Z\s*\)",
        "-Im(Z)",
        text,
        flags=re.IGNORECASE,
    )
    return text


def _scientific_multiplier_exponent(axis: dict) -> int | None:
    factor = _scientific_multiplier_factor(axis.get("scientific_multiplier"))
    if factor is None or factor <= 0:
        return None
    exponent = round(math.log10(factor))
    return exponent if math.isclose(factor, 10.0 ** exponent) else None


def _visible_numeric_axis_range(axis: dict) -> tuple[float, float] | None:
    ticks = [_numeric_tick(label) for label in axis.get("tick_labels", [])]
    factor = _scientific_multiplier_factor(axis.get("scientific_multiplier"))
    if factor is None or len(ticks) < 2 or any(value is None for value in ticks):
        return None
    values = [float(value) * factor for value in ticks]
    return min(values), max(values)


def _validate_nyquist_panel(panel: dict) -> None:
    group = str(panel.get("group", "")).strip()
    if not group:
        raise StructuredOutputError("Each Nyquist panel needs a sample/group identity.")
    series_text = " ".join(map(str, panel.get("series", []))).casefold()
    fit_text = " ".join(
        map(str, [*panel.get("shape_features", []), *panel.get("visible_trends", [])])
    )
    model_and_data = "data" in series_text and "model" in series_text
    if model_and_data:
        if not re.search(
            r"\b(?:good|close|closely|near|follow|fit|overlaid)\b",
            fit_text,
            re.IGNORECASE,
        ):
            raise StructuredOutputError(
                "Each model-versus-data Nyquist panel needs a visible fit-quality description."
            )
        if not re.search(r"\bgood\b", fit_text, re.IGNORECASE):
            panel["visible_trends"].append(
                "The data and model show a good overall fit."
            )
            fit_text += " The data and model show a good overall fit."
    y_range = _visible_numeric_axis_range(panel["y_axis"])
    visible = panel["visible_range"]
    if y_range and isinstance(visible.get("min"), (int, float)) and isinstance(
        visible.get("max"), (int, float)
    ):
        expected_min, expected_max = y_range
        tolerance = max(1e-9, abs(expected_max - expected_min) * 0.08)
        if not (
            math.isclose(float(visible["min"]), expected_min, abs_tol=tolerance)
            and math.isclose(float(visible["max"]), expected_max, abs_tol=tolerance)
        ):
            raise StructuredOutputError(
                "Nyquist visible_range must apply the y-axis scientific multiplier."
            )


def infer_axis_scale(tick_labels: list, scientific_multiplier=None) -> str:
    """Infer linear/log from visible ticks; a multiplier does not affect scale."""
    if not isinstance(tick_labels, list) or len(tick_labels) < 3:
        return "unknown"
    exponents = [_power_of_ten_tick(label) for label in tick_labels]
    if all(value is not None for value in exponents):
        steps = [b - a for a, b in zip(exponents, exponents[1:])]
        return "log" if steps and max(steps) == min(steps) and steps[0] != 0 else "unknown"

    numbers = [_numeric_tick(label) for label in tick_labels]
    if any(value is None for value in numbers):
        return "unknown"
    differences = [b - a for a, b in zip(numbers, numbers[1:])]
    tolerance = max(1e-9, max(abs(value) for value in differences) * 0.03)
    if max(differences) - min(differences) <= tolerance:
        return "linear"
    if all(value > 0 for value in numbers):
        ratios = [b / a for a, b in zip(numbers, numbers[1:])]
        ratio_tolerance = max(1e-9, max(abs(value) for value in ratios) * 0.03)
        if max(ratios) - min(ratios) <= ratio_tolerance and ratios[0] > 1:
            return "log"
    return "unknown"


def _validate_axis(axis, panel_name: str, axis_name: str) -> None:
    if not isinstance(axis, dict):
        raise StructuredOutputError(f"Panel {panel_name} {axis_name}-axis must be an object.")
    _require_keys(
        axis,
        {"label", "unit", "scale", "tick_labels", "scientific_multiplier"},
        f"Panel {panel_name} {axis_name}-axis",
    )
    if not isinstance(axis["label"], str) or not axis["label"].strip():
        raise StructuredOutputError(f"Panel {panel_name} {axis_name}-axis label is missing.")
    if not isinstance(axis["tick_labels"], list):
        raise StructuredOutputError(f"Panel {panel_name} {axis_name}-axis ticks must be a list.")
    declared = str(axis["scale"]).strip().lower()
    if declared == "logarithmic":
        declared = "log"
    if declared not in {"linear", "log", "unknown"}:
        raise StructuredOutputError(f"Panel {panel_name} {axis_name}-axis scale is invalid.")
    inferred = infer_axis_scale(axis["tick_labels"], axis["scientific_multiplier"])
    if inferred != "unknown" and declared != inferred:
        raise StructuredOutputError(
            f"Panel {panel_name} {axis_name}-axis is {inferred} from its ticks, "
            f"not {declared}; scientific notation does not imply log scale."
        )


def _range_score(panel: dict) -> float | None:
    visible_range = panel.get("visible_range")
    if not isinstance(visible_range, dict):
        raise StructuredOutputError("A graph panel visible_range must be an object.")
    _require_keys(
        visible_range,
        {"min", "max", "unit", "confidence"},
        "Graph visible range",
    )
    minimum, maximum = visible_range.get("min"), visible_range.get("max")
    if not isinstance(minimum, (int, float)) or not isinstance(maximum, (int, float)):
        return None
    if maximum < minimum:
        raise StructuredOutputError("A graph visible range has max below min.")
    return (float(minimum) + float(maximum)) / 2


def _explicit_text_subjects(evidence_text: str, relation: str, metric: str) -> set[str]:
    if not evidence_text or relation not in {"highest", "lowest", "most_complex"}:
        return set()
    relation_words = {
        "highest": r"highest|largest|greatest",
        "lowest": r"lowest|smallest",
        "most_complex": r"most\s+complex",
    }[relation]
    metric_word = "complex" if relation == "most_complex" else re.escape(metric)
    subjects = set()
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", evidence_text):
        if not re.search(metric_word, sentence, re.IGNORECASE):
            continue
        if not re.search(relation_words, sentence, re.IGNORECASE):
            continue
        subject_pattern = r"\b(?:group|sample)\s+[A-Za-z0-9]+\b"
        if relation == "most_complex":
            forward = re.findall(
                rf"({subject_pattern})[^.;:]{{0,60}}(?:{relation_words})",
                sentence,
                re.IGNORECASE,
            )
            reverse = re.findall(
                rf"(?:{relation_words})[^.;:]{{0,30}}({subject_pattern})",
                sentence,
                re.IGNORECASE,
            )
        else:
            forward = re.findall(
                rf"({subject_pattern})[^.;:]{{0,60}}(?:{relation_words})[^.;:]{{0,40}}{metric_word}",
                sentence,
                re.IGNORECASE,
            )
            reverse = re.findall(
                rf"{metric_word}[^.;:]{{0,40}}(?:{relation_words})[^.;:]{{0,30}}({subject_pattern})",
                sentence,
                re.IGNORECASE,
            )
        subjects.update(match.lower() for match in [*forward, *reverse])
    return subjects


def _validate_comparison(
    comparison: dict,
    panels: list[dict],
    evidence_text: str = "",
) -> None:
    if not isinstance(comparison, dict):
        raise StructuredOutputError("Each graph comparison must be an object.")
    _require_keys(
        comparison,
        {
            "claim", "subject", "relation", "metric", "evidence", "confidence",
            "uncertain", "evidence_conflict",
        },
        "Graph comparison",
    )
    if not isinstance(comparison["claim"], str) or not isinstance(comparison["evidence"], list):
        raise StructuredOutputError("Graph comparison claim/evidence fields are invalid.")
    if not set(comparison["evidence"]).issubset({"vision", "text"}):
        raise StructuredOutputError("Graph comparison evidence must be vision and/or text.")
    if comparison["evidence_conflict"] and not comparison["uncertain"]:
        raise StructuredOutputError("Conflicting graph evidence must be marked uncertain.")

    relation = str(comparison["relation"]).lower()
    metric = str(comparison["metric"]).lower()
    subject = str(comparison["subject"]).strip().lower()
    panel_subjects = {
        str(panel.get("group", "")).strip().lower()
        for panel in panels if panel.get("group")
    }
    if (
        re.search(r"\b(?:fit|deviation|mismatch)\b", metric, re.IGNORECASE)
        and subject not in panel_subjects
        and not comparison["uncertain"]
    ):
        raise StructuredOutputError(
            "Fit/deviation comparisons must name a visible panel sample or group."
        )
    text_subjects = _explicit_text_subjects(evidence_text, relation, metric)
    if text_subjects:
        if subject in text_subjects:
            if "text" not in comparison["evidence"]:
                comparison["evidence"].append("text")
        elif not comparison["uncertain"] or not comparison["evidence_conflict"]:
            raise StructuredOutputError(
                "Graph comparison conflicts with an explicit caption/body statement "
                "and must be marked uncertain."
            )
    candidates = []
    if relation in {"highest", "lowest"} and "magnitude" in metric:
        for panel in panels:
            if "magnitude" not in str(panel.get("graph_kind", "")).lower():
                continue
            score = _range_score(panel)
            if score is not None and panel.get("group"):
                candidates.append((str(panel["group"]).strip().lower(), score))
        if len(candidates) >= 2 and not comparison["uncertain"]:
            expected = (max if relation == "highest" else min)(candidates, key=lambda row: row[1])[0]
            if subject != expected:
                raise StructuredOutputError(
                    "A graph magnitude comparison contradicts the independently estimated ranges."
                )
    if relation == "most_complex":
        candidates = [
            (str(panel.get("group", "")).strip().lower(), panel.get("complexity_score"))
            for panel in panels
            if isinstance(panel.get("complexity_score"), (int, float)) and panel.get("group")
        ]
        if len(candidates) >= 2 and not comparison["uncertain"]:
            expected = max(candidates, key=lambda row: row[1])[0]
            if subject != expected:
                raise StructuredOutputError(
                    "A graph complexity comparison contradicts panel shape estimates."
                )


def _comparison_claim_key(comparison: dict) -> str:
    claim = unicodedata.normalize("NFKC", str(comparison.get("claim", "")))
    return re.sub(r"[^a-z0-9]+", " ", claim.casefold()).strip()


def _deduplicate_graph_comparisons(comparisons: list[dict]) -> list[dict]:
    deduplicated = []
    by_claim = {}
    for comparison in comparisons:
        key = _comparison_claim_key(comparison)
        if not key or key not in by_claim:
            deduplicated.append(comparison)
            if key:
                by_claim[key] = comparison
            continue
        existing = by_claim[key]
        existing["evidence"] = list(dict.fromkeys([
            *existing.get("evidence", []), *comparison.get("evidence", []),
        ]))
        if isinstance(comparison.get("confidence"), (int, float)):
            existing["confidence"] = max(
                float(existing.get("confidence", 0.0)),
                float(comparison["confidence"]),
            )
        existing["uncertain"] = bool(
            existing.get("uncertain") or comparison.get("uncertain")
        )
        existing["evidence_conflict"] = bool(
            existing.get("evidence_conflict") or comparison.get("evidence_conflict")
        )
    return deduplicated


def _add_nyquist_scale_comparison(value: dict) -> None:
    scale_rows = []
    for panel in value["panels"]:
        if str(panel.get("graph_kind", "")).casefold() != "nyquist":
            continue
        x_exponent = _scientific_multiplier_exponent(panel["x_axis"])
        y_exponent = _scientific_multiplier_exponent(panel["y_axis"])
        group = str(panel.get("group", "")).strip()
        if not group or x_exponent is None or x_exponent != y_exponent:
            return
        scale_rows.append((group, x_exponent))
    if len(scale_rows) < 2:
        return
    largest_exponent = max(exponent for _, exponent in scale_rows)
    largest = [row for row in scale_rows if row[1] == largest_exponent]
    lower = [row for row in scale_rows if row[1] < largest_exponent]
    if len(largest) != 1 or not lower:
        return

    largest_group = largest[0][0]
    lower_phrases = [
        f"{group} uses approximately \u00d710^{exponent}"
        for group, exponent in lower
    ]
    if len(lower_phrases) == 1:
        lower_text = lower_phrases[0]
    else:
        lower_text = ", ".join(lower_phrases[:-1]) + f", and {lower_phrases[-1]}"
    comparison = {
        "claim": (
            f"{largest_group} has the larger overall impedance scale because its "
            f"Re(Z) and -Im(Z) axes use approximately \u00d710^{largest_exponent}, "
            f"while {lower_text}."
        ),
        "subject": largest_group,
        "relation": "highest",
        "metric": "overall impedance scale",
        "evidence": ["vision"],
        "confidence": 0.95,
        "uncertain": False,
        "evidence_conflict": False,
    }
    value["comparisons"] = [
        item for item in value["comparisons"]
        if "overallimpedancescale" not in _normal_name(
            f"{item.get('metric', '')} {item.get('claim', '')}"
        )
    ]
    value["comparisons"].append(comparison)


def validate_graph(value: dict, evidence_text: str = "") -> dict:
    _require_keys(
        value,
        {
            "figure_number", "panels", "comparisons",
            "frequency_direction_evidence", "uncertain_values",
        },
        "Graph",
    )
    if not isinstance(value["panels"], list) or not value["panels"]:
        raise StructuredOutputError("Graph must contain at least one panel.")
    for panel in value["panels"]:
        if not isinstance(panel, dict):
            raise StructuredOutputError("Each graph panel must be an object.")
        _require_keys(
            panel,
            {
                "panel", "graph_kind", "group", "x_axis", "y_axis", "series",
                "visible_range", "shape_features", "complexity_score", "visible_trends",
            },
            "Graph panel",
        )
        if not panel.get("group"):
            combined_identity = re.fullmatch(
                r"\s*\(?([A-Za-z0-9]+)\)?\s+(.+?)\s*",
                str(panel.get("panel", "")),
            )
            if combined_identity:
                panel["panel"] = combined_identity.group(1)
                panel["group"] = combined_identity.group(2)
        x_axis, y_axis = panel.get("x_axis"), panel.get("y_axis")
        looks_nyquist = str(panel.get("graph_kind", "")).casefold() == "nyquist"
        if isinstance(x_axis, dict):
            looks_nyquist = looks_nyquist or (
                "re(z)" in str(x_axis.get("label", "")).casefold().replace(" ", "")
            )
        if looks_nyquist and isinstance(y_axis, dict):
            y_axis["label"] = normalize_nyquist_axis_label(y_axis.get("label", ""))
            for field in ("shape_features", "visible_trends"):
                if isinstance(panel.get(field), list):
                    panel[field] = [
                        normalize_nyquist_axis_label(item)
                        if isinstance(item, str) else item
                        for item in panel[field]
                    ]
        _validate_axis(panel["x_axis"], str(panel["panel"]), "x")
        _validate_axis(panel["y_axis"], str(panel["panel"]), "y")
        if not all(
            isinstance(panel[key], list)
            for key in ("series", "shape_features", "visible_trends")
        ):
            raise StructuredOutputError("Graph series, shapes and trends must be lists.")
        complexity = panel["complexity_score"]
        if not isinstance(complexity, (int, float)) or isinstance(complexity, bool):
            raise StructuredOutputError("Graph complexity must be numeric.")
        if not math.isfinite(float(complexity)) or not 0 <= complexity <= 1:
            raise StructuredOutputError("Graph complexity must be between 0 and 1.")
        _range_score(panel)
    if not isinstance(value["comparisons"], list) or not isinstance(value["uncertain_values"], list):
        raise StructuredOutputError("Graph comparison/uncertainty fields are invalid.")
    if not isinstance(value["frequency_direction_evidence"], list):
        raise StructuredOutputError("Graph frequency direction evidence must be a list.")
    for comparison in value["comparisons"]:
        _validate_comparison(comparison, value["panels"], evidence_text)
    value["comparisons"] = _deduplicate_graph_comparisons(value["comparisons"])

    is_nyquist = any(
        str(panel.get("graph_kind", "")).lower() == "nyquist"
        or "re(z)" in str(panel.get("x_axis", {}).get("label", "")).lower().replace(" ", "")
        for panel in value["panels"]
    )
    if is_nyquist:
        for comparison in value["comparisons"]:
            for field in ("claim", "metric"):
                if isinstance(comparison.get(field), str):
                    comparison[field] = normalize_nyquist_axis_label(
                        comparison[field]
                    )
        panel_ids = [_normal_name(panel.get("panel", "")) for panel in value["panels"]]
        panel_groups = [
            _normal_name(panel.get("group", "")) for panel in value["panels"]
        ]
        if len(panel_ids) != len(set(panel_ids)) or len(panel_groups) != len(set(panel_groups)):
            raise StructuredOutputError(
                "Nyquist panel identifiers and sample/group identities must be unique."
            )
        for panel in value["panels"]:
            _validate_nyquist_panel(panel)
        _add_nyquist_scale_comparison(value)
        for comparison in value["comparisons"]:
            _validate_comparison(comparison, value["panels"], evidence_text)
        value["comparisons"] = _deduplicate_graph_comparisons(value["comparisons"])
        deviation_subjects = {
            str(comparison.get("subject", "")).strip().lower()
            for comparison in value["comparisons"]
            if re.search(
                r"\b(?:deviation|mismatch)\b",
                f"{comparison.get('metric', '')} {comparison.get('claim', '')}",
                re.IGNORECASE,
            )
        }
        for panel in value["panels"]:
            fit_text = " ".join(
                map(str, [*panel["shape_features"], *panel["visible_trends"]])
            )
            has_located_deviation = bool(re.search(
                r"\b(?:deviation|mismatch|separation)\w*\b[^.]{0,90}"
                r"\b(?:left|right|low|high)\b|"
                r"\b(?:left|right|low|high)\b[^.]{0,90}"
                r"\b(?:deviation|mismatch|separation)\w*\b",
                fit_text,
                re.IGNORECASE,
            ))
            if (
                str(panel.get("group", "")).strip().lower() not in deviation_subjects
                and not has_located_deviation
            ):
                continue
            if re.search(
                r"\b(?:excellent|perfect(?:ly)?|superimposed|indistinguishable)\b"
                r"[^.]{0,70}\b(?:entire|whole|throughout)\b[^.]{0,30}"
                r"\b(?:arc|plot|range)\b",
                fit_text,
                re.IGNORECASE,
            ):
                raise StructuredOutputError(
                    "Nyquist fit language contradicts a reported visible deviation."
                )
        if value["frequency_direction_evidence"] and not all(
            re.search(
                r"\b(?:arrow|frequency\s+label|caption|explicit\s+text)\b",
                str(item),
                re.IGNORECASE,
            )
            for item in value["frequency_direction_evidence"]
        ):
            raise StructuredOutputError(
                "Nyquist frequency direction evidence needs arrows, labels or explicit text."
            )
    if is_nyquist and not value["frequency_direction_evidence"]:
        prose = " ".join(
            [trend for panel in value["panels"] for trend in panel["visible_trends"]]
            + [comparison["claim"] for comparison in value["comparisons"]]
        )
        if re.search(
            r"\b(?:higher?|lower?)[ -]frequenc|\bfrequency\s+(?:increases|decreases)",
            prose,
            re.IGNORECASE,
        ):
            raise StructuredOutputError(
                "Nyquist frequency direction was inferred without arrows, labels or grounded text."
            )
    return value


def validate_table(value: dict) -> dict:
    required = {"table_number", "title", "columns", "rows", "units", "comparisons", "unreadable_cells"}
    _require_keys(value, required, "Table")
    if not isinstance(value["columns"], list) or not value["columns"]:
        raise StructuredOutputError("Table columns must be a non-empty list.")
    if len({str(column).strip().lower() for column in value["columns"]}) != len(value["columns"]):
        raise StructuredOutputError("Table contains duplicate column names.")
    if not isinstance(value["rows"], list):
        raise StructuredOutputError("Table rows must be a list.")
    for index, row in enumerate(value["rows"]):
        if not isinstance(row, list) or len(row) != len(value["columns"]):
            raise StructuredOutputError(
                f"Table row {index} has {len(row) if isinstance(row, list) else 'invalid'} "
                f"cells; expected {len(value['columns'])}."
            )
    if not isinstance(value["comparisons"], list) or not isinstance(value["unreadable_cells"], list):
        raise StructuredOutputError("Table comparison/unreadable fields are invalid.")
    return value


def validate_typed_response(visual_type: str, value: dict, evidence_text: str = "") -> dict:
    if visual_type == "labelled_diagram":
        return validate_labelled_diagram(value, evidence_text)
    if visual_type == "graph":
        return validate_graph(value, evidence_text)
    if visual_type == "table":
        return validate_table(value)
    raise StructuredOutputError(f"Unsupported visual type: {visual_type}")


def _response_text(response) -> str:
    message = response.get("message", {})
    return (message.get("content") or message.get("thinking") or "").strip()


def _chat_with_runner_retry(**kwargs):
    """Retry one identical request when Ollama's local model runner crashes."""
    try:
        return ollama.chat(**kwargs)
    except Exception as error:
        if "model runner has unexpectedly stopped" not in str(error).casefold():
            raise
        time.sleep(1)
        return ollama.chat(**kwargs)


def _call_model(
    image_path: Path,
    prompt: str,
    num_predict: int = STRUCTURED_NUM_PREDICT,
) -> str:
    arguments = {
        "model": VISION_MODEL,
        "messages": [{
            "role": "user", "content": prompt, "images": [str(image_path)],
        }],
        "format": "json",
        "options": {
            "temperature": 0, "num_ctx": 8192, "num_predict": num_predict,
        },
    }
    for _ in range(2):
        raw = _response_text(_chat_with_runner_retry(**arguments))
        if raw:
            return raw
    raise StructuredOutputError("Vision model returned an empty response twice.")


def _nyquist_fit_verification_prompt(panel: dict) -> str:
    return f"""
Analyse only this single Nyquist panel, identified as panel
{panel.get('panel')} ({panel.get('group')}). Ignore any legend sample marker.
Compare the experimental data markers with the fitted model curve. Estimate
the largest visible vertical model-data separation separately in the left and
right halves of the plotted x range. Express each as a fraction of the full
displayed y-axis span, so 0.05 means five percent of that panel's y range.
Compare blue data-marker centres against the orange dashed curve, not marker
arms. Inspect the descending high-Re(Z) endpoint. Left means low Re(Z); right
means high Re(Z). Do not round a visible nonzero separation down to zero.
Do not compare this panel with another image and do not use its scientific
multiplier or raw impedance magnitude as a fit-quality proxy.

Return JSON only:
{{
  "panel": "{panel.get('panel')}",
  "group": "{panel.get('group')}",
  "left_normalized_largest_deviation": 0.0,
  "right_normalized_largest_deviation": 0.0,
  "confidence": 0.0
}}
"""


def _cluster_nearby_pixels(
    points: set[tuple[int, int]], radius: int = 2
) -> list[list[tuple[int, int]]]:
    remaining = set(points)
    components = []
    while remaining:
        start = remaining.pop()
        component = [start]
        pending = [start]
        while pending:
            x, y = pending.pop()
            for near_x in range(x - radius, x + radius + 1):
                for near_y in range(y - radius, y + radius + 1):
                    neighbour = (near_x, near_y)
                    if neighbour in remaining:
                        remaining.remove(neighbour)
                        component.append(neighbour)
                        pending.append(neighbour)
        components.append(component)
    return components


def _colored_nyquist_fit_reading(
    image_path: Path, panel: dict
) -> dict | None:
    """Measure blue-marker/orange-curve separation when those pixels are visible."""
    try:
        pixmap = fitz.Pixmap(str(image_path))
    except Exception:
        return None
    if pixmap.n < 3 or pixmap.width < 80 or pixmap.height < 80:
        return None
    width, height, channels = pixmap.width, pixmap.height, pixmap.n
    samples = pixmap.samples
    blue_points: set[tuple[int, int]] = set()
    orange_by_x: dict[int, list[int]] = {}
    for y in range(height):
        row_offset = y * width * channels
        for x in range(width):
            offset = row_offset + x * channels
            red, green, blue = samples[offset:offset + 3]
            if (
                blue > 130 and green > 60
                and blue > red + 20 and blue > green + 15
            ):
                blue_points.add((x, y))
            if (
                red > 160 and red > green + 35
                and green > 30 and blue < 180
            ):
                orange_by_x.setdefault(x, []).append(y)
    if len(blue_points) < 80 or sum(map(len, orange_by_x.values())) < 80:
        return None

    centres = []
    for component in _cluster_nearby_pixels(blue_points):
        if not 1 <= len(component) <= 160:
            continue
        x = sum(point[0] for point in component) / len(component)
        y = sum(point[1] for point in component) / len(component)
        if not (
            width * 0.06 < x < width * 0.95
            and height * 0.06 < y < height * 0.86
        ):
            continue
        # Scientific plots commonly place the legend in the upper-right.
        # Exclude only that compact corner; the descending curve remains below it.
        if x > width * 0.72 and y < height * 0.22:
            continue
        nearest_squared = None
        for orange_x in range(max(0, int(x) - 24), min(width, int(x) + 25)):
            for orange_y in orange_by_x.get(orange_x, []):
                distance_squared = (orange_x - x) ** 2 + (orange_y - y) ** 2
                if nearest_squared is None or distance_squared < nearest_squared:
                    nearest_squared = distance_squared
        if nearest_squared is not None and nearest_squared <= 24 ** 2:
            centres.append((x, math.sqrt(nearest_squared)))
    if len(centres) < 12:
        return None
    x_values = [item[0] for item in centres]
    midpoint = (min(x_values) + max(x_values)) / 2
    left = [distance for x, distance in centres if x < midpoint]
    right = [distance for x, distance in centres if x >= midpoint]
    if len(left) < 4 or len(right) < 4:
        return None
    left_score = sum(left) / len(left) / height
    right_score = sum(right) / len(right) / height
    return {
        "panel": panel.get("panel"),
        "group": panel.get("group"),
        "left_normalized_largest_deviation": left_score,
        "right_normalized_largest_deviation": right_score,
        "confidence": min(0.95, 0.70 + len(centres) / 200),
    }


def _validate_nyquist_fit_reading(
    value: dict, expected_panel: dict
) -> dict:
    _require_keys(
        value,
        {
            "panel", "group", "left_normalized_largest_deviation",
            "right_normalized_largest_deviation", "confidence",
        },
        "Nyquist fit verification",
    )
    if _normal_name(value["panel"]) != _normal_name(expected_panel["panel"]):
        raise StructuredOutputError("Nyquist fit verification returned the wrong panel.")
    if _normal_name(value["group"]) != _normal_name(expected_panel["group"]):
        raise StructuredOutputError("Nyquist fit verification returned the wrong group.")
    left_score = value["left_normalized_largest_deviation"]
    right_score = value["right_normalized_largest_deviation"]
    confidence = value["confidence"]
    for score in (left_score, right_score):
        if (
            not isinstance(score, (int, float))
            or isinstance(score, bool)
            or not math.isfinite(float(score))
            or not 0 <= score <= 1
        ):
            raise StructuredOutputError(
                "Nyquist normalized deviations must be numbers between 0 and 1."
            )
    if (
        not isinstance(confidence, (int, float))
        or isinstance(confidence, bool)
        or not math.isfinite(float(confidence))
        or not 0 <= confidence <= 1
    ):
        raise StructuredOutputError(
            "Nyquist fit verification confidence must be between 0 and 1."
        )
    left_score, right_score = float(left_score), float(right_score)
    largest = max(left_score, right_score)
    gap = abs(left_score - right_score)
    value["normalized_largest_deviation"] = largest
    value["largest_deviation_side"] = (
        "uncertain"
        if gap < 0.002 or gap / max(largest, 1e-9) < 0.10
        else "left" if left_score > right_score else "right"
    )
    return value


def _replace_nyquist_fit_comparison(
    value: dict,
    readings: list[dict],
    *,
    force_uncertain: bool = False,
) -> None:
    panels_by_id = {
        _normal_name(panel.get("panel", "")): panel for panel in value["panels"]
    }
    ordered = [
        (panels_by_id[_normal_name(reading["panel"])], reading)
        for reading in readings
        if _normal_name(reading["panel"]) in panels_by_id
    ]
    if len(ordered) < 2:
        force_uncertain = True
        ordered = [(panel, {}) for panel in value["panels"][:2]]
    groups = [str(panel.get("group", "")).strip() for panel, _ in ordered]
    for panel, reading in ordered:
        side = reading.get("largest_deviation_side")
        if side not in {"left", "right"}:
            continue
        for field in ("shape_features", "visible_trends"):
            panel[field] = [
                text for text in panel[field]
                if not re.search(
                    r"\b(?:excellent|perfect(?:ly)?|superimposed|indistinguishable)\b"
                    r"[^.]{0,70}\b(?:entire|whole|throughout)\b[^.]{0,30}"
                    r"\b(?:arc|plot|range)\b",
                    str(text),
                    re.IGNORECASE,
                )
            ]
        location = "high Re(Z)" if side == "right" else "low Re(Z)"
        deviation_trend = (
            f"The largest visible model-data deviation is on the {side} side "
            f"at {location}."
        )
        if _normal_name(deviation_trend) not in {
            _normal_name(trend) for trend in panel["visible_trends"]
        }:
            panel["visible_trends"].append(deviation_trend)
    comparisons = [
        item for item in value["comparisons"]
        if not re.search(
            r"\b(?:closer|better)\b.{0,30}\bfit\b|\bfit\b.{0,30}\b(?:closer|better)\b",
            f"{item.get('claim', '')} {item.get('metric', '')}",
            re.IGNORECASE,
        )
    ]
    scores = [
        float(reading["normalized_largest_deviation"])
        for _, reading in ordered if "normalized_largest_deviation" in reading
    ]
    confidences = [
        float(reading["confidence"])
        for _, reading in ordered if "confidence" in reading
    ]
    if len(scores) >= 2:
        low, next_low = sorted(scores)[:2]
        gap = next_low - low
        relative_gap = gap / max(next_low, 1e-9)
    else:
        gap = relative_gap = 0.0
    uncertain = (
        force_uncertain
        or len(scores) < 2
        or len(confidences) < 2
        or min(confidences) < 0.65
        or gap < 0.02
        or relative_gap < 0.10
    )
    if uncertain:
        claim = (
            "The relative visual fit is too close to distinguish reliably between "
            + " and ".join(groups)
            + "."
        )
        comparison = {
            "claim": claim,
            "subject": groups[0],
            "relation": "other",
            "metric": "normalized largest model-data deviation",
            "evidence": ["vision"],
            "confidence": min(confidences) if confidences else 0.0,
            "uncertain": True,
            "evidence_conflict": False,
        }
        note = "Relative Nyquist fit closeness could not be distinguished reliably."
        if note not in value["uncertain_values"]:
            value["uncertain_values"].append(note)
    else:
        best_index = min(
            range(len(ordered)),
            key=lambda index: float(
                ordered[index][1]["normalized_largest_deviation"]
            ),
        )
        best_group = groups[best_index]
        other_groups = [group for index, group in enumerate(groups) if index != best_index]
        comparison = {
            "claim": (
                f"{best_group} appears to have a slightly closer visual fit than "
                + " and ".join(other_groups)
                + "."
            ),
            "subject": best_group,
            "relation": "lowest",
            "metric": "normalized largest model-data deviation",
            "evidence": ["vision"],
            "confidence": min(confidences),
            "uncertain": False,
            "evidence_conflict": False,
        }
    value["comparisons"] = [*comparisons, comparison]


def verify_nyquist_fit_comparison(
    value: dict,
    panel_images: list[tuple[str, Path]],
    evidence_text: str = "",
) -> tuple[dict, list[str], str]:
    """Ground a two-panel fit comparison in independent tight-crop readings."""
    panels_by_id = {
        _normal_name(panel.get("panel", "")): panel
        for panel in value.get("panels", [])
        if str(panel.get("graph_kind", "")).casefold() == "nyquist"
    }
    if len(panels_by_id) != 2 or len(panel_images) != 2:
        return value, [], "not_applicable"
    raw_responses = []
    readings = []
    verification_error = ""
    for panel_id, image_path in panel_images:
        panel = panels_by_id.get(_normal_name(panel_id))
        if panel is None and len(readings) < len(panels_by_id):
            panel = sorted(
                panels_by_id.values(),
                key=lambda item: _normal_name(item.get("panel", "")),
            )[len(readings)]
        if panel is None:
            verification_error = "tight crop did not match a validated panel"
            break
        pixel_reading = _colored_nyquist_fit_reading(image_path, panel)
        raw = (
            json.dumps(pixel_reading)
            if pixel_reading is not None
            else _call_model(
                image_path,
                _nyquist_fit_verification_prompt(panel),
                num_predict=NYQUIST_FIT_NUM_PREDICT,
            )
        )
        raw_responses.append(raw)
        try:
            readings.append(_validate_nyquist_fit_reading(
                parse_json_response(raw), panel
            ))
        except StructuredOutputError as error:
            verification_error = str(error)
            break
    _replace_nyquist_fit_comparison(
        value,
        readings,
        force_uncertain=bool(verification_error),
    )
    validate_graph(value, evidence_text)
    return value, raw_responses, verification_error or "verified"


def _call_model_images(
    image_paths: list[Path],
    prompt: str,
    num_predict: int,
) -> str:
    arguments = {
        "model": VISION_MODEL,
        "messages": [{
            "role": "user",
            "content": prompt,
            "images": [str(path) for path in image_paths],
        }],
        "format": "json",
        "options": {
            "temperature": 0, "num_ctx": 8192, "num_predict": num_predict,
        },
    }
    for _ in range(2):
        raw = _response_text(_chat_with_runner_retry(**arguments))
        if raw:
            return raw
    raise StructuredOutputError("Vision model returned an empty response twice.")


def _compact_panel_prompt(
    question: str,
    panel_ids: list[str],
    evidence_text: str,
    strict: bool = False,
) -> str:
    retry = (
        "The prior response was truncated or invalid. Keep every string under 20 words."
        if strict else ""
    )
    return f"""
Analyse only graph panels {', '.join(panel_ids)} in this crop.
Question: {question}
{retry}

Only complexity_score is bounded: use 0.0 (simple/smooth) to 1.0 (most
complex). approximate_curve_level is not bounded; report the typical numerical
y-axis level so magnitude panels can be compared. Estimate the visible minimum
and maximum across all series, then use their midpoint; never use only the
rightmost/final value.

Return compact JSON only. Do not include tick arrays, axis limits, explanations
or Markdown. approximate_curve_level is the typical visible curve level, not
the full y-axis limits. Preserve exact readable variable labels. Determine log
scale from tick progression, not from a scientific-notation multiplier.

Schema:
{{
  "panels": [
    {{
      "panel": "...", "group": "...", "graph_kind": "magnitude|phase|other",
      "x_axis": {{"label": "...", "unit": "...", "scale": "linear|log|unknown"}},
      "y_axis": {{"label": "...", "unit": "...", "scale": "linear|log|unknown"}},
      "series": ["..."], "visible_trend": "...",
      "approximate_curve_level": 0.0, "complexity_score": 0.0
    }}
  ],
  "uncertain_values": ["..."]
}}

Caption/text cross-check:
{evidence_text[:2500]}
"""


def detect_compact_graph_panel_ids(image_path: Path) -> tuple[list[str], str]:
    prompt = """
Inspect only the panel markers in this graph figure. Return compact JSON only:
{"panel_ids": ["a", "b"]}
List visible lettered panels once, in reading order. Do not describe the plots.
"""
    raw = _call_model(image_path, prompt, 220)
    try:
        value = parse_json_response(raw)
        panel_ids = value.get("panel_ids", [])
        if not isinstance(panel_ids, list):
            raise StructuredOutputError("Panel IDs must be a list.")
        normalized = [str(panel).strip().lower() for panel in panel_ids]
        if (
            len(normalized) >= 4
            and len(normalized) % 2 == 0
            and len(normalized) == len(set(normalized))
            and all(re.fullmatch(r"[a-z]", panel) for panel in normalized)
        ):
            return normalized, raw
    except StructuredOutputError:
        pass
    retry = _call_model(
        image_path,
        prompt + "\nThe previous output was invalid. Return only the panel_ids object.",
        220,
    )
    value = parse_json_response(retry)
    panel_ids = value.get("panel_ids", [])
    normalized = [str(panel).strip().lower() for panel in panel_ids] if isinstance(panel_ids, list) else []
    if (
        len(normalized) < 4
        or len(normalized) % 2
        or len(normalized) != len(set(normalized))
        or not all(re.fullmatch(r"[a-z]", panel) for panel in normalized)
    ):
        raise StructuredOutputError("Could not verify multi-panel graph IDs.")
    return normalized, retry


def _validate_compact_axis(axis: dict, panel: str, name: str) -> None:
    if not isinstance(axis, dict):
        raise StructuredOutputError(f"Compact panel {panel} {name}-axis is invalid.")
    _require_keys(axis, {"label", "unit", "scale"}, f"Compact panel {panel} {name}-axis")
    if not isinstance(axis["label"], str) or not axis["label"].strip():
        raise StructuredOutputError(f"Compact panel {panel} {name}-axis label is missing.")
    if str(axis["scale"]).lower() not in {"linear", "log", "unknown"}:
        raise StructuredOutputError(f"Compact panel {panel} {name}-axis scale is invalid.")


def validate_compact_panel_response(value: dict, expected_panels: list[str]) -> dict:
    _require_keys(value, {"panels", "uncertain_values"}, "Compact graph response")
    if not isinstance(value["panels"], list) or not isinstance(value["uncertain_values"], list):
        raise StructuredOutputError("Compact graph panels/uncertainty must be lists.")
    observed = []
    for panel in value["panels"]:
        if not isinstance(panel, dict):
            raise StructuredOutputError("Each compact graph panel must be an object.")
        _require_keys(
            panel,
            {
                "panel", "group", "graph_kind", "x_axis", "y_axis", "series",
                "visible_trend", "approximate_curve_level", "complexity_score",
            },
            "Compact graph panel",
        )
        panel_id = re.sub(r"[^a-z0-9]+", "", str(panel["panel"]).lower())
        panel["panel"] = panel_id
        observed.append(panel_id)
        _validate_compact_axis(panel["x_axis"], panel_id, "x")
        _validate_compact_axis(panel["y_axis"], panel_id, "y")
        if not isinstance(panel["series"], list) or not isinstance(panel["visible_trend"], str):
            raise StructuredOutputError("Compact graph series/trend fields are invalid.")
        level = panel["approximate_curve_level"]
        if level is not None and not isinstance(level, (int, float)):
            raise StructuredOutputError("Approximate curve level must be numeric or null.")
        complexity = panel["complexity_score"]
        if not isinstance(complexity, (int, float)) or not 0 <= complexity <= 1:
            raise StructuredOutputError("Compact graph complexity must be between 0 and 1.")
    expected = [
        re.sub(r"[^a-z0-9]+", "", str(panel).lower())
        for panel in expected_panels
    ]
    if sorted(observed) != sorted(expected) or len(observed) != len(set(observed)):
        raise StructuredOutputError(
            f"Compact response panels {observed} do not match expected panels {expected}."
        )
    return value


def _comparison_expectations(panels: list[dict]) -> tuple[list[str], str | None]:
    magnitude = [
        panel for panel in panels
        if "magnitude" in str(panel["graph_kind"]).lower()
        and isinstance(panel["approximate_curve_level"], (int, float))
    ]
    magnitude.sort(key=lambda panel: panel["approximate_curve_level"], reverse=True)
    order = [str(panel["group"]) for panel in magnitude]
    phase = [
        panel for panel in panels
        if "phase" in str(panel["graph_kind"]).lower()
    ]
    most_complex = (
        str(max(phase, key=lambda panel: panel["complexity_score"])["group"])
        if phase else None
    )
    return order, most_complex


def validate_compact_comparison(value: dict, panels: list[dict]) -> dict:
    _require_keys(
        value,
        {
            "magnitude_order_high_to_low", "greatest_phase_complexity_group",
            "x_axis", "magnitude_y_axis", "confidence", "uncertain",
        },
        "Compact graph comparison",
    )
    if not isinstance(value["magnitude_order_high_to_low"], list) or not isinstance(value["uncertain"], list):
        raise StructuredOutputError("Compact graph comparison lists are invalid.")
    _validate_compact_axis(value["x_axis"], "merged", "x")
    _validate_compact_axis(value["magnitude_y_axis"], "merged", "magnitude-y")
    expected_order, expected_complexity = _comparison_expectations(panels)
    if value["magnitude_order_high_to_low"] != expected_order:
        raise StructuredOutputError(
            "Compact comparison contradicts independently estimated curve levels."
        )
    if value["greatest_phase_complexity_group"] != expected_complexity:
        raise StructuredOutputError(
            "Compact comparison contradicts independently estimated phase complexity."
        )
    return value


def _consensus_panel_scale(panels: list[dict], axis_name: str, kind: str = "") -> str:
    scales = {
        str(panel.get(axis_name, {}).get("scale", "unknown")).lower()
        for panel in panels
        if not kind or kind in str(panel.get("graph_kind", "")).lower()
    }
    scales.discard("unknown")
    return next(iter(scales)) if len(scales) == 1 else "unknown"


def _merge_shared_axis_reading(panel_axis: dict, shared_axis: dict) -> dict:
    """Apply a shared label/unit reread without overwriting panel scale."""
    merged = dict(panel_axis)
    for field in ("label", "unit"):
        value = shared_axis.get(field)
        if isinstance(value, str) and value.strip():
            merged[field] = value
    if str(merged.get("scale", "unknown")).lower() == "unknown":
        shared_scale = str(shared_axis.get("scale", "unknown")).lower()
        if shared_scale in {"linear", "log"}:
            merged["scale"] = shared_scale
    return merged


def _compact_comparison_prompt(panels: list[dict], evidence_text: str) -> str:
    observations = [
        {
            "panel": panel["panel"],
            "group": panel["group"],
            "graph_kind": panel["graph_kind"],
            "approximate_curve_level": panel["approximate_curve_level"],
            "complexity_score": panel["complexity_score"],
            "visible_trend": panel["visible_trend"],
        }
        for panel in panels
    ]
    return f"""
Compare these already validated compact panel observations. Return JSON only.
Order magnitude groups by approximate_curve_level, highest to lowest. Select
phase complexity from complexity_score. If caption/body text explicitly
conflicts, put a concise note in uncertain rather than changing observations.
Independently reread and report the shared x-axis and magnitude y-axis labels
from the supplied panel images, preserving the visible variable name.

Panel observations:
{json.dumps(observations, ensure_ascii=False)}

Schema:
{{
  "magnitude_order_high_to_low": ["..."],
  "greatest_phase_complexity_group": "...",
  "x_axis": {{"label": "...", "unit": "...", "scale": "linear|log|unknown"}},
  "magnitude_y_axis": {{"label": "...", "unit": "...", "scale": "linear|log|unknown"}},
  "confidence": 0.0,
  "uncertain": ["..."]
}}

Caption/text cross-check:
{evidence_text[:1800]}
"""


def format_compact_graph_result(value: dict) -> str:
    lines = [f"**Figure {value.get('figure_number', '')} graph analysis**"]
    for panel in value["panels"]:
        lines.extend([
            f"\n- **Panel {panel['panel']} — {panel['group']}**",
            f"  - x-axis: {panel['x_axis']['label']} ({panel['x_axis']['unit']}, {panel['x_axis']['scale']})",
            f"  - y-axis: {panel['y_axis']['label']} ({panel['y_axis']['unit']}, {panel['y_axis']['scale']})",
            f"  - {panel['visible_trend']}",
        ])
    comparison = value["comparisons"]
    if comparison["magnitude_order_high_to_low"]:
        lines.append(
            "\n- Magnitude level, high to low: "
            + " > ".join(comparison["magnitude_order_high_to_low"])
        )
    if comparison["greatest_phase_complexity_group"]:
        lines.append(
            "- Greatest phase complexity: "
            + comparison["greatest_phase_complexity_group"]
        )
    if comparison["uncertain"] or value["uncertain_values"]:
        lines.append(
            "\n**Uncertain:** "
            + "; ".join([*comparison["uncertain"], *value["uncertain_values"]])
        )
    return "\n".join(lines)


def analyse_compact_multi_panel_graph(
    image_paths: list[Path],
    panel_groups: list[list[str]],
    figure_number: str,
    question: str,
    evidence_text: str = "",
    axis_label_image_paths: list[Path] | None = None,
    debug_info: dict | None = None,
) -> str:
    if len(image_paths) != len(panel_groups):
        raise StructuredOutputError("Panel image/group counts do not match.")
    panels = []
    uncertain_values = []
    raw_panel_responses = []
    for image_path, expected_panels in zip(image_paths, panel_groups):
        raw = _call_model(
            image_path,
            _compact_panel_prompt(question, expected_panels, evidence_text),
            COMPACT_PANEL_NUM_PREDICT,
        )
        raw_panel_responses.append(raw)
        if debug_info is not None:
            debug_info.update({
                "visual_type": "graph",
                "raw_panel_responses": raw_panel_responses,
                "raw_vision_response": "\n\n--- PANEL RESPONSE ---\n\n".join(
                    raw_panel_responses
                ),
            })
        try:
            pair = validate_compact_panel_response(
                parse_json_response(raw), expected_panels
            )
        except StructuredOutputError as first_error:
            retry = _call_model(
                image_path,
                _compact_panel_prompt(question, expected_panels, evidence_text, strict=True),
                COMPACT_PANEL_NUM_PREDICT,
            )
            raw_panel_responses.append(retry)
            if debug_info is not None:
                debug_info.update({
                    "raw_panel_responses": raw_panel_responses,
                    "raw_vision_response": "\n\n--- PANEL RESPONSE ---\n\n".join(
                        raw_panel_responses
                    ),
                })
            try:
                pair = validate_compact_panel_response(
                    parse_json_response(retry), expected_panels
                )
            except StructuredOutputError as second_error:
                if debug_info is not None:
                    debug_info.update({
                        "visual_type": "graph",
                        "raw_panel_responses": raw_panel_responses,
                        "raw_vision_response": "\n\n--- PANEL RESPONSE ---\n\n".join(raw_panel_responses),
                        "validation_error": f"{first_error} | {second_error}",
                        "final_answer_path": "could_not_verify_compact_graph",
                    })
                raise StructuredOutputError(
                    "Could not verify compact multi-panel graph JSON; truncated output "
                    "is not displayed."
                ) from second_error
        panels.extend(pair["panels"])
        uncertain_values.extend(pair["uncertain_values"])

    observed_ids = [str(panel["panel"]).lower() for panel in panels]
    expected_ids = [str(panel).lower() for group in panel_groups for panel in group]
    if sorted(observed_ids) != sorted(expected_ids) or len(observed_ids) != len(set(observed_ids)):
        if debug_info is not None:
            debug_info.update({
                "validation_error": "Merged compact graph panels are incomplete or duplicated.",
                "final_answer_path": "could_not_verify_compact_graph",
            })
        raise StructuredOutputError("Merged compact graph panels are incomplete or duplicated.")

    comparison_raw = _call_model_images(
        image_paths,
        _compact_comparison_prompt(panels, evidence_text),
        COMPACT_COMPARISON_NUM_PREDICT,
    )
    axis_label_retry_raw = ""
    try:
        comparison_candidate = parse_json_response(comparison_raw)
        magnitude_axis = comparison_candidate.get("magnitude_y_axis")
        original_magnitude_label = (
            magnitude_axis.get("label", "") if isinstance(magnitude_axis, dict) else ""
        )
        if isinstance(magnitude_axis, dict):
            magnitude_axis["label"] = normalize_impedance_axis_label(
                magnitude_axis.get("label", ""), evidence_text
            )
        if (
            str(figure_number).lower() == "9"
            and axis_label_image_paths
            and isinstance(magnitude_axis, dict)
            and _axis_label_needs_retry(
                original_magnitude_label,
                comparison_candidate.get("confidence"),
            )
        ):
            corrected_label, axis_label_retry_raw = reread_magnitude_axis_label(
                axis_label_image_paths, evidence_text
            )
            magnitude_axis["label"] = corrected_label
        if (
            isinstance(comparison_candidate.get("confidence"), (int, float))
            and comparison_candidate["confidence"] >= 0.7
            and isinstance(comparison_candidate.get("x_axis"), dict)
            and isinstance(comparison_candidate.get("magnitude_y_axis"), dict)
        ):
            x_scale = _consensus_panel_scale(panels, "x_axis")
            magnitude_scale = _consensus_panel_scale(
                panels, "y_axis", "magnitude"
            )
            if x_scale != "unknown":
                comparison_candidate["x_axis"]["scale"] = x_scale
            if magnitude_scale != "unknown":
                comparison_candidate["magnitude_y_axis"]["scale"] = magnitude_scale
            for panel in panels:
                panel["x_axis"] = _merge_shared_axis_reading(
                    panel["x_axis"], comparison_candidate["x_axis"]
                )
                if "magnitude" in str(panel["graph_kind"]).lower():
                    panel["y_axis"] = _merge_shared_axis_reading(
                        panel["y_axis"],
                        comparison_candidate["magnitude_y_axis"],
                    )
        comparison = validate_compact_comparison(comparison_candidate, panels)
    except StructuredOutputError as error:
        if debug_info is not None:
            debug_info.update({
                "raw_comparison_response": comparison_raw,
                "raw_axis_label_retry_response": axis_label_retry_raw,
                "raw_vision_response": (
                    "\n\n--- PANEL RESPONSE ---\n\n".join(raw_panel_responses)
                    + "\n\n--- COMPARISON RESPONSE ---\n\n"
                    + comparison_raw
                    + ("\n\n--- AXIS LABEL RETRY RESPONSE ---\n\n" + axis_label_retry_raw
                       if axis_label_retry_raw else "")
                ),
                "validation_error": str(error),
                "final_answer_path": "could_not_verify_compact_graph",
            })
        raise
    result = {
        "figure_number": figure_number,
        "panels": sorted(panels, key=lambda panel: str(panel["panel"]).lower()),
        "comparisons": comparison,
        "uncertain_values": uncertain_values,
    }
    if debug_info is not None:
        debug_info.update({
            "visual_type": "graph",
            "raw_panel_responses": raw_panel_responses,
            "raw_comparison_response": comparison_raw,
            "raw_axis_label_retry_response": axis_label_retry_raw,
            "raw_vision_response": (
                "\n\n--- PANEL RESPONSE ---\n\n".join(raw_panel_responses)
                + "\n\n--- COMPARISON RESPONSE ---\n\n"
                + comparison_raw
                + ("\n\n--- AXIS LABEL RETRY RESPONSE ---\n\n" + axis_label_retry_raw
                   if axis_label_retry_raw else "")
            ),
            "validated_json": result,
            "validation_error": "",
            "final_answer_path": "validated_typed_vision",
        })
    return format_compact_graph_result(result)


def _repair_prompt(visual_type: str, raw: str, error: str) -> str:
    graph_rules = (
        "For Nyquist panels, reread each panel marker and sample label, apply axis "
        "multipliers to visible_range, describe both fits, and locate deviations using "
        "left/right or low/high Re(Z). A fit/deviation comparison subject must be the "
        "visible sample/group. Do not infer frequency direction without visible arrows "
        "or labels, and do not claim a perfect entire-range fit when deviations remain."
        if visual_type == "graph" else ""
    )
    return f"""
Repair the following malformed {visual_type} response. Return exactly one JSON
object matching the schema, with no fences or explanation. Preserve only
information already present; use null or "unreadable" instead of inventing data.
{graph_rules}

Schema:
{_schema_text(visual_type)}

Validation error:
{error}

Malformed response:
{raw}
"""


def _graph_reinspection_prompt(
    question: str, error: str, evidence_text: str
) -> str:
    return build_structured_prompt("graph", question, evidence_text) + f"""

The prior image reading failed strict validation: {error}
Reinspect the image from scratch; do not copy prior invalid fields. In
particular, infer linear/log only from visible ticks, keep complexity_score
between 0 and 1, apply scientific multipliers to visible_range, and make every
fit/deviation comparison subject a visible panel sample or group. For Nyquist
plots, inspect the separation between data points and the model curve at both
leftmost and rightmost endpoints before reporting deviations as left/right or
low/high Re(Z). Compare fit closeness by the largest visible separation divided
by each panel's displayed y-axis range; do not compare raw separation across
different scientific multipliers, and do not use point density as a fit proxy.
Leave frequency direction unsupported unless an arrow or explicit frequency
label is visible.
"""


def _is_topology_error(error: Exception) -> bool:
    return bool(re.search(
        r"\b(?:circuit|topology|branch|node|junction|edge|component|connection|parallel|series)\b",
        str(error),
        re.IGNORECASE,
    ))


def _topology_retry_prompt(raw: str, error: str, evidence_text: str) -> str:
    grounded_series = _explicit_series_pairs(evidence_text)
    grounded_parallel = _explicit_parallel_pairs(evidence_text)
    return f"""
The circuit topology failed validation. Reinspect only the visible wires,
junctions and branches. Do not rewrite the surrounding explanation.

Rules:
- list every electrical component as a separate component object;
- use two shared terminal node IDs for branches that are parallel;
- each edge represents exactly one component;
- list components in traversal order within each branch;
- connections may reference component IDs, biological labels, node IDs or branch IDs;
- obey explicit caption text and impedance equations;
- grounded series component pairs: {json.dumps(grounded_series)};
- grounded parallel component pairs from impedance equations: {json.dumps(grounded_parallel)};
- components in a grounded parallel pair must occupy different branches that
  share the same two endpoints;
- return exactly one compact JSON object and no prose.

Schema:
{{
  "components": [{{"name": "...", "description": "..."}}],
  "connections": [{{"from": "...", "to": "...", "relationship": "..."}}],
  "circuit_topology": {{
    "nodes": [{{"id": "...", "label": "..."}}],
    "edges": [{{"from_node": "...", "to_node": "...", "component": "..."}}],
    "branches": [
      {{"id": "...", "start_node": "...", "end_node": "...", "components": ["..."]}}
    ],
    "parallel_branch_sets": [["branch id", "branch id"]]
  }},
  "uncertain_items": ["..."]
}}

Validation error: {error}
Caption/equation evidence: {evidence_text[:4000]}
Previous response: {raw[:4000]}
"""


def format_structured_result(visual_type: str, value: dict) -> str:
    if visual_type == "table":
        title = value.get("title") or f"Table {value.get('table_number', '')}".strip()
        lines = [f"**{title}**", "", " | ".join(map(str, value["columns"]))]
        lines.append(" | ".join(["---"] * len(value["columns"])))
        for row in value["rows"]:
            lines.append(" | ".join("unreadable" if cell is None else str(cell) for cell in row))
        if value["comparisons"]:
            lines.extend(["", *[f"- {item}" for item in value["comparisons"]]])
        return "\n".join(lines)
    if visual_type == "graph":
        lines = [f"**Figure {value.get('figure_number', '')} graph analysis**"]
        for panel in value["panels"]:
            lines.append(f"\n- **Panel {panel['panel']}**")
            x_axis, y_axis = panel["x_axis"], panel["y_axis"]
            lines.append(
                f"  - x-axis: {x_axis['label']} "
                f"({x_axis['unit'] or 'no unit'}, {x_axis['scale']})"
            )
            lines.append(
                f"  - y-axis: {y_axis['label']} "
                f"({y_axis['unit'] or 'no unit'}, {y_axis['scale']})"
            )
            lines.append(f"  - series: {', '.join(map(str, panel['series']))}")
            lines.extend(f"  - {trend}" for trend in panel["visible_trends"])
        for comparison in value["comparisons"]:
            qualifier = "Uncertain: " if comparison["uncertain"] else ""
            lines.append(f"\n- {qualifier}{comparison['claim']}")
        if value["uncertain_values"]:
            lines.append("\n**Uncertain:** " + ", ".join(map(str, value["uncertain_values"])))
        return "\n".join(lines)
    lines = [value["explanation"]]
    if value["components"]:
        if value.get("circuit_topology"):
            lines.append("\n**Components:**")
            for item in value["components"]:
                if isinstance(item, str):
                    lines.append(f"- {item}")
                else:
                    name = str(item.get("name", "")).strip()
                    description = str(item.get("description", "")).strip()
                    if name:
                        lines.append(
                            f"- {name}: {description}" if description else f"- {name}"
                        )
        else:
            names = [
                item if isinstance(item, str) else item.get("name", "")
                for item in value["components"]
            ]
            lines.append("\n**Components:** " + ", ".join(filter(None, names)))
    if value["connections"]:
        lines.append("\n**Connections:**")
        lines.extend(
            f"- {item['from']} — {item['relationship']} — {item['to']}"
            for item in value["connections"]
        )
    if value.get("circuit_topology"):
        lines.append("\n**Circuit branches:**")
        lines.extend(
            f"- {branch['id']}: {' then '.join(map(str, branch['components']))} "
            f"({branch['start_node']} to {branch['end_node']})"
            for branch in value["circuit_topology"]["branches"]
        )
    if value["uncertain_items"]:
        lines.append("\n**Uncertain:** " + ", ".join(map(str, value["uncertain_items"])))
    return "\n".join(lines)


def _grounded_caption_fallback(evidence_text: str, visual_type: str) -> str:
    caption_match = re.search(
        r"TARGET FIGURE CAPTION[^\n]*:\s*\n(.*?)"
        r"(?=\n\s*\n(?:PAGE TEXT|RETRIEVED TEXT)[^\n]*:|\Z)",
        evidence_text,
        re.IGNORECASE | re.DOTALL,
    )
    caption = (
        re.sub(r"\s+", " ", caption_match.group(1)).strip()
        if caption_match else ""
    )
    if caption:
        return (
            f"**Grounded caption summary:** {caption}\n\n"
            "I could not verify every structured relationship in the image, so "
            "no unvalidated model output is shown."
        )
    readable_type = visual_type.replace("_", " ")
    return (
        f"Could not verify a structured reading of this {readable_type}. "
        "No unvalidated model output is shown."
    )


def analyse_typed_image(
    image_path: Path,
    visual_type: str,
    question: str,
    evidence_text: str = "",
    debug_info: dict | None = None,
    fit_verification_images: list[tuple[str, Path]] | None = None,
) -> str:
    prompt = build_structured_prompt(visual_type, question, evidence_text)
    circuit_context = visual_type == "labelled_diagram" and bool(re.search(
        r"\b(?:circuit|resistor|capacitor|impedance)\b",
        question,
        re.IGNORECASE,
    ))
    try:
        raw = _call_model(image_path, prompt)
    except Exception as error:
        if debug_info is not None:
            debug_info.update({
                "visual_type": visual_type,
                "initial_response": "",
                "raw_vision_response": "",
                "initial_parsed_json": None,
                "initial_validation_errors": [],
                "repaired_json": None,
                "repaired_validation_result": "not_attempted",
                "validation_error": str(error),
                "final_answer_path": "vision_model_error",
                "final_answer_code_path": "vision_model_error",
            })
        raise
    initial_raw = raw
    errors = []
    parsed = None
    initial_validation_errors = []
    retry_kind = ""
    retry_raw = ""
    repaired_json = None
    repaired_validation_result = "not_attempted"
    used_repair = False
    try:
        parsed = parse_json_response(raw)
        value = validate_typed_response(visual_type, parsed, evidence_text)
    except StructuredOutputError as first_error:
        errors.append(str(first_error))
        initial_validation_errors.append(str(first_error))
        topology_retry = (
            visual_type == "labelled_diagram"
            and isinstance(parsed, dict)
            and str(parsed.get("diagram_kind", "")).lower() == "circuit"
            and _is_topology_error(first_error)
        )
        if topology_retry:
            try:
                repaired_json = ground_circuit_topology_from_evidence(
                    parsed, evidence_text
                )
                value = validate_typed_response(
                    visual_type, repaired_json, evidence_text
                )
                repaired_validation_result = "passed"
                retry_kind = "grounded_topology_repair"
                used_repair = True
            except StructuredOutputError as grounding_error:
                errors.append(str(grounding_error))

        if not used_repair:
            if topology_retry:
                retry_kind = "targeted_topology_retry"
                retry_raw = _call_model(
                    image_path,
                    _topology_retry_prompt(raw, str(first_error), evidence_text),
                )
            elif visual_type == "graph":
                retry_kind = "targeted_graph_reinspection"
                retry_raw = _call_model(
                    image_path,
                    _graph_reinspection_prompt(
                        question, str(first_error), evidence_text
                    ),
                )
            else:
                retry_kind = "json_repair"
                retry_raw = _call_model(
                    image_path,
                    _repair_prompt(visual_type, raw, str(first_error)),
                )
            try:
                repaired_json = parse_json_response(retry_raw)
                if topology_retry:
                    candidate = dict(parsed)
                    for field in (
                        "components", "connections", "circuit_topology",
                        "uncertain_items",
                    ):
                        if field == "uncertain_items":
                            continue
                        if field in repaired_json and not (
                            field in {"components", "connections"}
                            and not repaired_json[field]
                            and parsed.get(field)
                        ):
                            candidate[field] = repaired_json[field]
                    repaired_json = candidate
                value = validate_typed_response(
                    visual_type, repaired_json, evidence_text
                )
                repaired_validation_result = "passed"
                raw = retry_raw
                used_repair = True
            except StructuredOutputError as second_error:
                errors.append(str(second_error))
                recovered = False
                if topology_retry and isinstance(repaired_json, dict):
                    try:
                        repaired_json = ground_circuit_topology_from_evidence(
                            repaired_json, evidence_text
                        )
                        value = validate_typed_response(
                            visual_type, repaired_json, evidence_text
                        )
                        repaired_validation_result = "passed"
                        raw = retry_raw
                        retry_kind = "targeted_topology_grounded_repair"
                        used_repair = True
                        recovered = True
                    except StructuredOutputError as grounding_error:
                        errors.append(str(grounding_error))
                if not recovered:
                    repaired_validation_result = str(second_error)
                    if debug_info is not None:
                        fallback_path = (
                            "could_not_verify_topology"
                            if topology_retry or circuit_context
                            else "grounded_caption_summary_fallback"
                        )
                        debug_info.update({
                            "visual_type": visual_type,
                            "initial_response": initial_raw,
                            "raw_vision_response": raw,
                            "retry_response": retry_raw,
                            "initial_parsed_json": parsed,
                            "initial_validation_errors": initial_validation_errors,
                            "repaired_json": repaired_json,
                            "repaired_validation_result": repaired_validation_result,
                            "retry_kind": retry_kind,
                            "validation_error": " | ".join(errors),
                            "final_answer_path": fallback_path,
                            "final_answer_code_path": fallback_path,
                        })
                    if topology_retry or circuit_context:
                        return (
                            "Could not verify the circuit topology from the visible wire "
                            "junctions and caption evidence."
                        )
                    return _grounded_caption_fallback(evidence_text, visual_type)

    fit_verification_raw = []
    fit_verification_status = "not_applicable"
    if (
        visual_type == "graph"
        and fit_verification_images
    ):
        value, fit_verification_raw, fit_verification_status = (
            verify_nyquist_fit_comparison(
                value, fit_verification_images, evidence_text
            )
        )

    if debug_info is not None:
        recorded_raw = raw
        if fit_verification_raw:
            recorded_raw += (
                "\n\n--- TIGHT PANEL FIT VERIFICATION ---\n\n"
                + "\n\n--- PANEL ---\n\n".join(fit_verification_raw)
            )
        debug_info.update({
            "visual_type": visual_type,
            **({"initial_response": initial_raw} if errors else {}),
            "raw_vision_response": recorded_raw,
            "retry_response": retry_raw,
            "raw_fit_verification_responses": fit_verification_raw,
            "fit_verification_status": fit_verification_status,
            "initial_parsed_json": parsed,
            "initial_validation_errors": initial_validation_errors,
            "repaired_json": repaired_json,
            "repaired_validation_result": repaired_validation_result,
            "validated_json": value,
            "validation_error": "",
            "retry_kind": retry_kind if errors else "",
            "final_answer_path": "validated_typed_vision",
            "final_answer_code_path": (
                "validated_repaired_structured_vision"
                if used_repair else "validated_structured_vision"
            ),
        })
    return format_structured_result(visual_type, value)
