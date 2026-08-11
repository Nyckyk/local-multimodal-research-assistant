from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from rag.retrieval import (
    extract_discussion_limitations,
    retrieve_context,
)
from services.document_matching import explicit_document_matches
from services.query_rewriter import rewrite_question
from services.scientific_evidence import (
    caption_results_fallback,
    classify_experimental_domain,
    document_glossary,
    extract_caption_panels,
    merge_mixed_figure_with_caption,
    remove_unsupported_acronym_expansions,
    requested_answer_slots,
    structured_semantically_sufficient,
    unsupported_source_completion,
)
from services.structured_vision import validate_compact_comparison, validate_compact_panel_response, validate_labelled_diagram
from services.visual_index import flatten_visual_targets, load_or_build_visual_index
from services.visual_locator import resolve_visual_target
from services.visual_runtime import analyse_resolved_visual


PDF_NAME = "Detection of senescence using machine learning algorithms based on nuclear features.pdf"
EEG_NAME = "A hybrid boundary element-finite element approach for solving the EEG forward problem in brain modeling.pdf"
TITLE = Path(PDF_NAME).stem


class Vector:
    def tolist(self):
        return [0.1, 0.2]


class Embedder:
    def encode(self, *args, **kwargs):
        return Vector()


class Reranker:
    def predict(self, pairs):
        return [float(index) / 10 for index, _ in enumerate(pairs)]


class EvidenceCollection:
    def __init__(self):
        self.rows = [
            ("Abstract This original research develops nuclear-feature classifiers and validates them across cell culture, animal tissue and patients.", {"pdf": PDF_NAME, "source": PDF_NAME.replace(".pdf", ".txt"), "page": 2, "chunk": 0}),
            ("Results Nuclear features can identify senescent cells in treated A549 cell cultures.", {"pdf": PDF_NAME, "page": 3, "chunk": 0}),
            ("Discussion The classifier was validated in mouse liver tissue and human patient liver samples.", {"pdf": PDF_NAME, "page": 12, "chunk": 0}),
            ("Methods Each plate contained 30 treated wells and 30 DMSO control wells, using at least 3 plates and 0.1-0.9 million cells per condition.", {"pdf": PDF_NAME, "page": 16, "chunk": 0}),
            ("Methods Training datasets randomly selected 10,000 normal and 10,000 treated cells. Independent randomizations were constructed for classification tree and random forest algorithms.", {"pdf": PDF_NAME, "page": 16, "chunk": 1}),
            ("Methods Area, Form Factor, Elongation, Compactness, Chord Ratio, Gyration, and Displacement were used as nuclear features.", {"pdf": PDF_NAME, "page": 16, "chunk": 2}),
            ("Methods Classification trees used 30% test size and cost-complexity pruning with optimal alpha. Random forests used test size 0.5 and probability > 0.5 as senescent.", {"pdf": PDF_NAME, "page": 16, "chunk": 3}),
            ("Methods Human patient samples used circularity >0.7 to select hepatocytes rather than fibroblasts or immune cells; samples below 10,000 cells were excluded.", {"pdf": PDF_NAME, "page": 17, "chunk": 0}),
            ("Abstract EEG forward modelling paper.", {"pdf": EEG_NAME, "page": 1, "chunk": 0}),
            ("Discussion EEG solver conclusions.", {"pdf": EEG_NAME, "page": 10, "chunk": 0}),
        ]

    def count(self):
        return len(self.rows)

    def query(self, **kwargs):
        return {
            "documents": [[row[0] for row in self.rows]],
            "metadatas": [[row[1] for row in self.rows]],
            "distances": [[0.1] * len(self.rows)],
        }

    def get(self, where=None, include=None):
        rows = self.rows
        if where:
            key, value = next(iter(where.items()))
            rows = [row for row in rows if row[1].get(key) == value]
        return {"documents": [row[0] for row in rows], "metadatas": [row[1] for row in rows]}


@pytest.fixture(scope="module")
def targets():
    return [
        row for row in flatten_visual_targets(load_or_build_visual_index())
        if row["pdf_name"] == PDF_NAME and row.get("match_kind") == "caption"
    ]


def target(targets, number):
    return next(row for row in targets if row["target_type"] == "figure" and row["target_number"] == str(number))


def test_exact_title_matches_and_overrides_previous_document():
    matches = explicit_document_matches(f"Summarise {TITLE}", [PDF_NAME, EEG_NAME])
    assert matches == {PDF_NAME}
    _, sources = retrieve_context(
        f"Summarise the main findings of {TITLE}", "previous EEG question",
        EvidenceCollection(), Embedder(), Reranker(), selected_source=EEG_NAME,
    )
    assert sources and {source["source"] for source in sources} == {PDF_NAME}


def test_partial_unique_title_matches():
    assert explicit_document_matches(
        "Summarise detection of senescence using machine learning algorithms",
        [PDF_NAME, EEG_NAME],
    ) == {PDF_NAME}


def test_long_summary_uses_correct_document():
    context, sources = retrieve_context(
        f"Give me a detailed overview of the paper {TITLE}", "",
        EvidenceCollection(), Embedder(), Reranker(), selected_source=EEG_NAME,
    )
    assert PDF_NAME in context
    assert sources and all(source["source"] == PDF_NAME for source in sources)


def test_figure_one_has_complete_a_to_i_panel_map(targets):
    row = target(targets, 1)
    assert row["page_number"] == 3 and row["caption_page_number"] == 4
    assert [item["panel"] for item in extract_caption_panels(row["full_caption"])] == list("abcdefghi")


def test_schema_valid_but_empty_diagram_is_rejected():
    sufficient, details = structured_semantically_sufficient(
        "labelled_diagram",
        {"labels": list("abcdefghi"), "components": [], "connections": [], "spatial_relationships": [], "explanation": ""},
        "Explain every panel A-I",
    )
    assert not sufficient and not details["informative"]


@pytest.mark.parametrize("number", [2, 5, 8])
def test_mixed_figure_uses_caption_results_fallback_on_vision_failure(targets, number):
    index = load_or_build_visual_index()
    resolution = resolve_visual_target(
        f"Explain Figure {number} in {TITLE}", index
    )
    debug = {}
    with patch("services.visual_runtime.analyse_pdf_page", side_effect=lambda **kwargs: kwargs["debug_info"].update({"final_answer_path": "grounded_caption_summary_fallback"}) or "unsafe raw"):
        answer = analyse_resolved_visual("Explain every panel", resolution, debug_info=debug)
    assert debug["final_answer_code_path"] == "grounded_caption_results_fallback"
    assert "caption- and Results-grounded" in answer
    assert "unsafe raw" not in answer


def test_figure_three_full_caption_contains_required_panels(targets):
    panels = {row["panel"] for row in extract_caption_panels(target(targets, 3)["full_caption"])}
    assert set("cdegi").issubset(panels)


def test_figure_seven_bad_optional_connection_does_not_discard_components():
    value = validate_labelled_diagram({
        "diagram_kind": "other", "labels": ["a", "cell"],
        "components": [{"name": "cell", "description": "visible cell"}],
        "spatial_relationships": [],
        "connections": [{"from": "cell", "to": "hallucinated endpoint", "relationship": "points to"}],
        "circuit_topology": None, "explanation": "Panel a shows a cell.", "uncertain_items": [],
    })
    assert value["components"] and value["connections"] == []
    assert any("unverified endpoint" in item for item in value["uncertain_items"])


def test_figure_nine_methods_inclusion_criteria_are_retrieved():
    context, _ = retrieve_context(
        f"For Figure 9 in {TITLE}, what inclusion criteria and threshold were used?", "",
        EvidenceCollection(), Embedder(), Reranker(),
    )
    assert "METHODS-AWARE RETRIEVAL" in context
    assert "circularity >0.7" in context and "below 10,000 cells" in context


def test_followup_these_samples_resolves_recent_human_experiment():
    question = "Why did they impose that threshold for these samples?"
    history = [{"role": "assistant", "content": "The analysis used human NAFLD liver samples."}]
    assert "human NAFLD liver samples" in rewrite_question(question, history)


def test_methods_training_question_covers_every_requested_field():
    question = (
        f"In {TITLE}, include the nuclear features used, how training libraries were assembled, "
        "number of cells, CT split, RF split, overfitting control and RF threshold."
    )
    context, _ = retrieve_context(question, "", EvidenceCollection(), Embedder(), Reranker())
    for expected in (
        "30 treated wells", "at least 3 plates", "0.1-0.9 million", "10,000 normal",
        "Independent randomizations", "30%", "test size 0.5", "cost-complexity pruning",
        "probability > 0.5", "Form Factor", "Displacement",
    ):
        assert expected.casefold() in context.casefold()
    assert not unsupported_source_completion(context)


def test_requested_methods_slots_are_explicit():
    slots = requested_answer_slots(
        "include the features, library construction, number of cells, CT split, RF split, overfitting and RF threshold"
    )
    assert {"features", "library construction", "training cell counts", "CT split", "RF split", "CT overfitting method", "RF threshold"}.issubset(slots)


def test_discussion_limitations_are_author_stated():
    result = extract_discussion_limitations([
        "Discussion. The choice of marker affects comparisons between classifiers. "
        "The tissue score may need to be adapted for other tissues."
    ])
    assert result["status"] == "explicit_author_limitations"
    assert len(result["items"]) == 2


def test_experimental_domains_remain_distinct():
    assert classify_experimental_domain("A549 human cells were treated in culture") == "in_vitro_cell_line"
    assert classify_experimental_domain("mouse liver tissue in vivo") == "animal_model"
    assert classify_experimental_domain("human patient liver samples") == "human_clinical_patient_tissue"


def test_human_cell_line_is_not_patient_evidence():
    assert classify_experimental_domain("human A549 cell line in vitro") != "human_clinical_patient_tissue"


def test_unsupported_acronym_expansion_is_removed():
    text = "AEM (assumption-based etoposide model) was defined. IEM was not defined."
    glossary = document_glossary(text)
    answer = remove_unsupported_acronym_expansions("AEM (assumption-based etoposide model); IEM (Isolation Forest model)", glossary)
    assert "assumption-based etoposide model" in answer
    assert "Isolation Forest" not in answer and "IEM" in answer


def test_partial_mixed_vision_is_completed_only_from_caption(targets):
    panel_map = extract_caption_panels(target(targets, 1)["full_caption"])
    result = {"figure_number": "1", "panels": [{
        "panel": "a", "visual_type": "workflow",
        "structured_analysis": {"summary": "visible workflow", "observations": []},
        "confidence": 0.9, "uncertain_items": [],
    }], "explanation": "", "uncertain_items": []}
    merged, added = merge_mixed_figure_with_caption(result, panel_map)
    assert added and [row["panel"] for row in merged["panels"]] == list("abcdefghi")
    assert merged["panels"][1]["structured_analysis"]["evidence"] == ["caption"]


def test_generic_figure_one_remains_ambiguous_across_five_pdfs():
    resolution = resolve_visual_target("Explain Figure 1.", load_or_build_visual_index())
    assert resolution.status == "ambiguous"
    assert len({row["pdf_name"] for row in resolution.candidates}) == 5


def test_compact_graph_group_sample_suffixes_normalize_before_comparison():
    panels = []
    for panel, group, kind, level, complexity in (
        ("a", "Group 1: Samples 10-9", "magnitude", 115, 0.0),
        ("b", "Group 1: Samples 10-9", "phase", -45, 0.0),
        ("c", "Group 2", "magnitude", 92.5, 0.3),
        ("d", "Group 2", "phase", -50, 0.6),
        ("e", "Group 3", "magnitude", 125, 0.3),
        ("f", "Group 3", "phase", -75, 0.3),
    ):
        panels.append({
            "panel": panel, "group": group, "graph_kind": kind,
            "x_axis": {"label": "Frequency", "unit": "Hz", "scale": "log"},
            "y_axis": {"label": "value", "unit": "", "scale": "linear"},
            "series": [], "visible_trend": "visible", "approximate_curve_level": level,
            "complexity_score": complexity,
        })
    validated = validate_compact_panel_response(
        {"panels": panels, "uncertain_values": []}, list("abcdef")
    )
    comparison = validate_compact_comparison({
        "magnitude_order_high_to_low": ["Group 3", "Group 1", "Group 2"],
        "greatest_phase_complexity_group": "Group 2",
        "x_axis": {"label": "Frequency", "unit": "Hz", "scale": "log"},
        "magnitude_y_axis": {"label": "|Z|", "unit": "ohm", "scale": "linear"},
        "confidence": 0.9, "uncertain": [],
    }, validated["panels"])
    assert comparison["magnitude_order_high_to_low"] == ["Group 3", "Group 1", "Group 2"]
