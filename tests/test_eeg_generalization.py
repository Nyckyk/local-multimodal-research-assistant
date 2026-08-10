from pathlib import Path
from unittest.mock import patch
import json

import fitz
import yaml

from rag.retrieval import is_reported_trend_question, resolve_followup_retrieval_query
from services.equation_service import (
    analyse_resolved_equation, build_equation_evidence,
)
from services.multi_target import extract_experimental_context
from services.multi_target import analyse_equation_chain
from services.scientific_metrics import (
    best_by_metric, classify_numeric_trend, grounded_table_trends,
    infer_metric_semantics,
)
from services.structured_vision import detect_visual_type
from services.text_table import extract_text_table
from services.visual_index import load_or_build_visual_index
from services.visual_locator import resolve_visual_target, resolve_visual_targets
from services.visual_reference_parser import parse_visual_references
from services.visual_runtime import analyse_resolved_visual
from services.vision_service import _associated_graph_text


ROOT = Path(__file__).resolve().parents[1]
EXPECTED = yaml.safe_load((ROOT / "tests/fixtures/eeg_expected.yaml").read_text(encoding="utf-8"))
PDF = ROOT / "papers" / EXPECTED["pdf_filename"]


def test_multi_reference_parser_lists_and_ranges():
    cases = {
        "Figures 9 and 10": [("figure", "9"), ("figure", "10")],
        "Figs. 4–7": [("figure", str(n)) for n in range(4, 8)],
        "Tables 2 and 3": [("table", "2"), ("table", "3")],
        "Eqs. (18)–(20)": [("equation", str(n)) for n in range(18, 21)],
        "Equation 18, Equation 19 and Equation 20": [("equation", str(n)) for n in range(18, 21)],
    }
    for question, expected in cases.items():
        assert [(row.target_type, row.target_number) for row in parse_visual_references(question)] == expected


def test_equations_18_to_20_resolve_in_order_in_exact_document():
    question = "In the hybrid EEG paper, explain Equations 18, 19 and 20 and show how they couple the regions."
    values = resolve_visual_targets(question, load_or_build_visual_index())
    assert [value.target_number for value in values] == ["18", "19", "20"]
    assert all(value.status == "resolved" for value in values)
    assert {value.page_number for value in values} == {EXPECTED["equation_page"]}
    assert all(value.pdf_name == EXPECTED["pdf_filename"] for value in values)

    debug = {}
    with patch("services.multi_target.generate_answer") as generic_answer:
        answer = analyse_equation_chain(question, values, debug_info=debug)
    generic_answer.assert_not_called()
    assert debug["generation_code_path"] == "deterministic_interface_coupling"
    assert debug["symbolic_equation_chain"]["unknowns"] == ["φ_FE", "X_BE"]
    assert "mean of the three FE nodal potentials" in answer
    assert "opposite outward normals" in answer


def test_equation_one_preserves_operators_source_and_boundary_condition():
    resolution = resolve_visual_target(
        "Explain Equation 1 in the hybrid EEG paper.", load_or_build_visual_index()
    )
    debug = {}
    answer = analyse_resolved_equation("Explain Equation 1.", resolution, debug_info=debug)
    symbolic = debug["symbolic_equation"]
    assert symbolic["governing_equation"]["lhs_operators"] == ["divergence", "conductivity_tensor", "gradient"]
    assert symbolic["governing_equation"]["rhs"] == "∇·J_P(t)"
    assert symbolic["boundary_condition"]["expression"] == "σ ∂ϕ/∂n = 0"
    assert "\\nabla\\!\\cdot\\mathbf{J}_P" in answer


def test_eeg_figure_one_is_schematic_not_graph():
    resolution = resolve_visual_target("Explain Figure 1 in the hybrid EEG paper.", load_or_build_visual_index())
    assert resolution.visual_type == "anatomical_schematic"
    assert detect_visual_type("Explain Figure 1.", resolution.caption) == "anatomical_schematic"


def test_eeg_figure_one_runtime_does_not_require_graph_axes():
    resolution = resolve_visual_target("Explain Figure 1 in the hybrid EEG paper.", load_or_build_visual_index())
    response = {
        "diagram_kind": "other",
        "labels": ["EEG electrode", "Dipoles", "Brain", "Skull", "Scalp"],
        "components": [{"name": "head models", "description": "three schematic panels"}],
        "spatial_relationships": [], "connections": [], "circuit_topology": None,
        "explanation": "Panel A shows the EEG forward model; panels B and C show layered head models.",
        "uncertain_items": [],
    }
    debug = {}
    with patch("services.structured_vision._call_model", return_value=json.dumps(response)):
        answer = analyse_resolved_visual("Explain Figure 1 in the hybrid EEG paper.", resolution, debug_info=debug)
    assert debug["final_answer_path"] == "validated_typed_vision"
    assert debug["visual_type"] == "anatomical_schematic"
    assert "x-axis" not in answer


def test_table_one_extracts_clean_header_and_grounded_explanation():
    with fitz.open(PDF) as document:
        table = extract_text_table(
            document[6], "1",
            "Extract Table 1 and explain what the conductivity intervals and anisotropy ratio represent.",
        )
    assert table["columns"] == ["Tissue", "Brain", "Skull", "Scalp"]
    explanation = " ".join(table["comparisons"])
    assert "radial conductivity" in explanation and "tangential conductivity" in explanation
    assert "interval" in explanation.casefold()


def test_table_five_hierarchical_full_fixture():
    with fitz.open(PDF) as document:
        table = extract_text_table(document[13], "5", "Extract every row and column from Table 5.")
    assert len(table["header_rows"]) == 2
    assert table["header_spans"]
    assert table["columns"] == EXPECTED["table_5"]["columns"]
    assert table["rows"] == EXPECTED["table_5"]["rows"]


def test_table_five_runtime_uses_validated_hierarchical_text_fallback():
    resolution = resolve_visual_target(
        "Extract every row and column from Table 5 in the hybrid EEG paper.",
        load_or_build_visual_index(),
    )
    debug = {}
    with patch("services.structured_vision._call_model", return_value="{}"):
        answer = analyse_resolved_visual(
            "Extract every row and column from Table 5 in the hybrid EEG paper.",
            resolution, debug_info=debug,
        )
    assert debug["final_answer_path"] == "validated_text_table_fallback"
    assert debug["validated_json"]["rows"] == EXPECTED["table_5"]["rows"]
    assert "Radial direction" in answer and "Tangential direction" in answer


def test_metric_semantics_use_targets_not_larger_is_better():
    with fitz.open(PDF) as document:
        evidence = build_equation_evidence(PDF, 6, "21") + build_equation_evidence(PDF, 6, "22")
    semantics = infer_metric_semantics(evidence)
    assert semantics["RDM"]["target_value"] == 0
    assert semantics["MAG"]["target_value"] == 1
    assert best_by_metric({"a": 0.8, "b": 1.02, "c": 1.2}, semantics["MAG"]) == "b"


def test_figure_six_values_are_non_monotonic():
    values = [0.043, 0.0436, 0.0483, 0.0385, 0.103, 0.0984]
    assert classify_numeric_trend(values) == "non-monotonic"
    with fitz.open(PDF) as document:
        trends = grounded_table_trends(
            _associated_graph_text(document, 12, document[12].get_text("text"))
        )
    assert trends["RDM|Hybrid BE-FE"]["classification"] == "non-monotonic"


def test_figure_seven_peak_compares_all_eccentricities():
    eccentricities = ["50%", "60%", "70%", "80%", "90%", "98%"]
    values = [0.0216, 0.0269, 0.0329, 0.0395, 0.0848, 0.0526]
    assert eccentricities[values.index(max(values))] == "90%"
    assert classify_numeric_trend(values) != "monotonic increasing"
    with fitz.open(PDF) as document:
        trends = grounded_table_trends(
            _associated_graph_text(document, 13, document[13].get_text("text"))
        )
    assert trends["RDM|Hybrid BE-FE"]["maximum_position"] == "90%"


def test_figures_nine_and_ten_resolve_with_separate_provenance():
    values = resolve_visual_targets("Compare Figures 9 and 10 in the hybrid EEG paper.", load_or_build_visual_index())
    assert [(value.target_number, value.page_number) for value in values] == [("9", 15), ("10", 16)]
    contexts = [extract_experimental_context(value.caption) for value in values]
    assert [context["orientation"] for context in contexts] == ["radial", "tangential"]
    assert all(context["model_geometry"] == "realistic" for context in contexts)
    assert all(context["conductivity"] == "anisotropic" for context in contexts)


def test_experimental_context_keeps_spherical_scenarios_paired():
    values = resolve_visual_targets("Compare Figures 4–7 in the hybrid EEG paper.", load_or_build_visual_index())
    contexts = {value.target_number: extract_experimental_context(value.caption) for value in values}
    assert contexts["4"]["tissue_layers"] == contexts["5"]["tissue_layers"] == "three"
    assert contexts["6"]["tissue_layers"] == contexts["7"]["tissue_layers"] == "four"
    assert all(contexts[number]["model_geometry"] == "spherical" for number in contexts)


def test_limitations_and_future_work_are_adjacent_in_conclusion():
    with fitz.open(PDF) as document:
        text = document[16].get_text("text")
    lower = text.casefold()
    assert "more time consuming" in lower.replace("-", " ")
    assert "three times" in lower
    assert "mesh" in lower and "complex" in lower
    assert "future research" in lower
    assert "heterogeneous" in lower and "white matter" in lower
    assert "source localization" in lower or "source-localization" in lower


def test_followup_pronoun_resolution_preserves_distinctive_terms():
    rewritten = resolve_followup_retrieval_query(
        "Why does its error tend to increase when the dipole moves closer to the first conductivity jump?",
        "Why does the hybrid BE-FE method use BEM for the brain containing the dipoles?",
    )
    assert "hybrid BE-FE method's error" in rewritten
    for term in ("dipole", "first conductivity jump"):
        assert term in rewritten
    assert is_reported_trend_question(rewritten)


def test_cleared_figure_one_stays_ambiguous_across_four_pdfs():
    result = resolve_visual_target("Explain Figure 1.", load_or_build_visual_index())
    assert result.status == "ambiguous"
    assert len({candidate["pdf_name"] for candidate in result.candidates}) >= 4
