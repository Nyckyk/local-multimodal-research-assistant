from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from services.final_answer_composer import (
    build_figure_final_answer_evidence,
    build_provenance_final_answer_evidence,
    final_evidence_context,
    render_final_answer_evidence,
    validate_answer_consistency,
)
from services.ollama_service import generate_answer
from services.scientific_evidence import experimental_evidence_object, figure_local_evidence
from services.visual_index import flatten_visual_targets, load_or_build_visual_index


PDF_NAME = "Detection of senescence using machine learning algorithms based on nuclear features.pdf"


def _response(content: str):
    return {"message": {"content": content}, "done": True, "done_reason": "stop"}


@pytest.fixture(scope="module")
def figure_evidence():
    questions = {
        "1": (
            "Explain every panel in Figure 1, which nuclear features change, "
            "and the exception."
        ),
        "2": "Compare all conditions and explain the Figure 2 result.",
        "3": (
            "For Figure 3, explain panels C, D, E and G with their exact "
            "correlation statistics, then explain panel I performance metrics."
        ),
        "4": (
            "Using Figure 4, compare the classifier families and performance, "
            "including GM, AEM and the other relevant classifiers."
        ),
        "5": (
            "For Figure 5, how many compounds were screened, how many hits were "
            "specific to each cell line or active in both, and explain the first "
            "experiment and later screening experiment."
        ),
        "6": (
            "Explain Figure 6: how selected candidates were validated, how "
            "senescence was distinguished from toxicity, and the downstream "
            "one-two-punch validation."
        ),
        "7": (
            "Explain Figure 7: how the Tissue Senescence Score was constructed "
            "and validated in the NrasG12V model using nuclear morphology scoring."
        ),
        "9": (
            "Using Figure 9, explain the human NAFLD cohort, how p16INK4a "
            "and TSS were compared, and report the correlation values."
        ),
    }
    targets = [
        row for row in flatten_visual_targets(load_or_build_visual_index())
        if row.get("pdf_name") == PDF_NAME
        and row.get("target_type") == "figure"
        and row.get("match_kind") == "caption"
    ]
    values = {}
    for number, question in questions.items():
        target = next(row for row in targets if row.get("target_number") == number)
        local = figure_local_evidence(
            Path(target["pdf_path"]), target["page_number"],
            target.get("caption_page_number"), number, target["full_caption"], question,
        )
        values[number] = build_figure_final_answer_evidence(
            question=question,
            document=PDF_NAME,
            figure_number=number,
            page=target["page_number"],
            local_evidence=local,
        )
    return values


def test_figure_one_final_evidence_panel_map_is_authoritative(figure_evidence):
    roles = {row["panel"]: row["role"] for row in figure_evidence["1"]["panels"]}
    assert roles == {
        "a": "experimental_design",
        "b": "marker_quantification",
        "c": "marker_quantification",
        "d": "marker_quantification",
        "e": "microscopy",
        "f": "nuclear_feature_distribution",
        "g": "training_workflow",
        "h": "training_result",
        "i": "test_validation_result",
    }


def test_figure_one_display_is_repaired_to_match_evidence(figure_evidence):
    evidence = figure_evidence["1"]
    repaired = render_final_answer_evidence(evidence)
    debug = {}
    with patch(
        "services.ollama_service.ollama.chat",
        side_effect=[
            _response("Panels C-E likely display nuclear morphology."),
            _response(repaired),
        ],
    ) as chat:
        answer = generate_answer(
            evidence["question"], final_evidence_context(evidence), [], debug,
        )
    assert chat.call_count == 2
    assert debug["evidence_repair_used"]
    assert debug["final_evidence_consistency_errors"] == []
    assert "Panel c" in answer and "Brdu" in answer
    assert "Panel e" in answer and "DAPI" in answer


def test_figure_one_display_contains_all_seven_features(figure_evidence):
    answer = render_final_answer_evidence(figure_evidence["1"]).casefold()
    for feature in (
        "nuclear area", "gyration radius", "compactness", "chord ratio",
        "displacement", "elongation", "form factor",
    ):
        assert feature in answer


def test_figure_one_display_preserves_form_factor_exception(figure_evidence):
    answer = render_final_answer_evidence(figure_evidence["1"])
    assert "All except form factor differed significantly" in answer
    assert "etoposide-treated" in answer and "DMSO-treated" in answer


def test_figure_one_display_explains_training_assumption_and_model_types(figure_evidence):
    answer = render_final_answer_evidence(figure_evidence["1"])
    assert "etoposide-treated cells were treated as senescent" in answer
    assert "DMSO-treated cells as normal/non-senescent" in answer
    assert "AEM is the classification-tree-based model" in answer
    assert "AERFM is the random-forest-based model" in answer


def test_figure_one_display_explains_training_and_independent_validation(figure_evidence):
    answer = render_final_answer_evidence(figure_evidence["1"])
    assert "Training result:" in answer
    assert "similar extent as SA-β-Gal staining" in answer
    assert "Independent validation:" in answer and "new test samples" in answer
    assert "no single feature alone was sufficient" in answer


def test_figure_one_rejects_speculative_panel_assignment(figure_evidence):
    errors = validate_answer_consistency(
        "Panels C-E likely display nuclear morphology.", figure_evidence["1"],
    )
    assert "speculative panel assignment despite explicit caption" in errors


def test_figure_two_condition_records_remain_separate(figure_evidence):
    identifiers = {row["id"] for row in figure_evidence["2"]["condition_records"]}
    assert identifiers == {
        "growing", "quiescent", "irradiated_dd",
        "mln8054_senescence", "etoposide_senescence",
    }


def test_figure_two_display_states_etoposide_has_significant_53bp1(figure_evidence):
    answer = render_final_answer_evidence(figure_evidence["2"])
    row = next(line for line in answer.splitlines() if "**etoposide senescence:**" in line)
    assert "53BP1 foci showed a significant increase" in row


def test_figure_two_display_resolves_mln8054_without_significant_damage(figure_evidence):
    answer = render_final_answer_evidence(figure_evidence["2"])
    row = next(line for line in answer.splitlines() if "**MLN8054 senescence:**" in line)
    assert "AEM identified" in row
    assert "no significant detectable DNA-damage increase" in row
    assert "AEM is not merely a detector" in answer


def test_figure_two_display_preserves_irradiated_less_than_30_percent(figure_evidence):
    answer = render_final_answer_evidence(figure_evidence["2"])
    row = next(line for line in answer.splitlines() if "**irradiated (DD):**" in line)
    assert "Less than 30%" in row and "AEM" in row


def test_figure_two_measurement_semantics_do_not_migrate_between_markers(figure_evidence):
    answer = render_final_answer_evidence(figure_evidence["2"])
    assert "53BP1 foci showed cell-cycle arrest" not in answer
    assert "53BP1 foci showed that the cells were arrested" not in answer
    assert "53BP1 foci showed a significant increase (role: DNA-damage marker)" in answer
    assert "BrdU incorporation showed that the cells were dividing" in answer


def test_figure_two_has_no_duplicate_classifier_conclusions(figure_evidence):
    answer = render_final_answer_evidence(figure_evidence["2"])
    assert "AEM prediction established the condition as senescent" not in answer
    irradiated = next(line for line in answer.splitlines() if "**irradiated (DD):**" in line)
    assert irradiated.count("Less than 30%") == 1


def test_figure_two_preserves_relevant_condition_distinctions(figure_evidence):
    answer = render_final_answer_evidence(figure_evidence["2"])
    for condition in (
        "etoposide senescence", "irradiated (DD)", "growing", "MLN8054 senescence",
    ):
        assert f"**{condition}:**" in answer


def test_figure_three_display_keeps_all_requested_correlation_values(figure_evidence):
    answer = render_final_answer_evidence(figure_evidence["3"])
    for value in ("0.8745", "0.9478", "0.8681", "0.9969"):
        assert value in answer
    assert "precision" in answer.casefold() and "accuracy" in answer.casefold()
    assert "recall" in answer.casefold() and "F1" in answer
    assert validate_answer_consistency(answer, figure_evidence["3"]) == []


def test_figure_four_display_keeps_general_model_training_scope(figure_evidence):
    answer = render_final_answer_evidence(figure_evidence["4"])
    assert "GM" in answer
    assert "12 different senescence conditions" in answer
    assert validate_answer_consistency(answer, figure_evidence["4"]) == []


def test_figure_five_display_contains_explicit_stage_one(figure_evidence):
    answer = render_final_answer_evidence(figure_evidence["5"])
    stage = next(line for line in answer.splitlines() if "**Stage 1" in line)
    for term in ("GFP", "mCherry", "DMSO", "ABT-263", "ABT-737", "AEM"):
        assert term in stage
    assert "selectively reduced" in stage


def test_figure_six_display_keeps_quantitative_downstream_outcome(figure_evidence):
    answer = render_final_answer_evidence(figure_evidence["6"])
    assert "75%" in answer
    assert "less than half" in answer.casefold()


def test_figure_seven_display_keeps_construction_and_model_validation(figure_evidence):
    answer = render_final_answer_evidence(figure_evidence["7"])
    assert "nuclear morphology" in answer.casefold()
    assert "percentage of cells" in answer.casefold() and "CSS 1–5" in answer
    assert "higher TSS" in answer
    assert "NRasG12V" in answer and "D38A" in answer


def test_figure_nine_display_preserves_supported_question_acronym(figure_evidence):
    answer = render_final_answer_evidence(figure_evidence["9"])
    assert "TSS" in answer


def test_figure_five_display_contains_explicit_stage_two(figure_evidence):
    answer = render_final_answer_evidence(figure_evidence["5"])
    stage = next(line for line in answer.splitlines() if "**Stage 2" in line)
    assert "A549" in stage and "IMR90" in stage and "GM classifier" in stage
    assert "Toxic candidates" in stage


def test_figure_five_display_preserves_all_grounded_counts(figure_evidence):
    answer = render_final_answer_evidence(figure_evidence["5"])
    for value in ("676", "56", "27", "11", "18"):
        assert value in answer


def test_figure_five_stage_identity_prevents_cross_stage_entity_leakage(figure_evidence):
    answer = render_final_answer_evidence(figure_evidence["5"])
    stage_one = next(line for line in answer.splitlines() if "**Stage 1" in line)
    stage_two = next(line for line in answer.splitlines() if "**Stage 2" in line)
    assert "Stage 1 — Senolytic evaluation" in stage_one
    assert "Stage 2 — Senescence-inducing compound screen" in stage_two
    for term in ("GFP", "mCherry", "AEM", "ABT-263", "ABT-737"):
        assert term in stage_one and term not in stage_two
    for term in ("GM", "676", "56", "27", "11", "18"):
        assert term in stage_two


def test_figure_five_renderer_recovers_caption_defined_stage_boundaries(figure_evidence):
    evidence = json.loads(json.dumps(figure_evidence["5"]))
    evidence["question"] = "How many compounds were screened in Figure 5?"
    evidence["experiment_stages"] = [
        stage for stage in evidence["experiment_stages"] if stage["stage"] == 2
    ]
    evidence["numeric_facts"] = [
        fact for fact in evidence["numeric_facts"] if fact["key"] == "screened"
    ]
    # Reproduce the manual failure: Stage 1 entities reached the only retained
    # stage record. Caption panel boundaries must remain authoritative.
    evidence["experiment_stages"][0]["required_terms"].extend([
        "ABT-263", "ABT-737", "GFP", "mCherry", "AEM",
    ])
    evidence["experiment_stages"][0]["facts"].append(
        "The Stage 2 screen compared A549 and IMR90 and included ABT-263."
    )

    answer = render_final_answer_evidence(evidence)
    stage_one = next(line for line in answer.splitlines() if "**Stage 1" in line)
    stage_two = next(line for line in answer.splitlines() if "**Stage 2" in line)
    assert "Stage 1 — Senolytic evaluation" in stage_one
    assert "Stage 2 — Senescence-inducing compound screen" in stage_two
    for term in ("GFP", "mCherry", "AEM", "ABT-263", "ABT-737"):
        assert term in stage_one and term not in stage_two
    for term in ("A549", "IMR90", "676"):
        assert term in stage_two
    assert validate_answer_consistency(answer, evidence) == []


def test_malformed_or_contradictory_numeric_output_is_rejected(figure_evidence):
    bad = (
        "Finally, **1 was identified in both groups. "
        "Stage 2 says 18 drugs were active in both groups."
    )
    errors = validate_answer_consistency(bad, figure_evidence["5"])
    assert "malformed markdown before numeric clause" in errors
    assert any("contradictory numeric values for both" in error for error in errors)


def test_numeric_values_cannot_be_swapped_between_grounded_fields(figure_evidence):
    bad = render_final_answer_evidence(figure_evidence["5"]).replace(
        "56 total hits", "18 total hits",
    ).replace("18 hits were active in both", "56 hits were active in both")
    errors = validate_answer_consistency(bad, figure_evidence["5"])
    assert any("identified_total contradicts grounded value 56" in error for error in errors)
    assert any("both contradicts grounded value 18" in error for error in errors)


@pytest.fixture(scope="module")
def provenance_evidence():
    rows = []
    for domain, figures in (
        ("in_vitro_human_cell_line", ("1", "3")),
        ("mouse_animal_tissue", ("7", "8")),
        ("human_patient_tissue", ("9",)),
    ):
        for number in figures:
            rows.append({
                "document": PDF_NAME,
                "figure_number": number,
                "panel": None,
                "page": int(number) + 2,
                "experimental_domain": domain,
                "species": "mouse" if "mouse" in domain else "human",
                "sample_type": "resolved sample",
                "source_text": f"Explicit caption for Figure {number}.",
                "source_provenance": ["full_caption"],
            })
    return build_provenance_final_answer_evidence(
        "Which figures support each experimental domain?", rows,
    )


@pytest.fixture(scope="module")
def detailed_provenance_evidence():
    targets = [
        row for row in flatten_visual_targets(load_or_build_visual_index())
        if row.get("pdf_name") == PDF_NAME
        and row.get("target_type") == "figure"
        and row.get("match_kind") == "caption"
        and row.get("target_number") in {"1", "3", "7", "8", "9"}
    ]
    rows = [
        experimental_evidence_object(
            PDF_NAME, row["target_number"], row["page_number"],
            caption=row["full_caption"],
        )
        for row in sorted(targets, key=lambda item: int(item["target_number"]))
    ]
    return build_provenance_final_answer_evidence(
        "Which figures provide the strongest evidence from cell culture, mouse animal "
        "and human patient experiments, and what distinct evidence does each contribute?",
        rows,
    )


def test_whole_document_evidence_retains_exact_provenance(provenance_evidence):
    mapping = {}
    for row in provenance_evidence["provenance"]:
        mapping.setdefault(row["experimental_domain"], []).append(row["figure_number"])
    assert mapping == {
        "in_vitro_human_cell_line": ["1", "3"],
        "mouse_animal_tissue": ["7", "8"],
        "human_patient_tissue": ["9"],
    }


def test_whole_document_display_uses_provenance_figure_numbers(provenance_evidence):
    answer = render_final_answer_evidence(provenance_evidence)
    assert "Figure 1 and Figure 3" in answer
    assert "Figure 7 and Figure 8" in answer
    assert "Figure 9" in answer
    assert validate_answer_consistency(answer, provenance_evidence) == []


def test_whole_document_renderer_explains_each_selected_figure(detailed_provenance_evidence):
    answer = render_final_answer_evidence(detailed_provenance_evidence)
    assert "Figure 1" in answer and "training sets for the AEM and AERFM" in answer
    assert "Figure 3" in answer and "single-cell level" in answer
    assert "Figure 7" in answer and "liver cancer initiation" in answer
    assert "Figure 8" in answer and "senolytic drugs" in answer
    assert "liver fibrosis and aging" in answer
    assert "Figure 9" in answer and "34 patients" in answer
    assert "p16INK4a" in answer


def test_whole_document_renderer_preserves_three_domains_and_selected_figures(
    detailed_provenance_evidence,
):
    answer = render_final_answer_evidence(detailed_provenance_evidence)
    for heading in (
        "In vitro human cell-line evidence", "Mouse animal/tissue evidence",
        "Human patient-tissue evidence",
    ):
        assert heading in answer
    assert validate_answer_consistency(answer, detailed_provenance_evidence) == []
    referenced = set(__import__("re").findall(r"\bFigure\s+(\d+)\b", answer))
    assert referenced == {"1", "3", "7", "8", "9"}


def test_resolved_provenance_never_displays_unresolved_figure(provenance_evidence):
    answer = render_final_answer_evidence(provenance_evidence)
    assert "figure number not resolved" not in answer.casefold()
    errors = validate_answer_consistency(
        answer + " Figure number not resolved.", provenance_evidence,
    )
    assert "reports unresolved figure despite resolved provenance" in errors


def test_unverified_supplementary_figure_is_rejected(provenance_evidence):
    answer = render_final_answer_evidence(provenance_evidence) + " Mouse evidence is in Figure S10."
    errors = validate_answer_consistency(answer, provenance_evidence)
    assert any("unverified supplementary figure" in error for error in errors)


def test_unselected_main_figure_number_is_rejected(provenance_evidence):
    answer = render_final_answer_evidence(provenance_evidence) + " Figure 10 was also used."
    errors = validate_answer_consistency(answer, provenance_evidence)
    assert "unselected figure number: Figure 10" in errors


def test_bad_whole_document_prose_falls_back_to_provenance_renderer(provenance_evidence):
    context = (
        "[WHOLE-DOCUMENT EXPERIMENTAL-DOMAIN EVIDENCE]\n"
        + json.dumps(provenance_evidence["provenance"])
    )
    debug = {}
    with patch(
        "services.ollama_service.ollama.chat",
        side_effect=[
            _response("The figures are unresolved; perhaps Supplementary Figure S10."),
            _response("Figure number not resolved."),
        ],
    ):
        answer = generate_answer(
            provenance_evidence["question"], context, [], debug,
        )
    assert debug["deterministic_evidence_fallback_used"]
    assert "Figure 1 and Figure 3" in answer
    assert "Figure 7 and Figure 8" in answer
    assert "Figure 9" in answer
    assert "not resolved" not in answer.casefold()
    assert "S10" not in answer
