from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import fitz

from services.visual_index import load_or_build_visual_index
from services.equation_service import analyse_resolved_equation
from services.visual_locator import (
    VisualResolution,
    clear_visual_conversation_state,
    resolve_visual_target,
    should_activate_automatic_vision,
)
from services.visual_runtime import analyse_resolved_visual, build_visual_evidence


QUESTION = (
    "Using Figure 3, explain what RE, RI and C represent and how they are "
    "connected."
)


def _malformed_runtime_circuit() -> dict:
    return {
        "diagram_kind": "circuit",
        "labels": ["ECF", "ICF", "Cell membrane", "RE", "RI", "C"],
        "components": [
            {"name": "RE", "description": "extracellular-fluid resistance"},
            {"name": "RI", "description": "intracellular-fluid resistance"},
            {"name": "C", "description": "cell-membrane capacitance"},
        ],
        "spatial_relationships": [
            {"subject": "RI", "relationship": "is in series with", "object": "C"},
            {
                "subject": "RE",
                "relationship": "is connected in parallel to the RI-C branch",
            },
        ],
        "connections": [{
            "from": "C", "to": "RI", "relationship": "series branch",
        }],
        "circuit_topology": {
            "nodes": [
                {"id": "ECF terminal", "label": "ECF terminal"},
                {"id": "ICF terminal", "label": "ICF terminal"},
                {
                    "id": "Cell membrane terminal",
                    "label": "Cell membrane terminal",
                },
            ],
            "edges": [
                {
                    "from_node": "ECF terminal", "to_node": "ICF terminal",
                    "component": "RE",
                },
                {
                    "from_node": "ICF terminal",
                    "to_node": "Cell membrane terminal", "component": "C",
                },
            ],
            "branches": [{
                "id": "branch_RI_C", "start_node": "ICF terminal",
                "end_node": "Cell membrane terminal", "components": ["C", "RI"],
            }],
            "parallel_branch_sets": [],
        },
        "explanation": (
            "RE represents extracellular-fluid resistance, RI represents "
            "intracellular-fluid resistance, and C represents cell-membrane "
            "capacitance."
        ),
        "uncertain_items": [],
    }


def _runtime_pdf(path) -> None:
    document = fitz.open()
    page = document.new_page()
    page.insert_textbox(
        fitz.Rect(72, 72, 520, 300),
        "Fig. 3. Equivalent circuit representation of biological tissue.\n"
        "The extracellular fluid (ECF) is modeled by resistor RE. The "
        "intracellular fluid (ICF) and cell membrane are represented by a "
        "series resistor RI and capacitor C, respectively.\n"
        "R_infinity = (RI * RE) / (RI + RE).",
        fontsize=11,
    )
    document.save(path)
    document.close()


def test_streamlit_runtime_repairs_exact_malformed_figure_three(tmp_path):
    pdf_path = tmp_path / "runtime-figure-3.pdf"
    _runtime_pdf(pdf_path)
    resolution = VisualResolution(
        status="resolved",
        pdf_path=str(pdf_path),
        pdf_name=pdf_path.name,
        page_number=1,
        target_type="figure",
        target_number="3",
        visual_type="labelled_diagram",
        caption="Fig. 3. Equivalent circuit representation of biological tissue.",
    )
    debug = {}
    with patch(
        "services.structured_vision._call_model",
        return_value=json.dumps(_malformed_runtime_circuit()),
    ) as model:
        answer = analyse_resolved_visual(
            QUESTION, resolution, debug_info=debug
        )

    topology = debug["validated_json"]["circuit_topology"]
    normalized_edges = [
        "".join(character for character in edge["component"].lower() if character.isalnum())
        for edge in topology["edges"]
    ]
    assert sorted(normalized_edges) == ["c", "re", "ri"]
    series = next(
        branch for branch in topology["branches"]
        if {
            "".join(character for character in item.lower() if character.isalnum())
            for item in branch["components"]
        } == {"ri", "c"}
    )
    direct = next(
        branch for branch in topology["branches"]
        if len(branch["components"]) == 1
    )
    assert (series["start_node"], series["end_node"]) == (
        direct["start_node"], direct["end_node"]
    )
    assert {series["id"], direct["id"]} in [
        set(branch_set) for branch_set in topology["parallel_branch_sets"]
    ]
    assert debug["retry_kind"] == "grounded_topology_repair"
    assert debug["final_answer_path"] == "validated_typed_vision"
    assert debug["final_answer_code_path"] == "validated_repaired_structured_vision"
    assert "Could not verify the circuit topology" not in answer
    assert "series" in answer.lower() and "parallel" in answer.lower()
    assert model.call_count == 1


def test_cleared_conversation_keeps_real_figure_one_ambiguous():
    visual_index = load_or_build_visual_index()
    figure_three = resolve_visual_target(QUESTION, visual_index)
    assert figure_three.status == "resolved"
    assert figure_three.pdf_name.startswith("Bioimpedance spectroscopy")

    state = {
        "messages": [{
            "role": "assistant",
            "content": "Validated Figure 3 analysis.",
            "visual_target": figure_three.to_dict(),
            "sources": [{"pdf": figure_three.pdf_name}],
        }],
        "last_user_question": QUESTION,
        "last_visual_target": figure_three.to_dict(),
        "pending_visual_resolution": {"status": "resolved"},
        "pending_visual_question": QUESTION,
        "visual_candidate_choice": figure_three.pdf_name,
        "preferred_pdf_name": figure_three.pdf_name,
        "previous_pdf": figure_three.pdf_name,
        "previous_page": figure_three.page_number,
        "previous_figure": "3",
        "previous_panel": "a",
        "previous_visual_target": figure_three.to_dict(),
        "previous_source_context": [figure_three.pdf_name],
        "target_resolution": figure_three.to_dict(),
        "automatic_detection_candidates": figure_three.candidates,
        "pending_ambiguity_selection": figure_three.pdf_name,
        "rewritten_visual_query_context": QUESTION,
        "vision_result_1": "old result",
        "raw_vision_1": "old raw response",
    }
    clear_visual_conversation_state(state)
    state["preferred_pdf_name"] = state.pop("pending_preferred_pdf_name")
    resolution = resolve_visual_target(
        "Explain Figure 1.",
        visual_index,
        conversation_messages=state["messages"],
        current_source_names=[],
        selected_pdf=(
            None if state["preferred_pdf_name"] == "No preference"
            else state["preferred_pdf_name"]
        ),
    )
    assert resolution.status == "ambiguous"
    candidates = {candidate["pdf_name"] for candidate in resolution.candidates}
    assert "hall marks of aging.pdf" in candidates
    assert any(name.startswith("Bioimpedance spectroscopy") for name in candidates)
    reasons = " ".join([
        resolution.reason,
        *[candidate["reason"] for candidate in resolution.candidates],
    ]).lower()
    assert "previous visual pdf" not in reasons
    assert "conversation source context" not in reasons
    assert not should_activate_automatic_vision("Explain Figure 1.", resolution)
    with patch("services.visual_runtime.analyse_pdf_page") as analyse:
        if should_activate_automatic_vision("Explain Figure 1.", resolution):
            analyse_resolved_visual("Explain Figure 1.", resolution)
    analyse.assert_not_called()


def test_uncleared_conversation_keeps_genuine_visual_followup_context():
    visual_index = load_or_build_visual_index()
    figure_three = resolve_visual_target(QUESTION, visual_index)
    messages = [{
        "role": "assistant",
        "content": "Validated Figure 3 analysis.",
        "visual_target": figure_three.to_dict(),
        "sources": [{"pdf": figure_three.pdf_name}],
    }]
    followup = "Explain panel b."
    resolution = resolve_visual_target(
        followup,
        visual_index,
        conversation_messages=messages,
        current_source_names=[figure_three.pdf_name],
    )
    assert resolution.status == "resolved"
    assert resolution.pdf_name == figure_three.pdf_name
    assert resolution.target_number == "3"
    assert resolution.panel == "b"
    assert "previous visual PDF" in resolution.reason
    assert "conversation source context" in resolution.reason
    assert should_activate_automatic_vision(followup, resolution)


def test_equation_eight_uses_text_path_and_never_invokes_figure_vision():
    index = load_or_build_visual_index()
    thermal_name = next(
        name for name in index["files"] if name.startswith("Thermal wave and Pennes")
    )
    previous = [{
        "role": "assistant",
        "content": "Prior Figure 5 answer.",
        "visual_target": {
            "status": "resolved", "target_type": "figure", "target_number": "5",
            "pdf_name": thermal_name,
            "pdf_path": str(Path("papers") / thermal_name),
            "page_number": 9,
        },
    }]
    question = (
        "Explain Equation 8 and show how it reduces to the Pennes equation when "
        "relaxation time is zero."
    )
    resolution = resolve_visual_target(
        question, index, conversation_messages=previous
    )
    assert (resolution.status, resolution.target_type, resolution.page_number) == (
        "resolved", "equation", 6,
    )

    debug = {}
    with patch(
        "services.equation_service.generate_answer"
    ) as generate, patch(
        "services.visual_runtime.analyse_pdf_page"
    ) as vision:
        answer = analyse_resolved_equation(
            question, resolution, conversation_history=previous, debug_info=debug
        )

    assert "Complete Equation 8" in answer
    assert "Q_{met}=0" in answer
    assert "second-time-derivative term" in answer
    assert "time derivative of the external source" in answer
    assert r"\rho c\frac{\partial T}{\partial t}" in answer
    assert r"+\rho_b c_b\omega_b(T_b-T)+Q_{ext}" in answer
    assert "c_p" not in answer and "T_a" not in answer
    assert "steady-state" not in answer.casefold()
    generate.assert_not_called()
    vision.assert_not_called()
    assert debug["final_answer_code_path"] == "validated_text_equation_analysis"
    assert debug["grounded_zero_reduction_supplemented"] is False
    symbolic = debug["symbolic_equation"]
    assert symbolic["sign_validation"] == "passed"
    assert symbolic["paper_symbols"] == {
        "specific_heat": "c", "blood_temperature": "T_b",
    }
    assert [(term["kind"], term["sign"]) for term in symbolic["target_terms"]] == [
        ("second_time_derivative", "+"),
        ("conduction", "+"),
        ("tissue_temperature_perfusion", "-"),
        ("first_time_derivative", "-"),
        ("blood_temperature_perfusion", "+"),
        ("external_source", "+"),
        ("external_source_time_derivative", "+"),
    ]
    assert [(term["kind"], term["sign"]) for term in symbolic["rearranged_terms"]] == [
        ("first_time_derivative", "+"),
        ("conduction", "+"),
        ("blood_minus_tissue_perfusion", "+"),
        ("external_source", "+"),
    ]


def test_streamlit_runtime_accepts_grounded_coordinate_boundary_endpoints():
    question = (
        "Using Figure 2 in the thermal-wave paper, explain all boundary conditions "
        "applied to the skin model."
    )
    resolution = resolve_visual_target(question, load_or_build_visual_index())
    assert resolution.status == "resolved", resolution.to_dict()
    assert resolution.pdf_name.startswith("Thermal wave and Pennes")
    assert resolution.page_number == 5
    raw = {
        "diagram_kind": "other",
        "labels": [
            "Wave port boundary condition", "Scattering boundary condition",
            "Thermal insulation condition", "Data extraction line",
        ],
        "components": [{"name": "Skin model", "description": "2D layered domain"}],
        "spatial_relationships": [],
        "connections": [
            {"from": "Left edge of the model (x=0)", "to": "Skin model", "relationship": "bounds"},
            {"from": "Right edge of the model (x=W)", "to": "Skin model", "relationship": "bounds"},
            {"from": "Bottom edge of the model (y=0)", "to": "Skin model", "relationship": "bounds"},
            {"from": "Top edge of the model (y=H)", "to": "Skin model", "relationship": "bounds"},
        ],
        # Deliberately omit circuit_topology: Figure 2 is not a circuit.
        "explanation": "The figure labels electromagnetic and thermal boundaries.",
        "uncertain_items": [],
    }
    debug = {}
    with patch(
        "services.structured_vision._call_model", return_value=json.dumps(raw)
    ):
        answer = analyse_resolved_visual(question, resolution, debug_info=debug)

    assert debug["validated_json"]["circuit_topology"] is None
    assert debug["final_answer_code_path"] == "validated_structured_vision"
    for phrase in (
        "wave-port", "TM microwave", "scattering", "applied microwave heat flux",
        "thermal insulation", "internal sampling",
    ):
        assert phrase in answer
    assert answer.count("**Electromagnetic boundary conditions**") == 1
    assert answer.count("**Thermal boundary conditions**") == 1
    assert "same geometric edge" in answer
    assert "could not verify" not in answer.casefold()
    assert "ASSOCIATED EQUATIONS/NEARBY TEXT" not in answer
    assert [item["from"] for item in debug["validated_json"]["connections"]] == [
        "x = 0", "x = W", "y = 0", "y = H",
    ]


def test_streamlit_runtime_uses_validated_text_table_after_empty_vision_output():
    question = (
        "Extract Table 3 from the thermal paper and explain why the authors "
        "selected the Extra Fine mesh."
    )
    resolution = resolve_visual_target(question, load_or_build_visual_index())
    assert (resolution.status, resolution.page_number) == ("resolved", 8)
    debug = {}
    with patch("services.structured_vision._call_model", return_value="{}"):
        answer = analyse_resolved_visual(question, resolution, debug_info=debug)

    table = debug["validated_json"]
    assert debug["final_answer_code_path"] == "validated_text_table_fallback"
    assert len(table["columns"]) == 6
    assert len(table["rows"]) == 3
    assert table["unreadable_cells"] == []
    assert "Extra Fine was selected" in answer
    assert "refinement" in answer
    assert table is debug["repaired_json"]
    assert table is debug["final_structured_output"]
    assert table is debug["rendered_structured_object"]
    evidence = build_visual_evidence(resolution, answer, debug)
    assert evidence["structured_result"] is table


def test_matching_vision_table_keeps_grounded_text_selection_explanation():
    question = (
        "Extract Table 3 from the thermal paper and explain why the authors "
        "selected the Extra Fine mesh."
    )
    resolution = resolve_visual_target(question, load_or_build_visual_index())
    vision_table = {
        "table_number": "3",
        "title": "Grid test",
        "columns": [
            "Mesh type", "Normal", "Fine", "Finer", "Extra Fine",
            "Extremely Fine",
        ],
        "rows": [
            ["Degrees of freedom", "20463", "22546", "29507", "58907", "182970"],
            ["Elements", "2072", "2277", "2938", "5890", "18257"],
            ["SAR (W/Kg)", "357.5132", "357.8341", "358.0154", "358.1013", "358.1014"],
        ],
        "units": {"SAR (W/Kg)": "W/Kg"},
        "comparisons": ["SAR converges with mesh refinement."],
        "unreadable_cells": [],
    }
    debug = {}
    with patch(
        "services.structured_vision._call_model",
        return_value=json.dumps(vision_table),
    ):
        answer = analyse_resolved_visual(question, resolution, debug_info=debug)

    assert debug["final_answer_code_path"] in {
        "validated_typed_vision", "validated_structured_vision",
    }
    assert debug["text_table_cross_check"]["matched"] is True
    assert debug["text_table_details_merged"] is True
    assert "0.0001 W/kg" in answer
    assert "58907" in answer and "182970" in answer
    table = debug["validated_json"]
    assert table is debug["repaired_json"]
    assert table is debug["final_structured_output"]
    assert table is debug["rendered_structured_object"]
