from __future__ import annotations

import json
from unittest.mock import patch

import fitz

from services.visual_index import load_or_build_visual_index
from services.visual_locator import (
    VisualResolution,
    clear_visual_target_state,
    resolve_visual_target,
    should_activate_automatic_vision,
)
from services.visual_runtime import analyse_resolved_visual


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
    state = {
        "messages": [{
            "role": "assistant",
            "visual_target": {"pdf_name": "old.pdf", "target_number": "9"},
        }],
        "last_visual_target": {"pdf_name": "old.pdf", "target_number": "9"},
        "pending_visual_resolution": {"status": "resolved"},
        "pending_visual_question": "Explain Figure 9",
        "visual_candidate_choice": "old.pdf",
    }
    state["messages"] = []
    clear_visual_target_state(state)
    resolution = resolve_visual_target(
        "Explain Figure 1.",
        load_or_build_visual_index(),
        conversation_messages=state["messages"],
        current_source_names=[],
        selected_pdf=None,
    )
    assert resolution.status == "ambiguous"
    candidates = {candidate["pdf_name"] for candidate in resolution.candidates}
    assert "hall marks of aging.pdf" in candidates
    assert any(name.startswith("Bioimpedance spectroscopy") for name in candidates)
    assert not should_activate_automatic_vision("Explain Figure 1.", resolution)
    with patch("services.visual_runtime.analyse_pdf_page") as analyse:
        if should_activate_automatic_vision("Explain Figure 1.", resolution):
            analyse_resolved_visual("Explain Figure 1.", resolution)
    analyse.assert_not_called()
