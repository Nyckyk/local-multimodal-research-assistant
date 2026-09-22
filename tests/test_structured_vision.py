import importlib
import json
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


sys.modules.setdefault("ollama", types.SimpleNamespace(chat=lambda **kwargs: None))
structured = importlib.import_module("services.structured_vision")


def test_explicit_panel_identity_normalization():
    for raw, expected in [("a", "a"), ("(a)", "a"), ("(a) Sample 10", "a"),
                          ("Panel a", "a"), ("A", "a"), ("(b)", "b")]:
        panel = {"panel": raw, "group": "existing group"}
        structured.normalize_panel_identity(panel)
        assert panel["panel"] == expected
        assert panel["group"] == "existing group"
        if raw == "(a) Sample 10":
            assert panel["panel_title"] == "Sample 10"
    for raw in ["Sample 10", "arbitrary text", "Panel alpha"]:
        panel = {"panel": raw}
        structured.normalize_panel_identity(panel)
        assert panel["panel"] == raw


def test_magnitude_tie_never_becomes_input_order():
    panels = [{"panel": "a", "group": "One", "graph_kind": "magnitude", "approximate_curve_level": 115},
              {"panel": "c", "group": "Two", "graph_kind": "magnitude", "approximate_curve_level": 115}]
    assert structured._comparison_expectations(panels)[0] == []


def magnitude_record(panel, group, values, relative=False, model_identity=None):
    task = {"source_panel": panel, "source_group": group,
            "y_min": 0, "y_max": 100, "y_scale": "linear"}
    payload = {"panel": model_identity, "group": model_identity, "readings": [
        {"x": x, "y": None if relative else value,
         "vertical_position": value if relative else None, "readability": True, "confidence": .9}
        for x, value in zip([100, 1000], values)]}
    return structured._source_magnitude_readings(task, payload, [100, 1000])


def test_magnitude_source_identity_survives_missing_or_duplicate_model_identity():
    records = [magnitude_record("a", "One", [60, 50]),
               magnitude_record("c", "Two", [30, 20], model_identity="One"),
               magnitude_record("e", "Three", [90, 80], model_identity="One")]
    assert [r["source_panel"] for r in records] == ["a", "c", "e"]
    assert [r["source_group"] for r in records] == ["One", "Two", "Three"]
    assert structured.magnitude_reading_order(records) == ["Three", "One", "Two"]


def test_shared_relative_readings_rank_only_when_consistent():
    records = [magnitude_record("x", "First", [.6, .5], True),
               magnitude_record("y", "Second", [.3, .2], True),
               magnitude_record("z", "Third", [.9, .8], True)]
    assert structured.magnitude_reading_order(records) == ["Third", "First", "Second"]
    tied = [magnitude_record("x", "First", [60, 50]), magnitude_record("y", "Second", [60, 50])]
    assert structured.magnitude_reading_order(tied) == []
    conflicting = [magnitude_record("x", "First", [60, 20]), magnitude_record("y", "Second", [30, 50])]
    assert structured.magnitude_reading_order(conflicting) == []


def test_unreadable_magnitude_retry_stays_uncertain(tmp_path):
    from PIL import Image
    path = tmp_path / "panel.png"
    Image.new("RGB", (100, 100)).save(path)
    panels = [{"panel": "a", "group": "One", "graph_kind": "magnitude", "approximate_curve_level": 115,
               "y_axis": {"scale": "linear"}}]
    axis = json.dumps({"x_min": 10, "x_max": 10000, "y_min": 0, "y_max": 140, "confidence": .9})
    with patch.object(structured, "_call_model", side_effect=[axis, '{"readings": []}', '{"readings": []}']) as call:
        _, errors = structured.verify_magnitude_levels([path], panels, [["a"]])
    assert call.call_count == 3
    assert errors
    assert panels[0]["approximate_curve_level"] is None


def axis(label, unit, scale="linear", ticks=None, multiplier=None):
    return {
        "label": label,
        "unit": unit,
        "scale": scale,
        "tick_labels": ticks or ["0", "1", "2"],
        "scientific_multiplier": multiplier,
    }


def graph_panel(
    label, kind, group, x_axis, y_axis, minimum, maximum,
    complexity=0.2, trends=None,
):
    return {
        "panel": label,
        "graph_kind": kind,
        "group": group,
        "x_axis": x_axis,
        "y_axis": y_axis,
        "series": [group] if group else [],
        "visible_range": {
            "min": minimum, "max": maximum,
            "unit": y_axis["unit"], "confidence": 0.9,
        },
        "shape_features": ["smooth"],
        "complexity_score": complexity,
        "visible_trends": trends or ["Visible trend"],
    }


def comparison(claim, subject, relation, metric):
    return {
        "claim": claim,
        "subject": subject,
        "relation": relation,
        "metric": metric,
        "evidence": ["vision"],
        "confidence": 0.9,
        "uncertain": False,
        "evidence_conflict": False,
    }


def compact_pair(first, second, group, magnitude_level, phase_complexity):
    return {
        "panels": [
            {
                "panel": first, "group": group, "graph_kind": "magnitude",
                "x_axis": {"label": "Frequency", "unit": "Hz", "scale": "log"},
                "y_axis": {"label": "|Zfat|", "unit": "Ω", "scale": "linear"},
                "series": [group], "visible_trend": "Magnitude decreases with frequency.",
                "approximate_curve_level": magnitude_level, "complexity_score": 0.2,
            },
            {
                "panel": second, "group": group, "graph_kind": "phase",
                "x_axis": {"label": "Frequency", "unit": "Hz", "scale": "log"},
                "y_axis": {"label": "Phase", "unit": "°", "scale": "linear"},
                "series": [group], "visible_trend": "Phase changes across frequency.",
                "approximate_curve_level": -45, "complexity_score": phase_complexity,
            },
        ],
        "uncertain_values": [],
    }


def circuit_result():
    return {
        "diagram_kind": "circuit",
        "labels": ["ECF", "ICF", "Cell membrane", "RE", "RI", "C"],
        "components": [
            {"name": "RE", "description": "extracellular resistance"},
            {"name": "RI", "description": "intracellular resistance"},
            {"name": "C", "description": "cell membrane capacitance"},
        ],
        "spatial_relationships": [],
        "connections": [
            {"from": "RI", "to": "C", "relationship": "series"},
            {"from": "branch_1", "to": "branch_2", "relationship": "parallel"},
            {"from": "ICF", "to": "RI", "relationship": "represented by"},
        ],
        "circuit_topology": {
            "nodes": [
                {"id": "n_start", "label": "shared start junction"},
                {"id": "n_mid", "label": "series junction"},
                {"id": "n_end", "label": "shared end junction"},
            ],
            "edges": [
                {"from_node": "n_start", "to_node": "n_end", "component": "RE"},
                {"from_node": "n_start", "to_node": "n_mid", "component": "C"},
                {"from_node": "n_mid", "to_node": "n_end", "component": "RI"},
            ],
            "branches": [
                {"id": "branch_1", "start_node": "n_start", "end_node": "n_end", "components": ["RE"]},
                {"id": "branch_2", "start_node": "n_start", "end_node": "n_end", "components": ["C", "RI"]},
            ],
            "parallel_branch_sets": [["branch_1", "branch_2"]],
        },
        "explanation": "RE is parallel with the series branch containing C and RI.",
        "uncertain_items": [],
    }


class StructuredVisionTests(unittest.TestCase):
    def test_local_runner_crash_retries_same_request_once(self):
        recovered = {"message": {"content": '{"recovered": true}'}}
        with patch.object(
            structured.ollama,
            "chat",
            side_effect=[
                RuntimeError("model runner has unexpectedly stopped"),
                recovered,
            ],
        ) as chat, patch.object(structured.time, "sleep") as sleep:
            raw = structured._call_model_images(
                [Path("panel.png")], "inspect", 100
            )
        self.assertEqual(raw, '{"recovered": true}')
        self.assertEqual(chat.call_count, 2)
        self.assertEqual(chat.call_args_list[0].kwargs, chat.call_args_list[1].kwargs)
        sleep.assert_called_once_with(1)

    def test_empty_multi_image_response_retries_identical_request_once(self):
        with patch.object(
            structured,
            "_chat_with_runner_retry",
            side_effect=[
                {"message": {"content": ""}},
                {"message": {"content": '{"label": "|Zfat|"}'}},
            ],
        ) as chat:
            raw = structured._call_model_images(
                [Path("axis_a.png"), Path("axis_c.png")],
                "Read the label.",
                220,
            )
        self.assertEqual(raw, '{"label": "|Zfat|"}')
        self.assertEqual(chat.call_count, 2)
        self.assertEqual(chat.call_args_list[0].kwargs, chat.call_args_list[1].kwargs)

    def test_figure_one_detects_labelled_diagram(self):
        page_text = "Fig. 1. Histological and schematic representation of adipose tissue structure."
        self.assertEqual(
            structured.detect_visual_type("Explain Figure 1.", page_text),
            "labelled_diagram",
        )

    def test_figure_one_non_circuit_schema_is_unchanged(self):
        result = {
            "diagram_kind": "other",
            "labels": ["Adipocyte", "Connective tissue", "Nucleus"],
            "components": [{
                "name": "adipocyte and surrounding tissue",
                "description": "visible biological structure",
            }],
            "spatial_relationships": [],
            "connections": [],
            "circuit_topology": None,
            "explanation": "A labelled adipose-tissue diagram.",
            "uncertain_items": [],
        }
        self.assertEqual(
            structured.validate_labelled_diagram(result)["diagram_kind"],
            "other",
        )

    def test_non_circuit_may_omit_optional_circuit_topology(self):
        result = {
            "diagram_kind": "other",
            "labels": ["Boundary"],
            "components": [],
            "spatial_relationships": [],
            "connections": [],
            "explanation": "A non-circuit boundary diagram.",
            "uncertain_items": [],
        }
        validated = structured.validate_labelled_diagram(result)
        self.assertIsNone(validated["circuit_topology"])

    def test_circuit_still_requires_strict_topology_when_field_is_omitted(self):
        result = {
            "diagram_kind": "circuit",
            "labels": ["R1"],
            "components": [{"name": "R1", "description": "resistor"}],
            "spatial_relationships": [],
            "connections": [],
            "explanation": "A circuit.",
            "uncertain_items": [],
        }
        with self.assertRaisesRegex(
            structured.StructuredOutputError, "circuit_topology"
        ):
            structured.validate_labelled_diagram(result)

    def test_boundary_enrichment_uses_grounded_conditions_and_excludes_sampling_line(self):
        value = {
            "diagram_kind": "other",
            "labels": ["Wave port boundary", "Data extraction line"],
            "components": [], "spatial_relationships": [], "connections": [],
            "explanation": "The figure shows the domain.",
            "uncertain_items": [], "circuit_topology": None,
        }
        evidence = (
            "Scattering boundary conditions prevent undesired reflections. "
            "Wave port boundary condition governs the incident plane microwaves. "
            "At x = 0, -k dT(0,y,t)/dx = qmw . (9) "
            "At x = W, dT(W,y,t)/dx = 0 . (10) "
            "At y = 0, dT(x,0,t)/dy = 0 . (11) "
            "At y = H, dT(x,H,t)/dy = 0 . (12) Data extraction line"
        )
        enriched = structured.enrich_boundary_conditions(
            value, evidence, "Explain all boundary conditions."
        )
        explanation = enriched["explanation"]
        for phrase in (
            "wave-port", "scattering", "x = 0", "x = W", "y = 0", "y = H",
            "applied microwave heat flux", "thermal insulation", "internal sampling",
        ):
            self.assertIn(phrase, explanation)

    def test_grounded_coordinate_boundaries_are_declared_for_non_circuit_diagram(self):
        result = {
            "diagram_kind": "other",
            "labels": ["Skin model", "Wave port", "Thermal insulation"],
            "components": [{"name": "Skin model", "description": "2D domain"}],
            "spatial_relationships": [{
                "subject": "Top edge of model (y=H)",
                "relationship": "bounds",
                "object": "Skin model",
            }],
            "connections": [
                {"from": "Left edge of model (x=0)", "to": "Skin model", "relationship": "bounds"},
                {"from": "Right edge of model (x=W)", "to": "Skin model", "relationship": "bounds"},
                {"from": "Bottom edge of model (y=0)", "to": "Skin model", "relationship": "bounds"},
                {"from": "Top edge of model (y=H)", "to": "Skin model", "relationship": "bounds"},
            ],
            "circuit_topology": None,
            "explanation": "The coordinate edges bound the skin model.",
            "uncertain_items": [],
        }
        evidence = "The domain extends over x=0, x=W, y=0 and y=H."
        validated = structured.validate_labelled_diagram(result, evidence)
        self.assertEqual(
            [connection["from"] for connection in validated["connections"]],
            ["x = 0", "x = W", "y = 0", "y = H"],
        )
        self.assertTrue(
            {"x = 0", "x = W", "y = 0", "y = H"}.issubset(validated["labels"])
        )
        self.assertEqual(validated["spatial_relationships"][0]["subject"], "y = H")

    def test_boundary_coordinate_normalisation_does_not_touch_circuits(self):
        circuit = {
            "diagram_kind": "circuit",
            "connections": [{"from": "x=0", "to": "R1"}],
        }
        self.assertIs(
            structured.normalize_boundary_diagram_endpoints(circuit, "x=0"),
            circuit,
        )

    def test_grounded_boundary_fallback_is_concise_and_equation_complete(self):
        evidence = (
            "The plane wave is transverse magnetic (TM mode). The wave-port boundary "
            "governs the incident field. Scattering boundary conditions prevent "
            "undesired reflections. At x = 0, -k dT(0,y,t)/dx = qmw . (9) "
            "At x = W, dT(W,y,t)/dx = 0 . (10) At y = 0, "
            "dT(x,0,t)/dy = 0 . (11) At y = H, dT(x,H,t)/dy = 0 . (12) "
            "Data extraction line"
        )
        answer = structured.grounded_boundary_synthesis(evidence)
        for phrase in (
            "TM microwave field", "x=0", "x=W", "y=0", "y=H", "internal sampling",
        ):
            self.assertIn(phrase, answer)
        self.assertNotIn("could not verify", answer.casefold())

    def test_boundary_rendering_separates_physics_and_suppresses_duplicate_explanation(self):
        value = {
            "diagram_kind": "other",
            "labels": ["Wave port", "Scattering boundary", "Data extraction line"],
            "components": [],
            "spatial_relationships": [],
            "connections": [
                {"from": "x = 0", "relationship": "bounds", "to": "Skin model"},
                {"from": "Data extraction line", "relationship": "samples", "to": "Skin model"},
            ],
            "circuit_topology": None,
            "explanation": (
                "On the top and bottom edges, scattering boundary conditions prevent "
                "reflections. The complete thermal explanation is repeated here."
            ),
            "uncertain_items": [],
        }
        evidence = (
            "The wave-port boundary governs the incident plane microwaves in TM mode. "
            "Scattering boundary conditions prevent undesired reflections. "
            "At x = 0, -k dT(0,y,t)/dx = qmw . (9) At x = W, "
            "dT(W,y,t)/dx = 0 . (10) At y = 0, dT(x,0,t)/dy = 0 . (11) "
            "At y = H, dT(x,H,t)/dy = 0 . (12) Data extraction line"
        )
        rendered = structured.format_boundary_condition_result(value, evidence)
        self.assertEqual(rendered.count("**Electromagnetic boundary conditions**"), 1)
        self.assertEqual(rendered.count("**Thermal boundary conditions**"), 1)
        self.assertIn("incident TM microwave field", rendered)
        self.assertIn("artificial electromagnetic reflections", rendered)
        self.assertIn("same geometric edge", rendered)
        for coordinate in ("$x=0$", "$x=W$", "$y=0$", "$y=H$"):
            self.assertIn(coordinate, rendered)
        self.assertEqual(rendered.count("data-extraction line"), 1)
        self.assertIn("**Internal sampling**", rendered)
        self.assertNotIn("**Structured connections:**", rendered)
        self.assertNotIn("On the top and bottom edges", rendered)
        self.assertNotIn("x = 0 â€” bounds", rendered)

    def test_non_circuit_relationship_endpoint_can_be_grounded_in_caption(self):
        result = {
            "diagram_kind": "other",
            "labels": ["Intravascular fluid"],
            "components": [{"name": "Intravascular fluid", "description": "fluid"}],
            "spatial_relationships": [{
                "subject": "Intravascular fluid",
                "relationship": "within",
                "object": "blood vessel",
            }],
            "connections": [{
                "from": "Intravascular fluid",
                "to": "blood vessel",
                "relationship": "within",
            }],
            "circuit_topology": None,
            "explanation": "Fluid within a vessel.",
            "uncertain_items": [],
        }
        validated = structured.validate_labelled_diagram(
            result,
            "Intravascular fluid is within blood vessels.",
        )
        self.assertEqual(validated["connections"][0]["to"], "blood vessel")

    def test_non_circuit_view_qualifiers_normalize_to_visible_labels(self):
        result = {
            "diagram_kind": "other",
            "labels": ["Adipocyte", "Intracellular fluid"],
            "components": [
                {"name": "Adipocyte", "description": "cell"},
                {"name": "Intracellular fluid", "description": "fluid"},
            ],
            "spatial_relationships": [],
            "connections": [{
                "from": "Adipocyte (microscopic image)",
                "to": "Intracellular fluid (schematic diagram)",
                "relationship": "contains",
            }],
            "circuit_topology": None,
            "explanation": "The image corresponds to the schematic.",
            "uncertain_items": [],
        }
        validated = structured.validate_labelled_diagram(result)
        self.assertEqual(
            validated["connections"][0],
            {
                "from": "Adipocyte",
                "to": "Intracellular fluid",
                "relationship": "contains",
            },
        )

    def test_known_composite_endpoint_expands_to_two_relationships(self):
        result = {
            "diagram_kind": "other",
            "labels": [
                "Extracellular fluid", "Interstitial fluid",
                "Intravascular fluid",
            ],
            "components": [
                {"name": "Extracellular fluid", "description": "fluid"},
                {"name": "Interstitial fluid", "description": "fluid"},
                {"name": "Intravascular fluid", "description": "fluid"},
            ],
            "spatial_relationships": [{
                "subject": "Extracellular fluid",
                "relationship": "comprises",
                "object": "Interstitial fluid and Intravascular fluid",
            }],
            "connections": [{
                "from": "Extracellular fluid",
                "to": "Interstitial fluid and Intravascular fluid",
                "relationship": "comprises",
            }],
            "circuit_topology": None,
            "explanation": "The extracellular compartment has two fluid spaces.",
            "uncertain_items": [],
        }
        validated = structured.validate_labelled_diagram(result)
        self.assertEqual(
            [item["object"] for item in validated["spatial_relationships"]],
            ["Interstitial fluid", "Intravascular fluid"],
        )
        self.assertEqual(
            [item["to"] for item in validated["connections"]],
            ["Interstitial fluid", "Intravascular fluid"],
        )

    def test_duplicate_diagram_label_is_rejected(self):
        result = {
            "diagram_kind": "other",
            "labels": ["Adipocyte", "adipocyte"],
            "components": [], "spatial_relationships": [], "connections": [],
            "circuit_topology": None, "explanation": "Example", "uncertain_items": [],
        }
        with self.assertRaisesRegex(structured.StructuredOutputError, "duplicates"):
            structured.validate_labelled_diagram(result)

    def test_equivalent_circuit_topology_validates(self):
        validated = structured.validate_labelled_diagram(
            circuit_result(),
            "The intracellular fluid and cell membrane are represented by a series "
            "resistor RI and capacitor C. Rinf = RI * RE / (RI + RE).",
        )
        self.assertEqual(
            validated["circuit_topology"]["branches"],
            [
                {"id": "branch_1", "start_node": "n_start", "end_node": "n_end", "components": ["RE"]},
                {"id": "branch_2", "start_node": "n_start", "end_node": "n_end", "components": ["C", "RI"]},
            ],
        )
        self.assertEqual(structured._normal_name("RE"), structured._normal_name("R_E"))
        self.assertEqual(structured._normal_name("RE"), structured._normal_name("Rₑ"))

    def test_physiological_labels_are_not_electrical_nodes(self):
        result = json.loads(json.dumps(circuit_result()))
        topology = result["circuit_topology"]
        topology["nodes"][0] = {"id": "ECF", "label": "ECF terminal"}
        for edge in topology["edges"]:
            if edge["from_node"] == "n_start":
                edge["from_node"] = "ECF"
            if edge["to_node"] == "n_start":
                edge["to_node"] = "ECF"
        for branch in topology["branches"]:
            if branch["start_node"] == "n_start":
                branch["start_node"] = "ECF"
        with self.assertRaisesRegex(
            structured.StructuredOutputError, "Physiological labels"
        ):
            structured.validate_labelled_diagram(result)

    def test_series_caption_and_parallel_equation_build_exact_topology(self):
        evidence = (
            "R_I and C form a series branch. "
            "R_infinity = (R_I * R_E) / (R_I + R_E)."
        )
        grounded = structured.ground_circuit_topology_from_evidence(
            circuit_result(), evidence
        )
        validated = structured.validate_labelled_diagram(grounded, evidence)
        topology = validated["circuit_topology"]
        edges = {
            structured._normal_name(edge["component"]): edge
            for edge in topology["edges"]
        }
        self.assertEqual(set(edges), {"re", "ri", "c"})
        series = next(
            branch for branch in topology["branches"]
            if {structured._normal_name(item) for item in branch["components"]}
            == {"ri", "c"}
        )
        first, second = [
            edges[structured._normal_name(item)] for item in series["components"]
        ]
        self.assertEqual(first["to_node"], second["from_node"])
        parallel = [
            next(
                branch for branch in topology["branches"]
                if branch["id"] == branch_id
            )
            for branch_id in topology["parallel_branch_sets"][0]
        ]
        self.assertEqual(
            len({(branch["start_node"], branch["end_node"]) for branch in parallel}),
            1,
        )

    def test_unrelated_retrieved_circuit_does_not_override_target_components(self):
        evidence = (
            "RI and C form a series branch. "
            "R_infinity = (RI * RE) / (RI + RE). "
            "Another figure is in series with two parallel branches, each "
            "comprising a resistance (R1, R2)."
        )
        validated = structured.validate_labelled_diagram(
            circuit_result(), evidence
        )
        self.assertEqual(
            validated["circuit_topology"]["parallel_branch_sets"],
            [["branch_1", "branch_2"]],
        )

    def test_figure_three_rejects_one_series_chain(self):
        result = circuit_result()
        result["components"] = result["components"][:3]
        result["connections"] = [
            {"from": "RE", "to": "RI", "relationship": "series"},
            {"from": "RI", "to": "C", "relationship": "series"},
        ]
        result["circuit_topology"] = {
            "nodes": [{"id": name, "label": name} for name in ("n0", "n1", "n2", "n3")],
            "edges": [
                {"from_node": "n0", "to_node": "n1", "component": "RE"},
                {"from_node": "n1", "to_node": "n2", "component": "RI"},
                {"from_node": "n2", "to_node": "n3", "component": "C"},
            ],
            "branches": [{
                "id": "only", "start_node": "n0", "end_node": "n3",
                "components": ["RE", "RI", "C"],
            }],
            "parallel_branch_sets": [],
        }
        with self.assertRaisesRegex(structured.StructuredOutputError, "series branch"):
            structured.validate_labelled_diagram(
                result,
                "The intracellular fluid and membrane are represented by a series resistor RI and capacitor C.",
            )

    def test_figure_three_targeted_topology_retry_is_authoritative(self):
        initial = circuit_result()
        initial["connections"] = [
            {"from": "RE", "to": "RI", "relationship": "series"},
            {"from": "RI", "to": "C", "relationship": "series"},
        ]
        initial["circuit_topology"] = {
            "nodes": [{"id": name, "label": name} for name in ("n0", "n1", "n2", "n3")],
            "edges": [
                {"from_node": "n0", "to_node": "n1", "component": "RE"},
                {"from_node": "n1", "to_node": "n2", "component": "RI"},
                {"from_node": "n2", "to_node": "n3", "component": "C"},
            ],
            "branches": [{
                "id": "one_series_path", "start_node": "n0", "end_node": "n3",
                "components": ["RE", "RI", "C"],
            }],
            "parallel_branch_sets": [],
        }
        corrected = {
            key: circuit_result()[key]
            for key in ("components", "connections", "circuit_topology", "uncertain_items")
        }
        debug = {}
        evidence = (
            "The compartment is represented by a series resistor RI and capacitor C."
        )
        with patch.object(
            structured,
            "_call_model",
            side_effect=[json.dumps(initial), json.dumps(corrected)],
        ):
            structured.analyse_typed_image(
                Path("figure3.png"), "labelled_diagram", "Explain Figure 3",
                evidence_text=evidence, debug_info=debug,
            )
        topology = debug["validated_json"]["circuit_topology"]
        self.assertEqual(topology["branches"][0]["components"], ["RE"])
        self.assertEqual(topology["branches"][1]["components"], ["C", "RI"])
        self.assertEqual(topology["parallel_branch_sets"], [["branch_1", "branch_2"]])
        self.assertFalse({"α", "β", "δ", "γ"}.intersection(debug["validated_json"]["labels"]))
        self.assertEqual(debug["retry_kind"], "targeted_topology_retry")
        self.assertEqual(debug["final_answer_path"], "validated_typed_vision")

    def test_failed_topology_retry_can_be_grounded_from_explicit_equation(self):
        invalid = circuit_result()
        invalid["circuit_topology"] = {
            "nodes": [
                {"id": "n0", "label": "start"},
                {"id": "n1", "label": "end"},
            ],
            "edges": [
                {"from_node": "n0", "to_node": "n1", "component": "RE"},
            ],
            "branches": [{
                "id": "only", "start_node": "n0", "end_node": "n1",
                "components": ["RE"],
            }],
            "parallel_branch_sets": [],
        }
        retry = {
            key: invalid[key]
            for key in ("components", "connections", "circuit_topology", "uncertain_items")
        }
        retry["components"] = []
        debug = {}
        evidence = (
            "The compartment is represented by a series resistor RI and capacitor C. "
            "Rinf = RI * RE / (RI + RE)."
        )
        with patch.object(
            structured,
            "_call_model",
            side_effect=[json.dumps(invalid)],
        ):
            structured.analyse_typed_image(
                Path("figure3.png"), "labelled_diagram", "Explain this circuit",
                evidence_text=evidence, debug_info=debug,
            )
        topology = debug["validated_json"]["circuit_topology"]
        self.assertEqual(
            {frozenset(branch["components"]) for branch in topology["branches"]},
            {frozenset({"RE"}), frozenset({"RI", "C"})},
        )
        self.assertEqual(
            debug["retry_kind"],
            "grounded_topology_repair",
        )
        self.assertEqual(
            debug["final_answer_code_path"],
            "validated_repaired_structured_vision",
        )

    def test_six_panel_bode_graph_uses_graph_validation(self):
        panels = []
        frequency = axis("Frequency", "Hz", "log", ["10^2", "10^3", "10^4"])
        for label in "abcdef":
            group = f"Group {(ord(label) - ord('a')) // 2 + 1}"
            panels.append(graph_panel(
                label,
                "bode_magnitude" if label in "ace" else "bode_phase",
                group,
                frequency,
                axis("|Zfat|" if label in "ace" else "Phase", "Ω" if label in "ace" else "°"),
                50, 140,
            ))
        result = {
            "figure_number": "9", "panels": panels, "comparisons": [],
            "frequency_direction_evidence": ["Frequency axis tick labels"],
            "uncertain_values": [],
        }
        self.assertEqual(len(structured.validate_graph(result)["panels"]), 6)
        self.assertEqual(
            structured.detect_visual_type("Compare the groups in Figure 9", "Fig. 9. Bode diagrams"),
            "graph",
        )

    def test_figure_nine_axes_and_comparisons_validate(self):
        frequency = axis("Frequency", "Hz", "log", ["10^2", "10^3", "10^4"])
        panels = [
            graph_panel("a", "bode_magnitude", "Group 1", frequency, axis("|Zfat|", "Ω"), 87, 116),
            graph_panel("b", "bode_phase", "Group 1", frequency, axis("Phase", "°"), -90, 0, 0.2),
            graph_panel("c", "bode_magnitude", "Group 2", frequency, axis("|Zfat|", "Ω"), 55, 105),
            graph_panel("d", "bode_phase", "Group 2", frequency, axis("Phase", "°"), -92, 0, 0.95),
            graph_panel("e", "bode_magnitude", "Group 3", frequency, axis("|Zfat|", "Ω"), 81, 138),
            graph_panel("f", "bode_phase", "Group 3", frequency, axis("Phase", "°"), -91, 0, 0.3),
        ]
        result = {
            "figure_number": "9",
            "panels": panels,
            "comparisons": [
                comparison("Group 3 has the highest overall magnitude range.", "Group 3", "highest", "magnitude"),
                comparison("Group 2 has the lowest overall magnitude range.", "Group 2", "lowest", "magnitude"),
                comparison("Group 2 has the most complex phase behaviour.", "Group 2", "most_complex", "phase complexity"),
            ],
            "frequency_direction_evidence": ["Frequency tick labels are printed on each x-axis."],
            "uncertain_values": [],
        }
        validated = structured.validate_graph(result)
        self.assertEqual(validated["panels"][0]["x_axis"]["scale"], "log")
        self.assertEqual(validated["panels"][0]["y_axis"]["label"], "|Zfat|")
        self.assertEqual(
            [item["subject"] for item in validated["comparisons"]],
            ["Group 3", "Group 2", "Group 2"],
        )

    def test_wrong_figure_nine_ordering_is_rejected(self):
        frequency = axis("Frequency", "Hz", "log", ["10^2", "10^3", "10^4"])
        panels = [
            graph_panel("a", "bode_magnitude", "Group 1", frequency, axis("|Zfat|", "Ω"), 87, 116),
            graph_panel("c", "bode_magnitude", "Group 2", frequency, axis("|Zfat|", "Ω"), 55, 105),
            graph_panel("e", "bode_magnitude", "Group 3", frequency, axis("|Zfat|", "Ω"), 81, 138),
        ]
        result = {
            "figure_number": "9", "panels": panels,
            "comparisons": [comparison("Group 1 is highest.", "Group 1", "highest", "magnitude")],
            "frequency_direction_evidence": [], "uncertain_values": [],
        }
        with self.assertRaisesRegex(structured.StructuredOutputError, "estimated ranges"):
            structured.validate_graph(result)

    def test_explicit_text_conflict_must_be_uncertain(self):
        frequency = axis("Frequency", "Hz", "log", ["10^2", "10^3", "10^4"])
        panels = [
            graph_panel("a", "bode_magnitude", "Group 1", frequency, axis("Magnitude", "Ω"), 10, 20),
            graph_panel("c", "bode_magnitude", "Group 2", frequency, axis("Magnitude", "Ω"), 20, 30),
        ]
        result = {
            "figure_number": "x", "panels": panels,
            "comparisons": [comparison("Group 2 has the highest magnitude.", "Group 2", "highest", "magnitude")],
            "frequency_direction_evidence": [], "uncertain_values": [],
        }
        with self.assertRaisesRegex(structured.StructuredOutputError, "explicit caption"):
            structured.validate_graph(
                result,
                "The caption explicitly states that Group 1 has the highest magnitude.",
            )

    def test_figure_thirteen_scientific_multiplier_is_linear(self):
        result = {
            "figure_number": "13",
            "panels": [graph_panel(
                "b", "nyquist", "Sample 7",
                axis("Re(Z)", "Ω", "linear", ["0", "0.5", "1", "1.5", "2", "2.5"], "×10^4"),
                axis("-Im(Z)", "Ω", "linear", ["0", "0.2", "0.4", "0.6", "0.8", "1.0"], "×10^4"),
                0, 10000,
                trends=["The largest mismatch is on the right side at high Re(Z)."],
            )],
            "comparisons": [], "frequency_direction_evidence": [], "uncertain_values": [],
        }
        self.assertEqual(
            structured.validate_graph(result)["panels"][0]["x_axis"]["scale"],
            "linear",
        )

    def test_graph_complexity_boundaries_and_fraction_pass(self):
        for complexity in (0, 0.2, 1):
            with self.subTest(complexity=complexity):
                result = {
                    "figure_number": "13",
                    "panels": [graph_panel(
                        "a", "nyquist", "Sample 10",
                        axis("Re(Z)", "ohm"), axis("-Im(Z)", "ohm"), 0, 2,
                        complexity=complexity,
                    )],
                    "comparisons": [], "frequency_direction_evidence": [],
                    "uncertain_values": [],
                }
                self.assertEqual(
                    structured.validate_graph(result)["panels"][0]["complexity_score"],
                    complexity,
                )

    def test_graph_complexity_outside_boundaries_fails(self):
        for complexity in (-0.01, 1.01):
            with self.subTest(complexity=complexity):
                result = {
                    "figure_number": "13",
                    "panels": [graph_panel(
                        "a", "nyquist", "Sample 10",
                        axis("Re(Z)", "ohm"), axis("-Im(Z)", "ohm"), 0, 2,
                        complexity=complexity,
                    )],
                    "comparisons": [], "frequency_direction_evidence": [],
                    "uncertain_values": [],
                }
                with self.assertRaisesRegex(
                    structured.StructuredOutputError, "between 0 and 1"
                ):
                    structured.validate_graph(result)

    def test_graph_formatter_normalizes_figure_prefix_once(self):
        panel = graph_panel(
            "a", "nyquist", "Sample 10",
            axis("Re(Z)", "ohm"), axis("-Im(Z)", "ohm"), 0, 2,
        )
        for identifier in ("13", "Fig. 13", "Figure 13"):
            with self.subTest(identifier=identifier):
                rendered = structured.format_structured_result("graph", {
                    "figure_number": identifier,
                    "panels": [panel],
                    "comparisons": [],
                    "uncertain_values": [],
                })
                self.assertTrue(
                    rendered.startswith("**Figure 13 graph analysis**")
                )
                self.assertNotIn("Figure Fig.", rendered)
                self.assertNotIn("Figure Figure", rendered)

    def test_graph_formatter_includes_group_without_duplication(self):
        panels = [
            graph_panel(
                "a", "nyquist", "Sample 10",
                axis("Re(Z)", "ohm"), axis("-Im(Z)", "ohm"), 0, 2,
            ),
            graph_panel(
                "b \u2014 Sample 7", "nyquist", "Sample 7",
                axis("Re(Z)", "ohm"), axis("-Im(Z)", "ohm"), 0, 2,
            ),
        ]
        rendered = structured.format_structured_result("graph", {
            "figure_number": "Fig. 13",
            "panels": panels,
            "comparisons": [],
            "uncertain_values": [],
        })
        self.assertIn("**Panel a \u2014 Sample 10**", rendered)
        self.assertEqual(rendered.count("**Panel b \u2014 Sample 7**"), 1)
        self.assertNotIn("Sample 7 \u2014 Sample 7", rendered)

        compact_panel = {
            "panel": "a", "group": "Group 1",
            "x_axis": {"label": "Frequency", "unit": "Hz", "scale": "log"},
            "y_axis": {"label": "|Zfat|", "unit": "ohm", "scale": "linear"},
            "visible_trend": "Magnitude decreases.",
        }
        compact = structured.format_compact_graph_result({
            "figure_number": "Figure 9",
            "panels": [compact_panel],
            "comparisons": {
                "magnitude_order_high_to_low": [],
                "greatest_phase_complexity_group": "",
                "uncertain": [],
            },
            "uncertain_values": [],
        })
        self.assertTrue(compact.startswith("**Figure 9 graph analysis**"))
        self.assertIn("**Panel a \u2014 Group 1**", compact)

    def test_figure_thirteen_comparison_is_deduplicated_and_scale_grounded(self):
        sample_ten = graph_panel(
            "(a) Sample 10", "nyquist", None,
            axis("Re(Z)", "ohm", "linear", ["0", "1", "2"], "x10^5"),
            axis("-Img(Z)", "ohm", "linear", ["0", "1", "2"], "x10^5"),
            0, 200000,
            trends=["Raw data closely follow the fitted model."],
        )
        sample_seven = graph_panel(
            "b", "nyquist", "Sample 7",
            axis("Re(Z)", "ohm", "linear", ["0", "1", "2"], "x10^4"),
            axis("\u2212Img(Z)", "ohm", "linear", ["0", "1", "2"], "x10^4"),
            0, 20000,
            trends=[
                "Raw data closely follow the fitted model.",
                "A visible local deviation occurs on the right at high Re(Z).",
            ],
        )
        for panel in (sample_ten, sample_seven):
            panel["series"] = ["Raw Data", "Fitted Model"]
        duplicate = comparison(
            "Both panels show good model fits.",
            "Sample 10", "other", "fit quality",
        )
        result = {
            "figure_number": "13",
            "panels": [sample_ten, sample_seven],
            "comparisons": [duplicate, dict(duplicate)],
            "frequency_direction_evidence": [],
            "uncertain_values": [],
        }
        validated = structured.validate_graph(result)
        by_panel = {panel["panel"]: panel for panel in validated["panels"]}
        self.assertEqual(by_panel["a"]["group"], "Sample 10")
        self.assertEqual(by_panel["b"]["group"], "Sample 7")
        self.assertEqual(by_panel["a"]["y_axis"]["label"], "-Im(Z)")
        self.assertEqual(by_panel["b"]["y_axis"]["label"], "-Im(Z)")
        claims = [item["claim"] for item in validated["comparisons"]]
        normalized_claims = [structured._normal_name(claim) for claim in claims]
        self.assertEqual(len(normalized_claims), len(set(normalized_claims)))
        scale = next(
            item for item in validated["comparisons"]
            if item["metric"] == "overall impedance scale"
        )
        self.assertEqual(scale["subject"], "Sample 10")
        self.assertIn("\u00d710^5", scale["claim"])
        self.assertIn("\u00d710^4", scale["claim"])
        self.assertTrue(all(
            any("good" in trend.lower() for trend in panel["visible_trends"])
            for panel in validated["panels"]
        ))
        rendered = structured.format_structured_result("graph", validated)
        self.assertNotIn("img(z)", rendered.lower())
        self.assertNotRegex(rendered.lower(), r"higher?-frequency|lower?-frequency")

    def test_nyquist_visible_range_applies_scientific_multiplier(self):
        result = {
            "figure_number": "13",
            "panels": [graph_panel(
                "b", "nyquist", "Sample 7",
                axis("Re(Z)", "ohm", "linear", ["0", "1", "2"], "x10^4"),
                axis("-Im(Z)", "ohm", "linear", ["0", "0.5", "1"], "x10^4"),
                0, 100000,
            )],
            "comparisons": [], "frequency_direction_evidence": [],
            "uncertain_values": [],
        }
        with self.assertRaisesRegex(
            structured.StructuredOutputError, "scientific multiplier"
        ):
            structured.validate_graph(result)

    def test_nyquist_fit_comparison_names_visible_sample(self):
        result = {
            "figure_number": "13",
            "panels": [graph_panel(
                "b", "nyquist", "Sample 7",
                axis("Re(Z)", "ohm"), axis("-Im(Z)", "ohm"), 0, 2,
            )],
            "comparisons": [comparison(
                "Sample 7 has the closest fit.", "Visual fit quality",
                "highest", "fit quality",
            )],
            "frequency_direction_evidence": [], "uncertain_values": [],
        }
        with self.assertRaisesRegex(
            structured.StructuredOutputError, "visible panel sample"
        ):
            structured.validate_graph(result)

    def test_tight_panel_fit_verification_uses_normalized_deviation(self):
        panels = [
            graph_panel(
                "a", "nyquist", "Specimen Alpha",
                axis("Re(Z)", "ohm"), axis("-Im(Z)", "ohm"), 0, 2,
                trends=[
                    "Raw data have a good fit to the model.",
                    "The largest deviation is on the right side.",
                ],
            ),
            graph_panel(
                "b", "nyquist", "Specimen Beta",
                axis("Re(Z)", "ohm"), axis("-Im(Z)", "ohm"), 0, 2,
                trends=[
                    "Raw data have a good fit to the model.",
                    "The largest deviation is on the right side.",
                ],
            ),
        ]
        for panel in panels:
            panel["series"] = ["Raw Data", "Fitted Model"]
        result = {
            "figure_number": "x", "panels": panels,
            "comparisons": [comparison(
                "Specimen Beta has the closer fit.", "Specimen Beta",
                "lowest", "normalized largest model-data deviation",
            )],
            "frequency_direction_evidence": [], "uncertain_values": [],
        }
        readings = [
            {"panel": "a", "group": "Specimen Alpha",
             "left_normalized_largest_deviation": 0.04,
             "right_normalized_largest_deviation": 0.12,
             "confidence": 0.9},
            {"panel": "b", "group": "Specimen Beta",
             "left_normalized_largest_deviation": 0.08,
             "right_normalized_largest_deviation": 0.24,
             "confidence": 0.85},
        ]
        with patch.object(
            structured,
            "_call_model",
            side_effect=[json.dumps(reading) for reading in readings],
        ):
            validated, raw, status = structured.verify_nyquist_fit_comparison(
                result,
                [("a", Path("a.png")), ("b", Path("b.png"))],
            )
        self.assertEqual(status, "verified")
        self.assertEqual(len(raw), 2)
        self.assertEqual(validated["comparisons"][-1]["subject"], "Specimen Alpha")
        self.assertFalse(validated["comparisons"][-1]["uncertain"])
        self.assertIn("high Re(Z)", validated["panels"][1]["visible_trends"][-1])

    def test_tight_panel_fit_verification_is_uncertain_when_nearly_tied(self):
        panels = [
            graph_panel(
                "a", "nyquist", "Specimen Alpha",
                axis("Re(Z)", "ohm"), axis("-Im(Z)", "ohm"), 0, 2,
                trends=[
                    "The fit is good; the largest deviation is on the right."
                ],
            ),
            graph_panel(
                "b", "nyquist", "Specimen Beta",
                axis("Re(Z)", "ohm"), axis("-Im(Z)", "ohm"), 0, 2,
                trends=[
                    "The fit is good; the largest deviation is on the right."
                ],
            ),
        ]
        result = {
            "figure_number": "x", "panels": panels, "comparisons": [],
            "frequency_direction_evidence": [], "uncertain_values": [],
        }
        readings = [
            {"panel": "a", "group": "Specimen Alpha",
             "left_normalized_largest_deviation": 0.10,
             "right_normalized_largest_deviation": 0.20,
             "confidence": 0.9},
            {"panel": "b", "group": "Specimen Beta",
             "left_normalized_largest_deviation": 0.11,
             "right_normalized_largest_deviation": 0.21,
             "confidence": 0.9},
        ]
        with patch.object(
            structured,
            "_call_model",
            side_effect=[json.dumps(reading) for reading in readings],
        ):
            validated, _, _ = structured.verify_nyquist_fit_comparison(
                result,
                [("a", Path("a.png")), ("b", Path("b.png"))],
            )
        self.assertTrue(validated["comparisons"][-1]["uncertain"])
        self.assertIn("too close", validated["comparisons"][-1]["claim"])
        self.assertEqual(validated["uncertain_values"], [])
        rendered = structured.format_structured_result("graph", validated)
        self.assertEqual(rendered.lower().count("too close to distinguish"), 1)

    def test_tight_crop_deviations_remain_panel_specific_and_significant(self):
        panels = [
            graph_panel(
                "a", "nyquist", "Specimen Alpha",
                axis("Re(Z)", "ohm"), axis("-Im(Z)", "ohm"), 0, 2,
                trends=[
                    "The fit is good.",
                    "The largest deviation is on the right side.",
                ],
            ),
            graph_panel(
                "b", "nyquist", "Specimen Beta",
                axis("Re(Z)", "ohm"), axis("-Im(Z)", "ohm"), 0, 2,
                trends=[
                    "The fit is good.",
                    "The largest deviation is on the right side.",
                ],
            ),
        ]
        result = {
            "figure_number": "x", "panels": panels,
            "comparisons": [
                comparison(
                    "Specimen Alpha deviates on the right.",
                    "Specimen Alpha", "highest", "model-data deviation",
                ),
                comparison(
                    "Specimen Beta deviates on the left.",
                    "Specimen Beta", "highest", "model-data deviation",
                ),
            ],
            "frequency_direction_evidence": [], "uncertain_values": [
                "Relative Nyquist fit closeness could not be distinguished reliably."
            ],
        }
        readings = [
            {
                "panel": "a", "group": "Specimen Alpha",
                "left_normalized_largest_deviation": 0.004,
                "right_normalized_largest_deviation": 0.009,
                "right_endpoint_extension": 0.0, "confidence": 0.9,
            },
            {
                "panel": "b", "group": "Specimen Beta",
                "left_normalized_largest_deviation": 0.004,
                "right_normalized_largest_deviation": 0.008,
                "right_endpoint_extension": 0.05, "confidence": 0.9,
            },
        ]
        with patch.object(
            structured,
            "_call_model",
            side_effect=[json.dumps(reading) for reading in readings],
        ):
            validated, _, status = structured.verify_nyquist_fit_comparison(
                result,
                [("a", Path("a.png")), ("b", Path("b.png"))],
            )
        self.assertEqual(status, "verified")
        alpha_text = " ".join(validated["panels"][0]["visible_trends"])
        beta_text = " ".join(validated["panels"][1]["visible_trends"])
        self.assertNotRegex(alpha_text.lower(), r"deviation.*(?:right|high)")
        self.assertIn("For Specimen Beta", beta_text)
        self.assertIn("right side at high Re(Z)", beta_text)
        self.assertNotIn("Specimen Alpha", beta_text)
        self.assertEqual(validated["uncertain_values"], [])
        rendered = structured.format_structured_result("graph", validated)
        self.assertEqual(rendered.lower().count("too close to distinguish"), 1)

    def test_nyquist_combined_panel_and_sample_identity_is_normalized(self):
        panel = graph_panel(
            "(a) Sample 10", "nyquist", None,
            axis("Re(Z)", "ohm"), axis("-Im(Z)", "ohm"), 0, 2,
        )
        result = {
            "figure_number": "13", "panels": [panel], "comparisons": [],
            "frequency_direction_evidence": [], "uncertain_values": [],
        }
        validated = structured.validate_graph(result)
        self.assertEqual(validated["panels"][0]["panel"], "a")
        self.assertEqual(validated["panels"][0]["group"], "Sample 10")

    def test_nyquist_entire_range_claim_cannot_conflict_with_deviation(self):
        result = {
            "figure_number": "13",
            "panels": [graph_panel(
                "b", "nyquist", "Sample 7",
                axis("Re(Z)", "ohm"), axis("-Im(Z)", "ohm"), 0, 2,
                trends=["The fit is perfect across the entire range."],
            )],
            "comparisons": [comparison(
                "Sample 7 deviates on the right side.", "Sample 7",
                "other", "deviation magnitude",
            )],
            "frequency_direction_evidence": [], "uncertain_values": [],
        }
        with self.assertRaisesRegex(
            structured.StructuredOutputError, "contradicts"
        ):
            structured.validate_graph(result)

    def test_nyquist_frequency_direction_without_evidence_is_rejected(self):
        result = {
            "figure_number": "13",
            "panels": [graph_panel(
                "b", "nyquist", "Sample 7",
                axis("Re(Z)", "Ω", "linear", ["0", "1", "2"], "×10^4"),
                axis("-Im(Z)", "Ω", "linear", ["0", "1", "2"], "×10^4"),
                0, 20000, trends=["The mismatch occurs at higher frequencies."],
            )],
            "comparisons": [], "frequency_direction_evidence": [], "uncertain_values": [],
        }
        with self.assertRaisesRegex(structured.StructuredOutputError, "frequency direction"):
            structured.validate_graph(result)

    def test_table_one_columns_rows_and_repeated_groups_validate(self):
        rows = [
            [1, 10, 600, 534, 0.89], [1, 9, 540, 486, 0.90],
            [2, 8, 480, 446, 0.93], [2, 7, 420, 392, 0.93],
            [2, 6, 360, 350, 0.97], [2, 5, 300, 295, 0.98],
            [3, 4, 240, 257, 1.07], [3, 3, 180, 205, 1.14],
            [3, 2, 120, 151, 1.26], [3, 1, 60, 80, 1.33],
        ]
        result = {
            "table_number": "1",
            "title": "Volume, weight, and density of adipose tissue samples",
            "columns": ["Group", "Sample", "Volume", "Weight", "Density"],
            "rows": rows,
            "units": {"Group": None, "Sample": None, "Volume": "cm³", "Weight": "g", "Density": "g/cm³"},
            "comparisons": [], "unreadable_cells": [],
        }
        validated = structured.validate_table(result)
        self.assertEqual(validated["columns"], ["Group", "Sample", "Volume", "Weight", "Density"])
        self.assertEqual(validated["rows"], rows)

    def test_table_rejects_wrong_row_length_not_repeated_headings(self):
        result = {
            "table_number": "1", "title": "Example",
            "columns": ["Group", "Value"], "rows": [[1]], "units": {},
            "comparisons": [], "unreadable_cells": [],
        }
        with self.assertRaisesRegex(structured.StructuredOutputError, "expected 2"):
            structured.validate_table(result)

    def test_markdown_fenced_json_is_parsed(self):
        self.assertEqual(
            structured.parse_json_response("```json\n{\"panels\": []}\n```"),
            {"panels": []},
        )

    def test_json_object_is_extracted_from_short_model_preface(self):
        self.assertEqual(
            structured.parse_json_response('Result follows: {"panels": []}'),
            {"panels": []},
        )

    def test_balanced_malformed_json_is_not_reported_as_truncated(self):
        with self.assertRaises(structured.StructuredOutputError) as caught:
            structured.parse_json_response('{"panels": [invalid]}')
        self.assertNotIsInstance(caught.exception, structured.TruncatedJSONError)

    def test_truncated_json_is_detected_separately(self):
        with self.assertRaises(structured.TruncatedJSONError):
            structured.parse_json_response('{"panels":[{"panel":"a","visible_trend":"unterminated')

    def test_raster_panel_preflight_detects_six_panels_compactly(self):
        with patch.object(
            structured,
            "_call_model",
            return_value='{"panel_ids":["a","b","c","d","e","f"]}',
        ):
            panel_ids, _ = structured.detect_compact_graph_panel_ids(Path("figure9.png"))
        self.assertEqual(panel_ids, list("abcdef"))

    def test_figure_nine_compact_pairs_merge_and_compare(self):
        pairs = [
            compact_pair("a", "b", "Group 1", 102, 0.3),
            compact_pair("c", "d", "Group 2", 80, 0.95),
            compact_pair("e", "f", "Group 3", 110, 0.4),
        ]
        comparison_result = {
            "magnitude_order_high_to_low": ["Group 3", "Group 1", "Group 2"],
            "greatest_phase_complexity_group": "Group 2",
            "x_axis": {"label": "Frequency", "unit": "Hz", "scale": "log"},
            "magnitude_y_axis": {"label": "|Zfat|", "unit": "Ω", "scale": "linear"},
            "confidence": 0.93,
            "uncertain": [],
        }
        debug = {}
        with patch.object(
            structured,
            "_call_model",
            side_effect=[json.dumps(value) for value in pairs],
        ), patch.object(
            structured,
            "_call_model_images",
            return_value=json.dumps(comparison_result),
        ), patch.object(
            structured, "verify_magnitude_levels", return_value=([], [])
        ) as magnitude_verify:
            answer = structured.analyse_compact_multi_panel_graph(
                [Path("ab.png"), Path("cd.png"), Path("ef.png")],
                [["a", "b"], ["c", "d"], ["e", "f"]],
                "9",
                "Compare Figure 9",
                debug_info=debug,
            )
        magnitude_verify.assert_called_once()
        self.assertEqual(len(debug["validated_json"]["panels"]), 6)
        self.assertEqual(
            debug["validated_json"]["comparisons"]["magnitude_order_high_to_low"],
            ["Group 3", "Group 1", "Group 2"],
        )
        self.assertEqual(
            debug["validated_json"]["comparisons"]["greatest_phase_complexity_group"],
            "Group 2",
        )
        self.assertTrue(all(
            panel["x_axis"] == {"label": "Frequency", "unit": "Hz", "scale": "log"}
            for panel in debug["validated_json"]["panels"]
        ))
        magnitude_labels = {
            structured._normal_name(panel["y_axis"]["label"])
            for panel in debug["validated_json"]["panels"]
            if panel["graph_kind"] == "magnitude"
        }
        self.assertEqual(magnitude_labels, {structured._normal_name("|Zfat|")})
        self.assertEqual(debug["final_answer_path"], "validated_typed_vision")
        self.assertIn("Group 3 > Group 1 > Group 2", answer)

    def test_compact_bare_numeric_group_ids_are_canonicalized(self):
        pairs = [
            compact_pair("a", "b", "1", 102, 0.3),
            compact_pair("c", "d", "2", 80, 0.95),
            compact_pair("e", "f", "3", 110, 0.4),
        ]
        comparison_result = {
            "magnitude_order_high_to_low": ["Group 3", "Group 1", "Group 2"],
            "greatest_phase_complexity_group": "Group 2",
            "x_axis": {"label": "Frequency", "unit": "Hz", "scale": "log"},
            "magnitude_y_axis": {"label": "|Zfat|", "unit": "Ω", "scale": "linear"},
            "confidence": 0.93,
            "uncertain": [],
        }
        debug = {}
        with patch.object(
            structured, "_call_model",
            side_effect=[json.dumps(value) for value in pairs],
        ), patch.object(
            structured, "_call_model_images",
            return_value=json.dumps(comparison_result),
        ), patch.object(
            structured, "verify_magnitude_levels", return_value=([], [])
        ) as magnitude_verify:
            structured.analyse_compact_multi_panel_graph(
                [Path("ab.png"), Path("cd.png"), Path("ef.png")],
                [["a", "b"], ["c", "d"], ["e", "f"]],
                "9", "Compare Figure 9", debug_info=debug,
            )
        magnitude_verify.assert_called_once()
        value = debug["validated_json"]
        self.assertEqual(
            [value["panels"][index]["group"] for index in range(0, 6, 2)],
            ["Group 1", "Group 2", "Group 3"],
        )
        self.assertEqual(
            value["comparisons"]["greatest_phase_complexity_group"], "Group 2"
        )
        self.assertEqual(debug["final_answer_path"], "validated_typed_vision")

    def test_compact_db_and_signed_phase_axes_are_linear_coordinates(self):
        value = compact_pair("a", "b", "Group 1", 100, 0.3)
        value["panels"][0]["y_axis"] = {
            "label": "|Z|dB (Ω)", "unit": "Ω", "scale": "log",
        }
        value["panels"][1]["y_axis"] = {
            "label": "Phase", "unit": "°", "scale": "log",
        }
        validated = structured.validate_compact_panel_response(value, ["a", "b"])
        self.assertEqual(
            [panel["y_axis"]["scale"] for panel in validated["panels"]],
            ["linear", "linear"],
        )

    def test_mixed_magnitude_scales_normalize_after_shared_db_label_merge(self):
        pairs = [
            compact_pair("a", "b", "Group 1", 102, 0.3),
            compact_pair("c", "d", "Group 2", 80, 0.95),
            compact_pair("e", "f", "Group 3", 110, 0.4),
        ]
        pairs[2]["panels"][0]["y_axis"] = {
            "label": "|Z|", "unit": "Ω", "scale": "log",
        }
        comparison = {
            "magnitude_order_high_to_low": ["Group 3", "Group 1", "Group 2"],
            "greatest_phase_complexity_group": "Group 2",
            "x_axis": {"label": "Frequency", "unit": "Hz", "scale": "log"},
            "magnitude_y_axis": {
                "label": "|Zfat|_{dB}", "unit": "Ω", "scale": "log",
            },
            "confidence": 0.93,
            "uncertain": [],
        }
        debug = {}
        with patch.object(
            structured, "_call_model",
            side_effect=[json.dumps(value) for value in pairs],
        ), patch.object(
            structured, "_call_model_images", return_value=json.dumps(comparison),
        ), patch.object(
            structured, "verify_magnitude_levels", return_value=([], []),
        ):
            structured.analyse_compact_multi_panel_graph(
                [Path("ab.png"), Path("cd.png"), Path("ef.png")],
                [["a", "b"], ["c", "d"], ["e", "f"]],
                "9", "Compare Figure 9", debug_info=debug,
            )
        value = debug["validated_json"]
        magnitude_panels = [
            panel for panel in value["panels"]
            if panel["graph_kind"] == "magnitude"
        ]
        self.assertTrue(all(
            panel["x_axis"]["scale"] == "log" for panel in value["panels"]
        ))
        self.assertTrue(all(
            panel["y_axis"]["scale"] == "linear" for panel in magnitude_panels
        ))
        self.assertEqual(
            value["comparisons"]["magnitude_y_axis"]["scale"], "linear"
        )
        self.assertEqual(
            value["comparisons"]["magnitude_order_high_to_low"],
            ["Group 3", "Group 1", "Group 2"],
        )
        self.assertEqual(
            value["comparisons"]["greatest_phase_complexity_group"], "Group 2"
        )

    def test_fat_impedance_axis_variants_normalize_but_generic_z_does_not(self):
        variants = [
            "|Zfat|",
            "|Z_fat|",
            "|Z₍fat₎|",
            r"|Z_{\mathrm{fat}}|",
            "|Z<sub>fat</sub>|",
        ]
        for variant in variants:
            with self.subTest(variant=variant):
                self.assertIn(
                    "Zfat",
                    structured.normalize_impedance_axis_label(variant),
                )
        generic = "|Z|<sub>dB</sub> (Ω)"
        self.assertEqual(
            structured.normalize_impedance_axis_label(generic), generic
        )
        self.assertEqual(
            structured.normalize_impedance_axis_label(
                generic,
                "Fig. 9. Bode diagrams of adipose tissue impedance.",
            ),
            "|Zfat|<sub>dB</sub> (Ω)",
        )

    def test_figure_nine_missing_subscript_retries_only_axis_label_crop(self):
        comparison = {
            "magnitude_order_high_to_low": ["Group 1"],
            "greatest_phase_complexity_group": "Group 1",
            "x_axis": {"label": "Frequency", "unit": "Hz", "scale": "log"},
            "magnitude_y_axis": {
                "label": "|Z|<sub>dB</sub> (Ω)", "unit": "Ω", "scale": "log"
            },
            "confidence": 0.95,
            "uncertain": [],
        }
        retry = {"label": "|Z_fat|<sub>dB</sub> (Ω)", "confidence": 0.91,
                 "subscript_visible": True}
        debug = {}
        with patch.object(
            structured, "_call_model",
            return_value=json.dumps(compact_pair("a", "b", "Group 1", 100, 0.3)),
        ), patch.object(
            structured, "_call_model_images",
            side_effect=[json.dumps(comparison), json.dumps(retry)],
        ) as model_images:
            structured.analyse_compact_multi_panel_graph(
                [Path("panels_ab.png")], [["a", "b"]], "9", "Figure 9",
                evidence_text="Bode diagrams of adipose tissue impedance.",
                axis_label_image_paths=[Path("axis_a.png")], debug_info=debug,
            )
        self.assertEqual(model_images.call_count, 2)
        self.assertEqual(model_images.call_args_list[1].args[0], [Path("axis_a.png")])
        self.assertEqual(
            debug["validated_json"]["comparisons"]["magnitude_y_axis"]["label"],
            "|Zfat|<sub>dB</sub> (Ω)",
        )

    def test_comparison_axis_reread_does_not_change_panel_scales(self):
        comparison = {
            "magnitude_order_high_to_low": ["Group 1"],
            "greatest_phase_complexity_group": "Group 1",
            "x_axis": {"label": "Frequency", "unit": "Hz", "scale": "log"},
            "magnitude_y_axis": {
                "label": "|Zfat| dB", "unit": "ohm", "scale": "log"
            },
            "confidence": 0.95,
            "uncertain": [],
        }
        debug = {}
        with patch.object(
            structured,
            "_call_model",
            return_value=json.dumps(
                compact_pair("a", "b", "Group 1", 100, 0.3)
            ),
        ), patch.object(
            structured, "_call_model_images", return_value=json.dumps(comparison)
        ):
            structured.analyse_compact_multi_panel_graph(
                [Path("panels_ab.png")], [["a", "b"]], "9", "Figure 9",
                debug_info=debug,
            )
        panels = debug["validated_json"]["panels"]
        self.assertTrue(all(panel["x_axis"]["scale"] == "log" for panel in panels))
        self.assertEqual(
            next(panel for panel in panels if panel["graph_kind"] == "magnitude")["y_axis"]["scale"],
            "linear",
        )
        self.assertEqual(
            next(panel for panel in panels if panel["graph_kind"] == "phase")["y_axis"]["scale"],
            "linear",
        )
        self.assertEqual(
            debug["validated_json"]["comparisons"]["magnitude_y_axis"]["scale"],
            "linear",
        )

    def test_truncated_compact_pair_is_retried_without_showing_raw_json(self):
        valid = json.dumps(compact_pair("a", "b", "Group 1", 100, 0.3))
        with patch.object(
            structured,
            "_call_model",
            side_effect=['{"panels":[{"panel":"a","visible_trend":"cut', valid],
        ), patch.object(
            structured,
            "_call_model_images",
            return_value=json.dumps({
                "magnitude_order_high_to_low": ["Group 1"],
                "greatest_phase_complexity_group": "Group 1",
                "x_axis": {"label": "Frequency", "unit": "Hz", "scale": "log"},
                "magnitude_y_axis": {"label": "|Zfat|", "unit": "Ω", "scale": "linear"},
                "confidence": 0.8,
                "uncertain": [],
            }),
        ):
            answer = structured.analyse_compact_multi_panel_graph(
                [Path("ab.png")], [["a", "b"]], "9", "Explain Figure 9"
            )
        self.assertNotIn("cut", answer)
        self.assertNotIn("Unvalidated vision fallback", answer)

    def test_compact_panel_ids_accept_visible_parentheses(self):
        value = compact_pair("(a)", "(b)", "Group 1", 100, 0.3)
        validated = structured.validate_compact_panel_response(value, ["a", "b"])
        self.assertEqual(len(validated["panels"]), 2)

    def test_one_json_repair_retry_can_recover(self):
        valid = json.dumps({
            "figure_number": "9",
            "panels": [graph_panel(
                "a", "bode_magnitude", "Group 1",
                axis("Frequency", "Hz"), axis("Magnitude", "Ω"), 0, 1,
            )],
            "comparisons": [], "frequency_direction_evidence": [], "uncertain_values": [],
        })
        with patch.object(structured, "_call_model", side_effect=["{bad", valid]) as call:
            answer = structured.analyse_typed_image(
                Path("graph.png"), "graph", "Explain Figure 9"
            )
        self.assertIn("Figure 9", answer)
        self.assertEqual(call.call_count, 2)

    def test_repaired_graph_does_not_retain_stale_final_validation_error(self):
        repaired = {
            "figure_number": "13",
            "panels": [graph_panel(
                "a", "nyquist", "Sample 10",
                axis("Re(Z)", "ohm"), axis("-Img(Z)", "ohm"), 0, 2,
                complexity=0.2,
            )],
            "comparisons": [], "frequency_direction_evidence": [],
            "uncertain_values": [],
        }
        initial = json.loads(json.dumps(repaired))
        initial["panels"][0]["complexity_score"] = 2.0
        debug = {}
        with patch.object(
            structured,
            "_call_model",
            side_effect=[json.dumps(initial), json.dumps(repaired)],
        ):
            answer = structured.analyse_typed_image(
                Path("figure13.png"), "graph",
                "Compare Samples 10 and 7 in Figure 13.",
                debug_info=debug,
            )
        self.assertIn("Graph complexity must be between 0 and 1", " ".join(
            debug["initial_validation_errors"]
        ))
        self.assertEqual(debug["repaired_validation_result"], "passed")
        self.assertEqual(debug["validation_error"], "")
        self.assertEqual(
            debug["final_answer_code_path"],
            "validated_repaired_structured_vision",
        )
        self.assertIn("Figure 13", answer)

    def test_valid_raw_graph_has_no_initial_or_final_validation_error(self):
        valid = {
            "figure_number": "13",
            "panels": [
                graph_panel(
                    "a", "nyquist", "Sample 10",
                    axis("Re(Z)", "ohm"), axis("-Im(Z)", "ohm"), 0, 2,
                    complexity=0.2,
                ),
                graph_panel(
                    "b", "nyquist", "Sample 7",
                    axis("Re(Z)", "ohm"), axis("-Im(Z)", "ohm"), 0, 2,
                    complexity=0.2,
                ),
            ],
            "comparisons": [], "frequency_direction_evidence": [],
            "uncertain_values": [],
        }
        debug = {}
        with patch.object(structured, "_call_model", return_value=json.dumps(valid)):
            answer = structured.analyse_typed_image(
                Path("figure13.png"), "graph",
                "Compare Samples 10 and 7 in Figure 13.",
                debug_info=debug,
            )
        self.assertEqual(debug["initial_validation_errors"], [])
        self.assertEqual(debug["repaired_validation_result"], "not_attempted")
        self.assertEqual(debug["validation_error"], "")
        self.assertNotIn("Graph complexity", json.dumps(debug))
        self.assertEqual(
            debug["final_answer_code_path"], "validated_structured_vision"
        )
        self.assertIn("Figure 13", answer)

    def test_failed_validation_never_returns_unvalidated_model_text(self):
        debug = {}
        with patch.object(structured, "_call_model", side_effect=["useful partial output", "still invalid"]):
            answer = structured.analyse_typed_image(
                Path("table.png"), "table", "Read Table 1", debug_info=debug
            )
        self.assertIn("Could not verify a structured reading", answer)
        self.assertNotIn("useful partial output", answer)
        self.assertNotIn("still invalid", answer)
        self.assertTrue(debug["validation_error"])
        self.assertEqual(debug["final_answer_path"], "grounded_caption_summary_fallback")

    def test_failed_figure_one_validation_returns_grounded_caption_summary(self):
        debug = {}
        evidence = (
            "TARGET FIGURE CAPTION (authoritative for this crop):\n"
            "Figure 1. Histological and schematic representation of adipose tissue.\n\n"
            "PAGE TEXT CROSS-CHECK:\nNearby discussion."
        )
        with patch.object(
            structured, "_call_model", side_effect=["{bad", "still invalid"]
        ):
            answer = structured.analyse_typed_image(
                Path("figure1.png"), "labelled_diagram", "Explain Figure 1",
                evidence_text=evidence, debug_info=debug,
            )
        self.assertIn("Grounded caption summary", answer)
        self.assertIn("Histological and schematic representation", answer)
        self.assertNotIn("still invalid", answer)
        self.assertNotIn("{bad", answer)

    def test_failed_circuit_json_never_displays_raw_topology(self):
        debug = {}
        with patch.object(
            structured, "_call_model",
            side_effect=['{"circuit_topology":"RE then RI', "still invalid"],
        ):
            answer = structured.analyse_typed_image(
                Path("circuit.png"), "labelled_diagram", "Explain this circuit",
                debug_info=debug,
            )
        self.assertIn("Could not verify the circuit topology", answer)
        self.assertNotIn("RE then RI", answer)
        self.assertEqual(debug["final_answer_path"], "could_not_verify_topology")

    def test_output_limit_supports_large_tables_and_multi_panel_graphs(self):
        self.assertGreaterEqual(structured.STRUCTURED_NUM_PREDICT, 1500)


if __name__ == "__main__":
    unittest.main()
