from pathlib import Path
from unittest.mock import patch

from services.multi_target import analyse_equation_chain, analyse_visual_targets
from services.visual_locator import VisualResolution


def _resolved(kind, number, page):
    return VisualResolution(
        status="resolved", pdf_path=str(Path("paper.pdf")), pdf_name="paper.pdf",
        page_number=page, target_type=kind, target_number=number,
        caption=f"Figure {number}. anisotropic four-layer realistic model",
        confidence=1.0,
    )


def test_equation_chain_uses_all_ordered_evidence_and_never_false_absence():
    values = [_resolved("equation", str(number), 5) for number in (18, 19, 20)]
    seen = {}

    def answer(question, context, history, debug_info=None):
        seen["context"] = context
        return "Equations 18–20 do not exist."

    debug = {}
    with patch("services.multi_target.build_equation_evidence", side_effect=lambda path, page, number: f"evidence {number}"), patch(
        "services.multi_target.generate_answer", side_effect=answer
    ):
        result = analyse_equation_chain("Explain Equations 18–20.", values, debug_info=debug)
    assert [seen["context"].index(f"EQUATION {number}") for number in (18, 19, 20)] == sorted(
        seen["context"].index(f"EQUATION {number}") for number in (18, 19, 20)
    )
    assert "do not exist" not in result
    assert debug["final_answer_path"] == "validated_multi_equation_analysis"


def test_multi_visual_synthesis_uses_only_individually_validated_targets():
    values = [_resolved("figure", "9", 15), _resolved("figure", "10", 16)]

    def visual(question, resolution, text_evidence="", debug_info=None):
        debug_info.update({
            "final_answer_path": "validated_typed_vision",
            "validated_json": {"figure_number": resolution.target_number},
        })
        return f"validated figure {resolution.target_number}"

    debug = {}
    with patch("services.multi_target.analyse_resolved_visual", side_effect=visual), patch(
        "services.multi_target.generate_answer", return_value="Validated comparison."
    ):
        answer = analyse_visual_targets("Compare Figures 9 and 10.", values, debug_info=debug)
    assert answer == "Validated comparison."
    assert debug["final_answer_path"] == "validated_multi_visual_synthesis"
    assert [row["target_number"] for row in debug["targets"]] == ["9", "10"]


def test_multi_visual_partial_failure_retains_valid_target_and_names_missing():
    values = [_resolved("figure", "9", 15), VisualResolution(
        status="not_found", target_type="figure", target_number="10"
    )]

    def visual(question, resolution, text_evidence="", debug_info=None):
        debug_info.update({"final_answer_path": "validated_typed_vision", "validated_json": {}})
        return "validated figure 9"

    with patch("services.multi_target.analyse_resolved_visual", side_effect=visual), patch(
        "services.multi_target.generate_answer", return_value="Figure 9 result."
    ):
        answer = analyse_visual_targets("Compare Figures 9 and 10.", values)
    assert "Figure 9 result" in answer
    assert "Could not verify: Figure 10" in answer


def test_multi_visual_synthesis_uses_target_metric_semantics_and_author_exception():
    values = [_resolved("figure", "9", 15), _resolved("figure", "10", 16)]
    values[1].nearby_text = (
        "The realistic-head method outperforms PI-FEM regarding RDM. "
        "With regard to MAG, hybrid BE-FE outperforms PI-FEM in both directions "
        "except at 98% source eccentricity."
    )

    def visual(question, resolution, text_evidence="", debug_info=None):
        debug_info.update({
            "final_answer_path": "validated_typed_vision",
            "validated_json": {
                "figure_number": resolution.target_number,
                "metric_semantics": {
                    "RDM": {"objective": "target_value", "target_value": 0.0},
                    "MAG": {"objective": "target_value", "target_value": 1.0},
                },
                "grounded_trends": {},
            },
        })
        return f"validated figure {resolution.target_number}"

    wrong = (
        "Hybrid BE-FE wins MAG at every eccentricity because higher MAG is superior. "
        "Both metrics monotonically increase with a widening gap."
    )
    debug = {}
    with patch("services.multi_target.analyse_resolved_visual", side_effect=visual), patch(
        "services.multi_target.generate_answer", return_value=wrong
    ) as generated:
        answer = analyse_visual_targets("Compare Figures 9 and 10.", values, debug_info=debug)
    supplied = generated.call_args.args[1]
    assert '"target_value": 1.0' in supplied
    assert "higher MAG is superior" not in answer
    assert "monotonically" not in answer and "widening gap" not in answer
    assert "closeness to 1" in answer
    assert "except at 98% source eccentricity" in answer
    assert debug["explicit_metric_exceptions"]
