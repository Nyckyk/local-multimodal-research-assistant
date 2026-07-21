from __future__ import annotations

from unittest.mock import patch

from services.ollama_service import generate_answer
from settings import NORMAL_NUM_PREDICT, SUMMARY_NUM_PREDICT


def _response(content, done_reason="stop"):
    return {
        "message": {"content": content},
        "done": True,
        "done_reason": done_reason,
    }


def test_summary_length_limit_continues_once_without_duplicate_text():
    initial = (
        "## Research question\nCompare two models.\n\n"
        "## Methods\nThe authors used grounded numerical evidence.\n\n"
        "## Main results\nIn steady-state conditions, both models"
    )
    continuation = (
        "agree. Transient predictions differ.\n\n"
        "## Limitations\nThe evidence reports modelling assumptions and no experimental data."
    )
    debug = {}
    with patch(
        "services.ollama_service.ollama.chat",
        side_effect=[_response(initial, "length"), _response(continuation)],
    ) as chat:
        answer = generate_answer(
            "Summarise the paper, including its research question, methods, "
            "main results and limitations.",
            "[DOCUMENT SUMMARY MODE]\nSource: paper.pdf, page 1\nEvidence",
            [],
            debug_info=debug,
        )

    assert chat.call_count == 2
    assert chat.call_args_list[0].kwargs["options"]["num_predict"] == SUMMARY_NUM_PREDICT
    assert answer.count("## Research question") == 1
    assert "both models agree" in answer
    assert "## Limitations" in answer
    assert debug["continuation_used"]
    assert debug["generation_attempts"][0]["done_reason"] == "length"
    assert debug["final_missing_sections"] == []
    assert debug["stream"] is False
    assert debug["streaming_chunks_lost"] is False


def test_complete_short_answer_does_not_trigger_continuation_or_summary_budget():
    debug = {}
    with patch(
        "services.ollama_service.ollama.chat",
        return_value=_response("The supplied evidence supports a concise answer."),
    ) as chat:
        answer = generate_answer("What is the result?", "Source evidence", [], debug)
    assert answer.endswith("answer.")
    assert chat.call_count == 1
    assert chat.call_args.kwargs["options"]["num_predict"] == NORMAL_NUM_PREDICT
    assert not debug["continuation_used"]


def test_missing_requested_summary_section_triggers_only_one_retry():
    complete_ending_but_missing = (
        "Research question: Q. Methods: M. Main results: R."
    )
    with patch(
        "services.ollama_service.ollama.chat",
        side_effect=[
            _response(complete_ending_but_missing),
            _response("Limitations: L."),
        ],
    ) as chat:
        answer = generate_answer(
            "Summarise the paper with the research question, methods, main results "
            "and limitations.",
            "[DOCUMENT SUMMARY MODE]\nEvidence",
            [],
        )
    assert chat.call_count == 2
    assert answer.count("Limitations") == 1


def test_validated_framework_items_are_completed_with_exact_grounded_names():
    context = (
        '[DOCUMENT SUMMARY MODE]\n[VALIDATED FRAMEWORK GROUNDING]\n'
        '{"named_items": ["genomic instability", "telomere attrition"], '
        '"categories": {}, "uncertain": []}\nEvidence'
    )
    debug = {}
    with patch(
        "services.ollama_service.ollama.chat",
        side_effect=[
            _response("The framework includes DNA damage and telomere attrition."),
            _response("The canonical term is genomic instability."),
        ],
    ) as chat:
        answer = generate_answer("Summarise the framework.", context, [], debug)

    assert chat.call_count == 2
    assert "genomic instability" in answer
    assert debug["initial_missing_grounded_items"] == ["genomic instability"]
    assert debug["final_missing_grounded_items"] == []
    assert debug["grounded_items_appended"] == []


def test_grounded_inferred_limitations_are_completed_and_clearly_labelled():
    items = [
        {"key": "numerical_2d", "statement": "The work uses a numerical 2D model rather than an in-vivo experiment."},
        {"key": "fixed_tissue_properties", "statement": "Tissue properties are fixed or unvarying in the model."},
        {"key": "no_phase_changes", "statement": "Phase changes are excluded."},
        {"key": "no_chemical_reactions", "statement": "Chemical reactions are excluded."},
        {"key": "local_thermal_equilibrium", "statement": "Local blood-tissue thermal equilibrium is assumed."},
        {"key": "uniform_incident_irradiance", "statement": "Incident irradiance is uniform across the exposure area."},
        {"key": "simplified_environment", "statement": "The environmental geometry is simplified and excludes surrounding walls or metallic enclosures."},
        {"key": "benchmark_validation", "statement": "Validation is against benchmarks or prior studies rather than new experimental human data."},
    ]
    context = (
        "[DOCUMENT SUMMARY MODE]\n[INFERRED LIMITATIONS GROUNDING]\n"
        + __import__("json").dumps({
            "items": items,
            "status": "inferred_from_stated_assumptions",
        })
        + "\nEvidence"
    )
    debug = {}
    with patch(
        "services.ollama_service.ollama.chat",
        side_effect=[
            _response(
                "Research question: Q. Methods: M. Main results: R. "
                "Limitations: The modelling constraints require care."
            ),
            _response("The available assumptions delimit interpretation."),
        ],
    ) as chat:
        answer = generate_answer(
            "Summarise the research question, methods, main results and limitations.",
            context,
            [],
            debug,
        )

    assert chat.call_count == 2
    assert "Limitations inferred from stated assumptions" in answer
    assert "numerical 2D model" in answer
    assert "new experimental human data" in answer
    assert debug["initial_missing_inferred_limitations"] == [
        item["key"] for item in items
    ]
    assert debug["final_missing_inferred_limitations"] == []


def test_grounded_transient_comparison_is_preserved_after_summary_generation():
    context = (
        "[DOCUMENT SUMMARY MODE]\n"
        "Source: paper.pdf, page 16\n"
        "It reveals that initially, TWMBT forecasts a lower heat rise than "
        "Pennes' equation. As exposure progresses toward a steady state, "
        "TWMBT predictions converge with Pennes' equation."
    )
    debug = {}
    with patch(
        "services.ollama_service.ollama.chat",
        return_value=_response(
            "Research question: Q. Methods: M. Main results: The models converge. "
            "Limitations: L."
        ),
    ):
        answer = generate_answer(
            "Summarise the research question, methods, main results and limitations.",
            context,
            [],
            debug,
        )
    assert "TWMBT initially forecasts lower heat rise than Pennes' equation" in answer
    assert debug["grounded_transient_comparison_appended"] is True


def test_multi_figure_completion_keeps_trends_attached_to_their_figure():
    context = (
        "[MULTI-FIGURE EVIDENCE MODE]\n"
        "Source: paper.pdf, page 9, section figure evidence\n"
        "Fig. 5 shows absorbed power dissipation near the incident boundary. "
        "As frequency increases, at 4 GHz the heated area becomes small.\n"
        "Source: paper.pdf, page 10, section figure evidence\n"
        "Fig. 6 shows isothermal contours. At 0.9 GHz, 1.8 GHz, 2.45 GHz and 4 GHz, the "
        "maximal temperature values are 39.12 °C, 39.23 °C, 39.35 °C and "
        "39.52 °C, respectively."
    )
    debug = {}
    with patch(
        "services.ollama_service.ollama.chat",
        return_value=_response(
            "Figure 5 shows absorbed power. Temperature diminishes as frequency "
            "increases. Figure 6 shows isotherms."
        ),
    ):
        answer = generate_answer("Compare the figure evidence.", context, [], debug)
    assert "Temperature diminishes" not in answer
    assert "Figure 5" in answer and "localized near the incident boundary" in answer
    assert "inference" in answer and "not a direct depth measurement" in answer
    assert "reported peak temperatures increase" in answer
    assert "39.12 °C at 0.9 GHz" in answer
    assert "39.52 °C at 4 GHz" in answer
    assert debug["multi_figure_contradiction_removed"] is True
    assert debug["multi_figure_depth_language_qualified"] is False
