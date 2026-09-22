from __future__ import annotations

import re
from unittest.mock import patch

from services.ollama_service import (
    _clean_generation_artifacts,
    _enforce_experimental_figure_provenance,
    generate_answer,
)
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


def test_methods_answer_uses_complete_output_budget():
    debug = {}
    with patch(
        "services.ollama_service.ollama.chat",
        return_value=_response("The exact method is reported."),
    ) as chat:
        generate_answer(
            "Give the exact Methods.",
            "[METHODS-AWARE RETRIEVAL]\nSource evidence", [], debug,
        )
    assert chat.call_args.kwargs["options"]["num_predict"] == SUMMARY_NUM_PREDICT


def test_grounded_slot_blocks_false_not_found_and_repairs_displayed_answer():
    context = (
        '[QUESTION COVERAGE]\nRequested answer slots: ["inclusion criteria"]\n'
        '[GROUNDED ANSWER SLOT EVIDENCE]\n'
        '{"inclusion criteria":{"slot":"inclusion criteria","status":"grounded",'
        '"evidence":[{"page":17,"source":"document_text","text":'
        '"For patient samples, circularity > 0.7 selected nuclei predominantly from '
        'hepatocytes rather than fibroblasts or immune cells."}]}}\n'
        'Source: paper.pdf, page 17\nThe same grounded Methods sentence.'
    )
    debug = {}
    with patch(
        "services.ollama_service.ollama.chat",
        side_effect=[
            _response("The paper does not mention a circularity threshold."),
            _response("The requested threshold is not available."),
        ],
    ):
        answer = generate_answer(
            "Why was the circularity threshold used and which cells did it include?",
            context, [], debug,
        )
    lowered = answer.casefold()
    assert "0.7" in answer and "hepatocytes" in lowered
    assert "fibroblasts" in lowered and "immune cells" in lowered
    assert "does not mention" not in lowered and "not available" not in lowered
    assert debug["grounded_slots_appended"] == ["inclusion criteria"]
    assert debug["final_grounded_slot_coverage_errors"] == []


def test_grounded_slot_removes_not_provided_claim():
    context = (
        '[GROUNDED ANSWER SLOT EVIDENCE]\n'
        '{"RF threshold":{"slot":"RF threshold","status":"grounded",'
        '"evidence":[{"page":16,"source":"document_text","text":'
        '"RF senescence probability values > 0.5 were considered senescent."}]}}'
    )
    with patch(
        "services.ollama_service.ollama.chat",
        return_value=_response("The RF threshold is not provided."),
    ):
        answer = generate_answer("What RF threshold was used?", context, [])
    assert "not provided" not in answer.casefold()
    assert "> 0.5" in answer


def test_compound_screen_continuation_grammar_is_cleaned():
    cleaned, changed = _clean_generation_artifacts(
        "Finally, **1 were identified as selective and 18 were shared."
    )
    assert changed
    assert "1 was identified" in cleaned
    assert cleaned.count("**") % 2 == 0


def test_whole_document_provenance_rejects_invented_supplementary_figure():
    context = (
        '[WHOLE-DOCUMENT EXPERIMENTAL-DOMAIN EVIDENCE]\n'
        '[{"document":"paper.pdf","figure_number":"8","figure_label":"Figure 8",'
        '"panel":null,"page":11,"experimental_domain":"mouse_animal_tissue",'
        '"species":"mouse","sample_type":"liver","source_text":"caption",'
        '"source_provenance":["full_caption"]}]\nEvidence'
    )
    answer, invalid, appended = _enforce_experimental_figure_provenance(
        context, "Mouse ageing is shown in implied Figure S10.",
    )
    assert invalid == ["Figure S10"]
    assert "Figure S10" not in answer
    assert "Figure 8" in answer and appended


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


def test_explicit_plural_limitations_are_completed_without_losing_future_work():
    items = [
        {
            "key": "explicit_runtime",
            "statement": "The hybrid method is more time consuming than the comparison method.",
            "evidence": ["The simulation took approximately three times as long at the same DOF."],
        },
        {
            "key": "explicit_mesh",
            "statement": "Its mesh extraction algorithm is more complex than the comparison method.",
            "evidence": [],
        },
    ]
    context = (
        "[SECTION-AWARE AUTHOR EVIDENCE]\n[EXPLICIT AUTHOR LIMITATIONS]\n"
        + __import__("json").dumps({"items": items, "status": "explicit_author_limitations"})
        + "\nThe authors propose improved mesh generation as future work."
    )
    debug = {}
    with patch(
        "services.ollama_service.ollama.chat",
        side_effect=[
            _response(
                "The authors state that the hybrid method is more time consuming. "
                "Future work will improve mesh generation."
            ),
            _response("Its mesh extraction algorithm is more complex than the comparison method."),
        ],
    ) as chat:
        answer = generate_answer(
            "What limitations do the authors identify, and what future work do they propose?",
            context,
            [],
            debug,
        )
    assert chat.call_count == 2
    assert "more time consuming" in answer
    assert "mesh extraction algorithm is more complex" in answer
    assert "Future work" in answer
    assert "approximately three times" in answer
    assert debug["explicit_runtime_examples_appended"]
    assert debug["final_missing_explicit_limitations"] == []


def test_explicit_limitation_preserves_other_tissues_scope():
    statement = (
        "The TSS might need adaptation to identify senescence in other tissues."
    )
    context = (
        "[SECTION-AWARE AUTHOR EVIDENCE]\n[EXPLICIT AUTHOR LIMITATIONS]\n"
        + __import__("json").dumps({
            "items": [{"key": "other_tissues", "statement": statement}],
            "status": "explicit_author_limitations",
        })
    )
    with patch(
        "services.ollama_service.ollama.chat",
        return_value=_response(
            "The TSS might need adaptation for other types of senescent cells."
        ),
    ):
        answer = generate_answer("What limitations do the authors state?", context, [])
    assert "other tissues" in answer.casefold()


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


def test_summary_methods_complete_explicit_source_method_terminology():
    context = (
        "[DOCUMENT SUMMARY MODE]\n"
        "Source: paper.pdf, page 1, section abstract\n"
        "The equations use the finite element method (FEM) for solving."
    )
    debug = {}
    with patch(
        "services.ollama_service.ollama.chat",
        side_effect=[
            _response("Research question: Q. Methods: Numerical solution. Main results: R."),
            _response("The source method should also be named explicitly."),
        ],
    ) as chat:
        answer = generate_answer(
            "Summarise the research question, methods and main results.",
            context, [], debug,
        )
    assert chat.call_count == 2
    assert "finite element method" in answer.casefold()
    assert debug["explicit_method_terms_appended"] == ["FEM"]


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


def test_absorbed_power_alone_does_not_make_heating_depth_a_direct_measurement():
    context = (
        "[MULTI-FIGURE EVIDENCE MODE]\n"
        "Source: paper.pdf, page 9\n"
        "Fig. 5 shows absorbed power concentrated near the incident boundary."
    )
    debug = {}
    with patch(
        "services.ollama_service.ollama.chat",
        return_value=_response(
            "The absorbed-power distribution is a proxy for heating depth."
        ),
    ):
        answer = generate_answer("Which figure shows heating depth?", context, [], debug)
    depth_sentence = next(
        sentence for sentence in re.split(r"(?<=[.!?])\s+", answer)
        if "heating depth" in sentence.casefold()
    )
    assert "inferred heating depth from spatial absorbed-power contours" in depth_sentence
    assert debug["multi_figure_depth_language_qualified"] is True
