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
