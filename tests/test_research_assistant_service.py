from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from services.research_assistant import ResearchAssistant
from services.visual_locator import VisualResolution


class Collection:
    pass


def resolved():
    return VisualResolution(
        status="resolved", pdf_path="paper.pdf", pdf_name="paper.pdf",
        page_number=3, target_type="figure", target_number="2",
    )


def test_shared_backend_text_path_uses_real_service_interfaces():
    resolution = VisualResolution(status="not_found", reason="No visual reference")
    with patch("services.research_assistant.rewrite_question", return_value="rewritten") as rewrite, patch(
        "services.research_assistant.retrieve_context",
        return_value=("grounded context", [{"source": "paper.pdf", "page": 2, "document": "text", "score": 1.0}]),
    ) as retrieve, patch(
        "services.research_assistant.resolve_visual_targets", return_value=[resolution],
    ), patch(
        "services.research_assistant.generate_answer", return_value="grounded answer",
    ) as generate:
        result = ResearchAssistant(
            collection=Collection(), embedder=object(), reranker=object(), visual_index={},
        ).ask("question", conversation_messages=[{"role": "user", "content": "prior"}])
    assert result.answer == "grounded answer"
    assert result.retrieval_question == "rewritten"
    assert result.debug["final_answer_code_path"] == "text_rag"
    assert rewrite.call_args.kwargs["conversation_history"] == [{"role": "user", "content": "prior"}]
    assert retrieve.call_args.kwargs["question"] == "rewritten"
    assert generate.call_args.kwargs["question"] == "question"


def test_shared_backend_visual_path_returns_evaluator_diagnostics():
    resolution = resolved()
    vision_debug = {
        "final_answer_code_path": "validated_typed_vision",
        "requested_answer_slots": ["comparison"],
        "final_missing_answer_slots": [],
        "final_answer_evidence": {"figure": "2"},
        "final_evidence_consistency_errors": [],
    }

    def visual(**kwargs):
        kwargs["debug_info"].update(vision_debug)
        return "visual answer"

    with patch("services.research_assistant.rewrite_question", return_value="question"), patch(
        "services.research_assistant.retrieve_context", return_value=("context", []),
    ), patch("services.research_assistant.resolve_visual_targets", return_value=[resolution]), patch(
        "services.research_assistant.should_activate_automatic_vision", return_value=True,
    ), patch("services.research_assistant.analyse_resolved_visual", side_effect=visual), patch(
        "services.research_assistant.build_visual_evidence", return_value={"pdf": "paper.pdf", "page": 3, "vision_result": "visual answer"},
    ):
        result = ResearchAssistant(
            collection=Collection(), embedder=object(), reranker=object(), visual_index={},
        ).ask("Figure 2 question")
    assert result.answer.startswith("visual answer")
    assert result.used_vision
    assert result.resolution is resolution
    assert result.debug["final_answer_code_path"] == "validated_typed_vision"
    assert result.debug["requested_answer_slots"] == ["comparison"]
    assert result.debug["final_answer_evidence"] == {"figure": "2"}
