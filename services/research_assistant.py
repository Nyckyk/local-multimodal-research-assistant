"""Reusable orchestration for one Research Assistant question.

The Streamlit UI and automated end-to-end evaluator both call this service so
scientific evaluation exercises the production retrieval and model path.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rag.retrieval import retrieve_context
from services.equation_service import analyse_resolved_equation
from services.multi_target import analyse_equation_chain, analyse_visual_targets
from services.ollama_service import generate_answer
from services.query_rewriter import rewrite_question
from services.visual_fallback import resolve_with_visual_fallback
from services.visual_locator import (
    format_resolution_problem,
    manual_visual_resolution,
    resolve_visual_target,
    resolve_visual_targets,
    should_activate_automatic_vision,
)
from services.visual_reference_parser import has_visual_reference
from services.visual_runtime import analyse_resolved_visual, build_visual_evidence
from services.vision_service import COULD_NOT_VERIFY_MESSAGE
from settings import MAX_HISTORY_MESSAGES


def _message_history(messages: list[dict]) -> list[dict[str, str]]:
    history = []
    # Streamlit's historical path counted the just-submitted user message in
    # MAX_HISTORY_MESSAGES, leaving this many prior messages for rewriting.
    history_limit = max(0, MAX_HISTORY_MESSAGES - 1)
    for message in messages[-history_limit:] if history_limit else []:
        if not isinstance(message, dict):
            continue
        role, content = message.get("role"), message.get("content")
        if isinstance(role, str) and isinstance(content, str):
            history.append({"role": role, "content": content})
    return history


def _previous_sources(messages: list[dict]) -> list[str]:
    values = []
    for message in messages:
        if not isinstance(message, dict) or not isinstance(message.get("sources"), list):
            continue
        values.extend(
            str(source.get("pdf", source.get("source", "")))
            for source in message["sources"] if isinstance(source, dict)
        )
    return values


def _previous_visual_evidence(messages: list[dict]) -> dict | None:
    return next(
        (
            message.get("evidence")
            for message in reversed(messages)
            if isinstance(message, dict)
            and isinstance(message.get("evidence"), dict)
            and message["evidence"].get("vision_result")
        ),
        None,
    )


@dataclass
class AssistantResponse:
    answer: str
    sources: list[dict]
    retrieval_question: str
    evidence: dict | None
    resolution: Any
    multi_resolutions: list[Any] = field(default_factory=list)
    visual_answer: str = ""
    vision_error: str = ""
    equation_error: str = ""
    used_vision: bool = False
    debug: dict[str, Any] = field(default_factory=dict)

    def assistant_message(self) -> dict:
        """Return the same history record persisted by Streamlit."""
        return {
            "role": "assistant",
            "content": self.answer,
            "sources": self.sources,
            "retrieval_question": self.retrieval_question,
            "vision_answer": self.visual_answer,
            "vision_error": self.vision_error,
            "evidence": self.evidence,
            "visual_target": (
                self.resolution.to_dict() if self.resolution.status == "resolved" else None
            ),
        }

    def to_dict(self) -> dict:
        selected_document = self.resolution.pdf_name or (
            self.evidence.get("pdf") if isinstance(self.evidence, dict) else None
        )
        if not selected_document:
            source_names = {
                str(row.get("pdf", row.get("source", "")))
                for row in self.sources if isinstance(row, dict)
                and row.get("pdf", row.get("source"))
            }
            selected_document = next(iter(source_names)) if len(source_names) == 1 else None
        vision_status = (
            "failed" if self.vision_error else "used" if self.used_vision else "not_used"
        )
        return {
            "answer": self.answer,
            "sources": self.sources,
            "retrieval_question": self.retrieval_question,
            "evidence": self.evidence,
            "resolution": self.resolution.to_dict(),
            "multi_resolutions": [item.to_dict() for item in self.multi_resolutions],
            "visual_answer": self.visual_answer,
            "vision_error": self.vision_error,
            "equation_error": self.equation_error,
            "used_vision": self.used_vision,
            "vision_status": vision_status,
            "selected_document": selected_document,
            "resolved_target_type": self.resolution.target_type,
            "resolved_target_number": self.resolution.target_number,
            "resolved_page": self.resolution.page_number,
            "final_answer_code_path": self.debug.get("final_answer_code_path"),
            "requested_answer_slots": self.debug.get("requested_answer_slots", []),
            "unresolved_answer_slots": self.debug.get("missing_answer_slots", []),
            "final_answer_evidence": self.debug.get("final_answer_evidence"),
            "debug": self.debug,
        }


class ResearchAssistant:
    """Production question pipeline with injected local model/index components."""

    def __init__(self, *, collection, embedder, reranker, visual_index: dict):
        self.collection = collection
        self.embedder = embedder
        self.reranker = reranker
        self.visual_index = visual_index

    def ask(
        self,
        question: str,
        *,
        conversation_messages: list[dict] | None = None,
        preferred_pdf: Path | str | None = None,
        automatic_visual_detection: bool = True,
        manual_pdf: Path | str | None = None,
        manual_page_number: int = 1,
        save_vision_crops: bool = False,
        allow_vision: bool = True,
    ) -> AssistantResponse:
        messages = list(conversation_messages or [])
        conversation_history = _message_history(messages)
        preferred_path = Path(preferred_pdf) if preferred_pdf else None
        manual_path = Path(manual_pdf) if manual_pdf else None

        retrieval_question = rewrite_question(
            question=question, conversation_history=conversation_history,
        )
        context, sources = retrieve_context(
            question=retrieval_question,
            previous_question="",
            collection=self.collection,
            embedder=self.embedder,
            reranker=self.reranker,
            selected_source=preferred_path.name if preferred_path else None,
        )

        visual_answer = equation_answer = multi_answer = ""
        vision_error = equation_error = ""
        vision_debug = {"_save_crops": save_vision_crops}
        equation_debug: dict = {}
        multi_debug = {"_save_crops": save_vision_crops}
        text_generation_debug: dict = {}
        resolution_fallback_debug: dict = {}
        multi_resolutions = []

        if manual_path is not None:
            resolution = manual_visual_resolution(
                manual_path, int(manual_page_number), question,
            )
        elif automatic_visual_detection:
            multi_resolutions = resolve_visual_targets(
                question=question,
                index=self.visual_index,
                selected_pdf=preferred_path,
                conversation_messages=messages,
                current_source_names=_previous_sources(messages),
                embedder=self.embedder,
            )
            resolution = multi_resolutions[0]
            if allow_vision and resolution.target_type != "equation" and has_visual_reference(question) and (
                resolution.status == "not_found"
                or (
                    resolution.status == "ambiguous"
                    and "confidence" in resolution.reason.casefold()
                )
            ):
                resolution = resolve_with_visual_fallback(
                    question=question,
                    base_resolution=resolution,
                    index=self.visual_index,
                    selected_pdf=preferred_path,
                    debug_info=resolution_fallback_debug,
                )
        else:
            resolution = resolve_visual_target(question, {"files": {}})

        use_vision = allow_vision and (
            (manual_path is not None and resolution.target_type != "equation")
            or should_activate_automatic_vision(question, resolution)
        )
        selected_pdf = Path(resolution.pdf_path) if resolution.pdf_path else None
        vision_page_number = resolution.page_number or 1

        if len(multi_resolutions) > 1:
            target_types = {item.target_type for item in multi_resolutions}
            try:
                if target_types == {"equation"}:
                    multi_answer = analyse_equation_chain(
                        question, multi_resolutions,
                        conversation_history=conversation_history,
                        debug_info=multi_debug,
                    )
                elif allow_vision and "equation" not in target_types:
                    multi_answer = analyse_visual_targets(
                        question, multi_resolutions, text_evidence=context,
                        conversation_history=conversation_history,
                        debug_info=multi_debug,
                    )
            except Exception as error:
                equation_error = str(error)

        if not multi_answer and resolution.status == "resolved" and resolution.target_type == "equation":
            try:
                equation_answer = analyse_resolved_equation(
                    question, resolution,
                    conversation_history=conversation_history,
                    debug_info=equation_debug,
                )
            except Exception as error:
                equation_error = str(error)

        if not multi_answer and use_vision and selected_pdf is not None:
            try:
                visual_answer = analyse_resolved_visual(
                    question=question,
                    resolution=resolution,
                    debug_info=vision_debug,
                    text_evidence=context,
                )
            except Exception as error:
                vision_error = str(error)

            if visual_answer:
                visual_context = (
                    "[VISUAL ANALYSIS - VALID FIGURE/TABLE EVIDENCE]\n"
                    f"Source: {selected_pdf.name}\n"
                    f"PDF page: {int(vision_page_number)}\n"
                    f"Result: {visual_answer}"
                )
                context = (
                    f"{visual_context}\n\n[EXTRACTED PDF TEXT]\n{context}"
                    if context else visual_context
                )

        evidence = None
        visual_reference_requested = has_visual_reference(question)
        previous_visual_evidence = _previous_visual_evidence(messages)

        if multi_answer:
            answer = multi_answer
            resolved_multi = [item for item in multi_resolutions if item.status == "resolved"]
            first = resolved_multi[0] if resolved_multi else resolution
            evidence = {
                "summary": "Grounded ordered multi-target analysis.",
                "analysis_kind": "Multi-target analysis",
                "pdf": first.pdf_name or "Unknown source",
                "page": int(first.page_number or 1),
                "visual_targets": [item.to_dict() for item in multi_resolutions],
            }
        elif equation_answer:
            answer = (
                f"{equation_answer}\n\nSource: **{resolution.pdf_name}**, "
                f"PDF page **{int(resolution.page_number)}**."
            )
            evidence = {
                "summary": "Grounded analysis of an explicitly numbered equation.",
                "analysis_kind": "Equation analysis",
                "pdf": resolution.pdf_name,
                "page": int(resolution.page_number),
                "visual_target": resolution.to_dict(),
            }
        elif visual_answer:
            cross_visual_comparison = bool(
                previous_visual_evidence
                and re.search(r"\b(?:compare|versus|vs\.?|difference)\b", question, re.I)
                and re.search(r"\b(?:it|that|previous|them)\b", question, re.I)
            )
            if cross_visual_comparison:
                comparison_context = (
                    "[PREVIOUS VALIDATED VISUAL ANALYSIS]\n"
                    f"Source: {previous_visual_evidence['pdf']}\n"
                    f"PDF page: {previous_visual_evidence['page']}\n"
                    f"Result: {previous_visual_evidence['vision_result']}\n\n{context}"
                )
                answer = generate_answer(
                    question=question,
                    context=comparison_context,
                    conversation_history=conversation_history,
                    debug_info=text_generation_debug,
                )
            else:
                answer = (
                    f"{visual_answer}\n\nSource: **{selected_pdf.name}**, "
                    f"PDF page **{int(vision_page_number)}**."
                )
            evidence = build_visual_evidence(resolution, visual_answer, vision_debug)
        elif resolution.status == "ambiguous" and visual_reference_requested:
            answer = format_resolution_problem(resolution)
        elif resolution.status == "not_found" and visual_reference_requested:
            answer = format_resolution_problem(resolution)
        elif use_vision and selected_pdf is not None and vision_error:
            answer = COULD_NOT_VERIFY_MESSAGE
        elif resolution.target_type == "equation" and equation_error:
            answer = "Could not verify the requested equation from indexed PDF text."
        elif not context:
            answer = "No relevant information was found in the indexed papers."
        else:
            answer = generate_answer(
                question=question,
                context=context,
                conversation_history=conversation_history,
                debug_info=text_generation_debug,
            )

        debug = {
            "original_question": question,
            "rewritten_retrieval_query": retrieval_question,
            "retrieved_chunks": sources,
            "vision": vision_debug,
            "equation": equation_debug,
            "multi_target": multi_debug,
            "resolution_fallback": resolution_fallback_debug,
            "text_generation": text_generation_debug,
            "final_answer_code_path": (
                vision_debug.get("final_answer_code_path")
                or vision_debug.get("final_answer_path")
                or multi_debug.get("final_answer_code_path")
                or text_generation_debug.get("final_answer_code_path")
                or ("resolved_equation" if equation_answer else "resolution_problem" if visual_reference_requested and resolution.status != "resolved" else "text_rag")
            ),
            "requested_answer_slots": (
                vision_debug.get("requested_answer_slots")
                or text_generation_debug.get("requested_answer_slots")
                or []
            ),
            "missing_answer_slots": (
                vision_debug.get("final_missing_answer_slots")
                or text_generation_debug.get("final_missing_answer_slots")
                or []
            ),
            "final_answer_evidence": (
                vision_debug.get("final_answer_evidence")
                or text_generation_debug.get("final_answer_evidence")
            ),
            "final_evidence_consistency_errors": (
                vision_debug.get("final_evidence_consistency_errors")
                or text_generation_debug.get("final_evidence_consistency_errors")
                or []
            ),
        }
        return AssistantResponse(
            answer=answer,
            sources=sources,
            retrieval_question=retrieval_question,
            evidence=evidence,
            resolution=resolution,
            multi_resolutions=multi_resolutions,
            visual_answer=visual_answer,
            vision_error=vision_error,
            equation_error=equation_error,
            used_vision=use_vision,
            debug=debug,
        )
