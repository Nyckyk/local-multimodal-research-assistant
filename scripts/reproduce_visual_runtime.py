"""Reproduce the Streamlit Figure 3 visual-analysis runtime path locally."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from rag.database import get_collection
from rag.embeddings import load_embedder, load_reranker
from rag.retrieval import retrieve_context
from services.query_rewriter import rewrite_question
from services.structured_vision import parse_json_response
from services.visual_index import load_or_build_visual_index
from services.visual_locator import resolve_visual_target
from services.visual_runtime import analyse_resolved_visual
from services.vision_service import COULD_NOT_VERIFY_MESSAGE


QUESTION = (
    "Using Figure 3, explain what RE, RI and C represent and how they are "
    "connected."
)


def _parsed(raw: str):
    if not raw:
        return None
    try:
        return parse_json_response(raw)
    except Exception as error:
        return {"parse_error": str(error), "raw": raw}


def _show(title: str, value) -> None:
    print(f"\n=== {title} ===")
    if isinstance(value, (dict, list)):
        print(json.dumps(value, indent=2, ensure_ascii=False, default=str))
    else:
        print(value if value not in (None, "") else "<none>")


def reproduce() -> tuple[str, dict]:
    embedder = load_embedder()
    reranker = load_reranker()
    retrieval_question = rewrite_question(QUESTION, [])
    context, _ = retrieve_context(
        question=retrieval_question,
        previous_question="",
        collection=get_collection(),
        embedder=embedder,
        reranker=reranker,
    )
    resolution = resolve_visual_target(
        question=QUESTION,
        index=load_or_build_visual_index(),
        conversation_messages=[],
        current_source_names=[],
        embedder=embedder,
    )
    if resolution.status != "resolved":
        raise RuntimeError(f"Figure 3 did not resolve: {resolution.to_dict()}")
    debug = {"_save_crops": True}
    runtime_error = ""
    try:
        answer = analyse_resolved_visual(
            question=QUESTION,
            resolution=resolution,
            debug_info=debug,
            text_evidence=context,
        )
    except Exception as error:
        runtime_error = f"{type(error).__name__}: {error}"
        answer = COULD_NOT_VERIFY_MESSAGE

    initial_raw = debug.get("initial_response") or debug.get(
        "raw_vision_response", ""
    )
    retry_raw = debug.get("retry_response", "")
    repaired = debug.get("repaired_json") or _parsed(retry_raw)
    _show("RESOLVED TARGET", resolution.to_dict())
    _show("RAW MODEL OUTPUT", initial_raw)
    _show(
        "PARSED STRUCTURED OBJECT",
        debug.get("initial_parsed_json") or _parsed(initial_raw),
    )
    _show(
        "INITIAL VALIDATION ERRORS",
        debug.get("initial_validation_errors") or debug.get("validation_error"),
    )
    _show("REPAIRED STRUCTURED OBJECT", repaired)
    _show("REPAIRED VALIDATION RESULT", {
        "status": debug.get("repaired_validation_result"),
        "validated_json": debug.get("validated_json"),
    })
    _show("RUNTIME ERROR", runtime_error)
    _show("FINAL RENDERED ANSWER", answer)
    _show(
        "FINAL ANSWER CODE PATH",
        debug.get("final_answer_code_path") or debug.get("final_answer_path"),
    )
    return answer, debug


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    reproduce()
