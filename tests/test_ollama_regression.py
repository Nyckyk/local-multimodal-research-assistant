from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

import pytest

from rag.database import get_collection
from rag.embeddings import load_embedder, load_reranker
from rag.retrieval import retrieve_context
from services.ollama_service import generate_answer
from services.structured_vision import StructuredOutputError, TruncatedJSONError
from services.visual_index import load_or_build_visual_index
from services.visual_locator import resolve_visual_target
from services.vision_service import analyse_pdf_page
from settings import PAPERS_FOLDER


pytestmark = pytest.mark.ollama


def _normal(value) -> str:
    text = unicodedata.normalize("NFKC", str(value)).casefold()
    text = text.replace("–", "-").replace("—", "-").replace("‑", "-")
    return re.sub(r"\s+", " ", text)


def _contains(haystack, needle) -> bool:
    normalized_haystack = re.sub(r"[^a-z0-9]+", " ", _normal(haystack))
    normalized_needle = re.sub(r"[^a-z0-9]+", " ", _normal(needle)).strip()
    return normalized_needle in normalized_haystack


def _failure_kind(error: BaseException) -> str:
    if isinstance(error, AssertionError):
        return "assertion_failure"
    if isinstance(error, TruncatedJSONError):
        return "parser_failure_truncated_json"
    if isinstance(error, StructuredOutputError):
        return "validation_failure"
    if isinstance(error, (TimeoutError, pytest.fail.Exception)) and "timeout" in str(error).lower():
        return "timeout"
    if "ollama" in type(error).__module__.lower() or "model" in str(error).lower():
        return "model_failure"
    return "test_error"


def _write_debug(artifact_writer, case_id, debug, error=None, answer=""):
    raw_parts = []
    for key in ("initial_response", "raw_vision_response", "retry_response"):
        value = debug.get(key)
        if isinstance(value, str) and value and value not in raw_parts:
            raw_parts.append(value)
    raw = "\n\n--- MODEL RESPONSE ---\n\n".join(raw_parts) or answer
    structured = debug.get("validated_json") or {
        key: value for key, value in debug.items()
        if key not in {"raw_vision_response", "retry_response", "raw_panel_responses"}
    }
    failure = None
    if error is not None:
        failure = {
            "failure_kind": _failure_kind(error),
            "error_type": type(error).__name__,
            "message": str(error),
            "validation_error": debug.get("validation_error", ""),
            "final_answer_path": debug.get("final_answer_path", ""),
        }
    artifact_writer(case_id, raw=raw, structured=structured, failure=failure)


def _case_by_id(cases, case_id):
    return next(case for case in cases if case["case_id"] == case_id)


@pytest.mark.parametrize(
    "case_id",
    [
            "hallmarks_document_summary",
            "figure_6_auto_hallmarks",
        "figure_1_labelled_diagram",
        "figure_3_circuit",
        "figure_9_six_panel_bode_graph",
        "figure_13_nyquist_graph",
        "table_1",
    ],
)
def test_real_local_regression_case(
    case_id, regression_cases, require_ollama_models, artifact_writer
):
    selected = __import__("os").environ.get("REGRESSION_CASE_ID")
    if selected and selected != case_id:
        pytest.skip(f"runner selected {selected}")
    case = _case_by_id(regression_cases, case_id)
    if case["expected_response_type"] == "rag_summary":
        _run_summary_case(case, artifact_writer)
    else:
        _run_vision_case(case, artifact_writer)


def _run_summary_case(case, artifact_writer):
    debug = {}
    answer = ""
    try:
        context, sources = retrieve_context(
            case["question"], "", get_collection(), load_embedder(), load_reranker()
        )
        debug.update({"context": context, "sources": sources})
        assert sources, "RAG returned no evidence"
        answer = generate_answer(case["question"], context, [])
        for fact in case["required_facts"]:
            assert _contains(answer, fact), f"answer missing required fact: {fact}"
        assert not re.search(
            r"(?:original|nine|primary|antagonistic|integrative)\s+(?:named\s+)?hallmarks?[^.]{0,100}\binflammation\b",
            _normal(answer),
        ), "inflammation was presented as an original hallmark"
        pages = {source["page"] for source in sources}
        expected = case["expected_structured_fields"]
        assert len(sources) >= expected["minimum_chunks"]
        assert len(pages) >= expected["minimum_pages"]
        assert not any(
            source.get("section") == "excluded" or "references" in _normal(source["document"][:80])
            for source in sources
        ), "bibliography evidence was selected"
        retrieval_debug = sources[0].get("retrieval_debug", {})
        grounding = retrieval_debug.get("framework_grounding", {})
        actual_categories = grounding.get("categories", {})
        for group in ("primary", "antagonistic", "integrative"):
            assert {
                re.sub(r"[^a-z0-9]+", "", _normal(item))
                for item in actual_categories.get(group, [])
            } == {
                re.sub(r"[^a-z0-9]+", "", _normal(item))
                for item in expected[group]
            }, f"validated framework category mismatch: {group}"
        structured = {
            "answer": answer,
            "sources": sources,
            "retrieval_debug": retrieval_debug,
            "document_summary_mode": "[DOCUMENT SUMMARY MODE]" in context,
        }
        artifact_writer(case["case_id"], raw=answer, structured=structured)
    except BaseException as error:
        debug["answer"] = answer
        _write_debug(artifact_writer, case["case_id"], debug, error, answer)
        raise


def _run_vision_case(case, artifact_writer):
    debug = {"_save_crops": True}
    answer = ""
    try:
        resolution = resolve_visual_target(
            case["question"], load_or_build_visual_index()
        )
        assert resolution.status == "resolved", resolution.to_dict()
        assert resolution.pdf_name == case["pdf_filename"]
        assert resolution.page_number == int(case["pdf_page"])
        pdf_path = Path(resolution.pdf_path)
        answer = analyse_pdf_page(
            pdf_path, int(resolution.page_number), case["question"], debug_info=debug
        )
        debug["resolved_visual_target"] = resolution.to_dict()
        grouped = case["case_id"] == "figure_6_auto_hallmarks"
        value = debug.get("normalized_json" if grouped else "validated_json")
        assert isinstance(value, dict), "validated structured output was not produced"
        expected_path = case["expected_structured_fields"].get(
            "final_answer_path", "validated_typed_vision"
        )
        assert debug.get("final_answer_path") == expected_path
        if case["case_id"] == "figure_1_labelled_diagram":
            _assert_figure_1(case, value)
        elif case["case_id"] == "figure_6_auto_hallmarks":
            _assert_figure_6(case, value)
        elif case["case_id"] == "figure_3_circuit":
            _assert_figure_3(case, value)
        elif case["case_id"] == "figure_9_six_panel_bode_graph":
            _assert_figure_9(case, value)
        elif case["case_id"] == "figure_13_nyquist_graph":
            _assert_figure_13(case, value, answer)
        elif case["case_id"] == "table_1":
            _assert_table_1(case, value)
        _write_debug(artifact_writer, case["case_id"], debug, answer=answer)
    except BaseException as error:
        _write_debug(artifact_writer, case["case_id"], debug, error, answer)
        raise


def _assert_required_and_prohibited(case, value):
    rendered = json.dumps(value, ensure_ascii=False)
    for fact in case["required_facts"]:
        assert _contains(rendered, fact), f"missing required fact: {fact}"
    for fact in case["prohibited_facts"]:
        assert not re.search(rf"\b{re.escape(_normal(fact))}\b", _normal(rendered)), (
            f"prohibited fact present: {fact}"
        )


def _assert_figure_1(case, value):
    _assert_required_and_prohibited(case, value)
    relationships = _normal(value.get("spatial_relationships", []))
    assert "intracellular" in relationships and ("within" in relationships or "inside" in relationships)
    assert "interstitial" in relationships and ("surround" in relationships or "between" in relationships)
    assert "intravascular" in relationships and ("vessel" in relationships or "vascular" in relationships)


def _assert_figure_6(case, value):
    expected = case["expected_structured_fields"]
    _assert_required_and_prohibited(case, value)
    for group in ("primary", "antagonistic", "integrative"):
        assert {
            re.sub(r"[^a-z0-9]+", "", _normal(item))
            for item in value[group]
        } == {
            re.sub(r"[^a-z0-9]+", "", _normal(item))
            for item in expected[group]
        }
    assert value["uncertain"] == []


def _assert_figure_3(case, value):
    _assert_required_and_prohibited(case, value)
    topology = value["circuit_topology"]
    branches = topology["branches"]
    component_sets = [
        {re.sub(r"[^a-z0-9]+", "", _normal(item)) for item in branch["components"]}
        for branch in branches
    ]
    assert {"re"} in component_sets
    assert {"c", "ri"} in component_sets
    assert any(len(group) >= 2 for group in topology["parallel_branch_sets"])
    parallel = [
        next(branch for branch in branches if branch["id"] == branch_id)
        for branch_id in topology["parallel_branch_sets"][0]
    ]
    assert len({frozenset((b["start_node"], b["end_node"])) for b in parallel}) == 1


def _assert_figure_9(case, value):
    _assert_required_and_prohibited(case, value)
    panels = value["panels"]
    assert len(panels) == 6
    assert all(_contains(panel["x_axis"]["label"], "frequency") for panel in panels)
    assert all(_normal(panel["x_axis"]["scale"]) == "log" for panel in panels)
    kinds = [_normal(panel["graph_kind"]) for panel in panels]
    assert sum("magnitude" in kind for kind in kinds) == 3
    assert sum("phase" in kind for kind in kinds) == 3
    magnitude_labels = [panel["y_axis"]["label"] for panel in panels if "magnitude" in _normal(panel["graph_kind"])]
    assert all(_contains(label, "Zfat") for label in magnitude_labels)
    comparisons = value["comparisons"]
    assert comparisons["magnitude_order_high_to_low"] == ["Group 3", "Group 1", "Group 2"]
    assert comparisons["greatest_phase_complexity_group"] == "Group 2"


def _assert_figure_13(case, value, answer):
    rendered = json.dumps(value, ensure_ascii=False)
    for fact in ("Sample 10", "Sample 7"):
        assert _contains(rendered, fact)
    assert len(value["panels"]) == 2
    assert all(_normal(panel["x_axis"]["scale"]) == "linear" for panel in value["panels"])
    assert all(_normal(panel["y_axis"]["scale"]) == "linear" for panel in value["panels"])
    for panel in value["panels"]:
        fit_text = json.dumps(
            [panel.get("shape_features", []), panel.get("visible_trends", [])],
            ensure_ascii=False,
        )
        assert any(
            _contains(fit_text, phrase)
            for phrase in ("good", "closely follow", "nearly indistinguishable")
        ), f"fit quality was not described as good for {panel.get('group')}"
    assert _contains(rendered + answer, "right") or _contains(rendered + answer, "high Re(Z)")
    for phrase in case["prohibited_facts"]:
        assert not _contains(rendered + answer, phrase), f"unsupported direction inferred: {phrase}"


def _number(value):
    return float(str(value).replace(",", "").strip())


def _assert_table_1(case, value):
    expected = case["expected_structured_fields"]
    assert len(value["columns"]) == 5
    def semantic_column(column):
        return _normal(re.sub(r"\s*\([^)]*\)\s*$", "", str(column)))

    assert [semantic_column(column) for column in value["columns"]] == [
        semantic_column(column) for column in expected["columns"]
    ]
    assert len(value["rows"]) == 10
    actual_rows = [[_number(cell) for cell in row] for row in value["rows"]]
    expected_rows = [[float(cell) for cell in row] for row in expected["rows"]]
    assert actual_rows == expected_rows
    assert value["unreadable_cells"] == []
    densities = [row[-1] for row in actual_rows]
    assert densities == sorted(densities)


def test_automatic_resolution_leaves_duplicate_figure_one_ambiguous():
    result = resolve_visual_target("Explain Figure 1.", load_or_build_visual_index())
    assert result.status == "ambiguous"
    assert len({candidate["pdf_name"] for candidate in result.candidates}) >= 2


def test_automatic_resolution_retains_figure_for_panel_followup():
    index = load_or_build_visual_index()
    first = resolve_visual_target("Compare Groups 1, 2 and 3 in Figure 9.", index)
    assert first.status == "resolved"
    messages = [{
        "role": "assistant",
        "content": "Validated Figure 9 answer",
        "visual_target": first.to_dict(),
    }]
    followup = resolve_visual_target(
        "What about panel d?", index, conversation_messages=messages
    )
    assert followup.status == "resolved"
    assert followup.pdf_name == first.pdf_name
    assert followup.page_number == first.page_number
    assert followup.target_number == "9"
    assert followup.panel == "d"
