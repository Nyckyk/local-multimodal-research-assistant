from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from unittest.mock import patch

import pytest

from rag.database import get_collection
from rag.embeddings import load_embedder, load_reranker
from rag.retrieval import retrieve_context
from services.equation_service import analyse_resolved_equation
from services.ollama_service import generate_answer
from services.structured_vision import StructuredOutputError, TruncatedJSONError
from services.visual_index import load_or_build_visual_index
from services.visual_locator import resolve_visual_target
from services.visual_runtime import analyse_resolved_visual, build_visual_evidence
from settings import PAPERS_FOLDER


pytestmark = pytest.mark.ollama


def _normal(value) -> str:
    text = unicodedata.normalize("NFKC", str(value)).casefold()
    for symbol, name in {
        "τ": " tau ", "ρ": " rho ", "ω": " omega ",
        "∇": " nabla ", "∂": " partial ",
    }.items():
        text = text.replace(symbol, name)
    # Mathematical subscripts and common LaTeX wrappers are presentation
    # variants, not different scientific facts (Qmet == Q_{met}).
    text = re.sub(r"([a-z])_\{?([a-z]+)\}?", r"\1\2", text)
    text = re.sub(r"\\(?:mathrm|text|operatorname)\s*\{([^{}]+)\}", r"\1", text)
    text = re.sub(r"\\([a-z]+)", r"\1", text)
    text = re.sub(r"\b(?:is\s+)?assumed\s+to\s+be\s+zero\b", "assumed zero", text)
    text = re.sub(r"\bpower\s+dissipation(?:\s+density|\s+profiles?)?\b", "absorbed power", text)
    text = text.replace("–", "-").replace("—", "-").replace("‑", "-")
    return re.sub(r"\s+", " ", text)


def _contains(haystack, needle) -> bool:
    normalized_haystack = re.sub(r"[^a-z0-9]+", " ", _normal(haystack))
    normalized_needle = re.sub(r"[^a-z0-9]+", " ", _normal(needle)).strip()
    if normalized_needle in normalized_haystack:
        return True
    return (
        re.sub(r"[^a-z0-9]+", "", normalized_needle)
        in re.sub(r"[^a-z0-9]+", "", normalized_haystack)
    )


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
    structured = debug.get("final_structured_output") or debug.get("validated_json") or {
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
            "thermal_document_summary",
            "thermal_figure_2_boundary_conditions",
            "thermal_table_2",
            "thermal_table_3_text_fallback",
            "thermal_equation_8",
            "thermal_multi_figure_frequency_evidence",
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
    elif case["expected_response_type"] == "rag_multi_figure":
        _run_multi_figure_case(case, artifact_writer)
    elif case["expected_response_type"] == "equation":
        _run_equation_case(case, artifact_writer)
    else:
        _run_vision_case(case, artifact_writer)


def _run_summary_case(case, artifact_writer):
    debug = {}
    answer = ""
    try:
        context, sources = retrieve_context(
            case["question"], "", get_collection(), load_embedder(), load_reranker(),
            selected_source=case["pdf_filename"],
        )
        debug.update({"context": context, "sources": sources})
        assert sources, "RAG returned no evidence"
        answer = generate_answer(case["question"], context, [], debug_info=debug)
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
        if all(group in expected for group in ("primary", "antagonistic", "integrative")):
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
        for section in expected.get("requested_sections", []):
            assert _contains(answer, section), f"summary section missing: {section}"
        if case["case_id"] == "thermal_document_summary":
            assert _contains(answer, "limitations inferred from stated assumptions")
            normalized_answer = _normal(answer)
            limitation_patterns = {
                "numerical 2D model": r"(?:numerical.{0,30}2d|2d.{0,30}(?:numerical|model))",
                "fixed tissue properties": r"(?:fixed.{0,50}tissue properties|tissue properties.{0,50}fixed)",
                "phase changes": r"phase changes",
                "chemical reactions": r"chemical reactions",
                "blood-tissue thermal equilibrium": r"blood.tissue thermal equilibrium",
                "uniform incident irradiance": r"(?:uniform.{0,50}incident irradiance|incident irradiance.{0,50}uniform)",
                "simplified environment": r"(?:walls|metallic enclosures|simplified environmental geometry)",
                "no new human experiment": r"new experimental human data",
            }
            for limitation, pattern in limitation_patterns.items():
                assert re.search(pattern, normalized_answer, re.DOTALL), (
                    f"thermal summary missing grounded inferred limitation: {limitation}"
                )
            assert debug["final_missing_inferred_limitations"] == []
        expected_stem = Path(case["pdf_filename"]).stem.casefold()
        assert all(
            Path(str(source["source"])).stem.casefold() == expected_stem
            for source in sources
        ), "summary evidence came from the wrong document"
        structured = {
            "answer": answer,
            "sources": sources,
            "retrieval_debug": retrieval_debug,
            "document_summary_mode": "[DOCUMENT SUMMARY MODE]" in context,
            "generation_debug": debug,
        }
        artifact_writer(case["case_id"], raw=answer, structured=structured)
    except BaseException as error:
        debug["answer"] = answer
        _write_debug(artifact_writer, case["case_id"], debug, error, answer)
        raise


def _run_multi_figure_case(case, artifact_writer):
    debug = {}
    answer = ""
    try:
        context, sources = retrieve_context(
            case["question"], "", get_collection(), load_embedder(), load_reranker()
        )
        assert "[MULTI-FIGURE EVIDENCE MODE]" in context
        assert sources
        answer = generate_answer(case["question"], context, [], debug_info=debug)
        for fact in case["required_facts"]:
            assert _contains(answer, fact), f"answer missing required fact: {fact}"
        identifiers = {
            identifier for source in sources
            for identifier in source.get("figure_identifiers", [])
        }
        assert len(identifiers) >= case["expected_structured_fields"]["minimum_figures"]
        if case["case_id"] == "thermal_multi_figure_frequency_evidence":
            normalized = _normal(answer)
            assert re.search(r"39\.12.{0,240}39\.52", normalized, re.DOTALL), (
                "Figure 6 peak temperatures were not preserved"
            )
            assert re.search(
                r"(?:locali[sz]|concentrat).{0,180}(?:incident|exposure)",
                normalized,
                re.DOTALL,
            ), "Figure 5 spatial localisation was not described"
            assert not re.search(
                r"temperature.{0,100}(?:diminish|decreas).{0,100}frequency.{0,40}increas|"
                r"frequency.{0,40}increas.{0,100}temperature.{0,100}(?:diminish|decreas)",
                normalized,
                re.DOTALL,
            ), "an absorbed-power trend was incorrectly presented as a temperature trend"
            for sentence in re.split(r"(?<=[.!?])\s+", normalized):
                if "heating depth" in sentence or "penetration" in sentence:
                    assert any(
                        qualifier in sentence
                        for qualifier in ("infer", "contour", "absorbed power", "spatial")
                    ), "depth/localisation claim was not qualified as an inference"
        expected_stem = Path(case["pdf_filename"]).stem.casefold()
        assert all(Path(str(source["source"])).stem.casefold() == expected_stem for source in sources)
        artifact_writer(case["case_id"], raw=answer, structured={
            "answer": answer, "sources": sources, "generation_debug": debug,
        })
    except BaseException as error:
        debug.update({"answer": answer})
        _write_debug(artifact_writer, case["case_id"], debug, error, answer)
        raise


def _run_equation_case(case, artifact_writer):
    debug = {}
    answer = ""
    try:
        thermal_name = case["pdf_filename"]
        previous = [{
            "role": "assistant", "content": "Prior Figure 5 analysis.",
            "visual_target": {
                "status": "resolved", "target_type": "figure", "target_number": "5",
                "pdf_name": thermal_name,
                "pdf_path": str(PAPERS_FOLDER / thermal_name), "page_number": 9,
            },
        }]
        resolution = resolve_visual_target(
            case["question"], load_or_build_visual_index(),
            conversation_messages=previous,
        )
        assert resolution.status == "resolved", resolution.to_dict()
        assert resolution.target_type == "equation"
        assert resolution.pdf_name == thermal_name
        assert resolution.page_number == case["pdf_page"]
        with patch("services.visual_runtime.analyse_pdf_page") as figure_vision:
            answer = analyse_resolved_equation(
                case["question"], resolution,
                conversation_history=previous, debug_info=debug,
            )
        figure_vision.assert_not_called()
        for fact in case["required_facts"]:
            assert _contains(answer, fact), f"answer missing required fact: {fact}"
        for fact in case["prohibited_facts"]:
            assert not _contains(answer, fact), f"prohibited fact present: {fact}"
        assert debug["final_answer_code_path"] == case["expected_structured_fields"]["final_answer_path"]
        symbolic = debug.get("symbolic_equation")
        assert isinstance(symbolic, dict), "symbolic Equation 8 validation did not run"
        assert symbolic["sign_validation"] == "passed"
        assert symbolic["paper_symbols"] == {
            "specific_heat": "c", "blood_temperature": "T_b",
        }
        assert [(term["kind"], term["sign"]) for term in symbolic["target_terms"]] == [
            ("second_time_derivative", "+"),
            ("conduction", "+"),
            ("tissue_temperature_perfusion", "-"),
            ("first_time_derivative", "-"),
            ("blood_temperature_perfusion", "+"),
            ("external_source", "+"),
            ("external_source_time_derivative", "+"),
        ]
        assert [(term["kind"], term["sign"]) for term in symbolic["rearranged_terms"]] == [
            ("first_time_derivative", "+"),
            ("conduction", "+"),
            ("blood_minus_tissue_perfusion", "+"),
            ("external_source", "+"),
        ]
        assert "c_p" not in answer and "T_a" not in answer
        assert "steady-state" not in answer.casefold()
        artifact_writer(case["case_id"], raw=answer, structured={
            "answer": answer, "resolution": resolution.to_dict(), "debug": debug,
        })
    except BaseException as error:
        debug.update({"answer": answer})
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
        answer = analyse_resolved_visual(
            case["question"], resolution, debug_info=debug
        )
        grouped = case["case_id"] == "figure_6_auto_hallmarks"
        value = debug.get("normalized_json" if grouped else "validated_json")
        assert isinstance(value, dict), "validated structured output was not produced"
        expected_fields = case["expected_structured_fields"]
        allowed_paths = expected_fields.get("allowed_final_answer_paths")
        if allowed_paths:
            assert debug.get("final_answer_path") in allowed_paths
        else:
            expected_path = expected_fields.get(
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
            _assert_figure_13(case, value, answer, debug)
        elif case["case_id"] == "table_1":
            _assert_table_1(case, value)
        elif case["case_id"] == "thermal_figure_2_boundary_conditions":
            _assert_thermal_boundary_figure(case, value, answer)
        elif case["case_id"] in {
            "thermal_table_2", "thermal_table_3_text_fallback",
        }:
            _assert_exact_text_table(case, value, answer)
            assert value is debug["repaired_json"]
            assert value is debug["final_structured_output"]
            assert value is debug["rendered_structured_object"]
            evidence = build_visual_evidence(resolution, answer, debug)
            assert evidence["structured_result"] is value
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


def _assert_thermal_boundary_figure(case, value, answer):
    rendered = json.dumps(value, ensure_ascii=False) + "\n" + answer
    for fact in case["required_facts"]:
        assert _contains(rendered, fact), f"missing required fact: {fact}"
    for fact in case["prohibited_facts"]:
        assert not _contains(rendered, fact), f"prohibited fact present: {fact}"
    expected = case["expected_structured_fields"]
    assert value["diagram_kind"] == expected["diagram_kind"]
    assert value["circuit_topology"] is expected["circuit_topology"]
    assert "TM microwave" in answer
    assert "could not verify" not in answer.casefold()
    assert "ASSOCIATED EQUATIONS/NEARBY TEXT" not in answer


def _assert_exact_text_table(case, value, answer):
    expected = case["expected_structured_fields"]
    assert [str(column) for column in value["columns"]] == expected["columns"]
    assert [
        [None if cell is None else str(cell) for cell in row]
        for row in value["rows"]
    ] == expected["rows"]
    assert value["unreadable_cells"] == expected["unreadable_cells"]
    rendered = json.dumps(value, ensure_ascii=False) + "\n" + answer
    for fact in case["required_facts"]:
        assert _contains(rendered, fact), f"missing required fact: {fact}"


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
    edge_components = {
        re.sub(r"[^a-z0-9]+", "", _normal(edge["component"]))
        for edge in topology["edges"]
    }
    assert edge_components == {"re", "ri", "c"}
    series_branch = next(
        branch for branch in branches if len(branch["components"]) == 2
    )
    edges = {
        re.sub(r"[^a-z0-9]+", "", _normal(edge["component"])): edge
        for edge in topology["edges"]
    }
    first, second = [
        edges[re.sub(r"[^a-z0-9]+", "", _normal(component))]
        for component in series_branch["components"]
    ]
    assert (
        first["to_node"] == second["from_node"]
        or first["from_node"] == second["to_node"]
    )


def _assert_figure_9(case, value):
    _assert_required_and_prohibited(case, value)
    panels = value["panels"]
    assert len(panels) == 6
    assert all(_contains(panel["x_axis"]["label"], "frequency") for panel in panels)
    assert all(_normal(panel["x_axis"]["scale"]) == "log" for panel in panels)
    kinds = [_normal(panel["graph_kind"]) for panel in panels]
    assert sum("magnitude" in kind for kind in kinds) == 3
    assert sum("phase" in kind for kind in kinds) == 3
    assert all(
        _normal(panel["y_axis"]["scale"]) == "linear"
        for panel in panels
    )
    magnitude_labels = [panel["y_axis"]["label"] for panel in panels if "magnitude" in _normal(panel["graph_kind"])]
    assert all(_contains(label, "Zfat") for label in magnitude_labels)
    comparisons = value["comparisons"]
    assert comparisons["magnitude_order_high_to_low"] == ["Group 3", "Group 1", "Group 2"]
    assert comparisons["greatest_phase_complexity_group"] == "Group 2"


def _assert_figure_13(case, value, answer, debug):
    rendered = json.dumps(value, ensure_ascii=False)
    for fact in ("Sample 10", "Sample 7"):
        assert _contains(rendered, fact)
    assert len(value["panels"]) == 2
    by_panel = {
        re.sub(r"[^a-z0-9]+", "", _normal(panel["panel"])): panel
        for panel in value["panels"]
    }
    assert _normal(by_panel["a"]["group"]) == "sample 10"
    assert _normal(by_panel["b"]["group"]) == "sample 7"
    assert all(_normal(panel["x_axis"]["scale"]) == "linear" for panel in value["panels"])
    assert all(_normal(panel["y_axis"]["scale"]) == "linear" for panel in value["panels"])
    assert all("img(z)" not in _normal(panel["y_axis"]["label"]) for panel in value["panels"])
    assert all("im(z)" in _normal(panel["y_axis"]["label"]) for panel in value["panels"])
    assert "img(z)" not in _normal(json.dumps(value, ensure_ascii=False))
    for panel in value["panels"]:
        fit_text = json.dumps(
            [panel.get("shape_features", []), panel.get("visible_trends", [])],
            ensure_ascii=False,
        )
        assert any(
            _contains(fit_text, phrase)
            for phrase in ("good", "closely follow", "nearly indistinguishable")
        ), f"fit quality was not described as good for {panel.get('group')}"
        fit_key = _normal(fit_text)
        assert not re.search(
            r"\b(?:excellent|perfect|superimposed|indistinguishable)\b"
            r"[^.]{0,50}\b(?:entire|whole) range\b",
            fit_key,
        )

        multiplier = str(panel["y_axis"].get("scientific_multiplier") or "")
        exponent = re.search(r"10\s*(?:\^|\*\*)?\s*([-+]?\d+)", multiplier)
        factor = 10 ** int(exponent.group(1)) if exponent else 1
        ticks = [float(str(tick).replace(",", "")) for tick in panel["y_axis"]["tick_labels"]]
        expected_max = max(ticks) * factor
        actual_max = float(panel["visible_range"]["max"])
        assert abs(actual_max - expected_max) <= max(1e-9, expected_max * 0.08)

    sample_seven_evidence = json.dumps(
        [
            by_panel["b"],
            *[
                item for item in value["comparisons"]
                if _contains(item.get("subject", ""), "Sample 7")
                or _contains(item.get("claim", ""), "Sample 7")
            ],
        ],
        ensure_ascii=False,
    )
    assert _contains(sample_seven_evidence, "right") or _contains(
        sample_seven_evidence, "high Re(Z)"
    )
    sample_ten_trends = " ".join(map(str, by_panel["a"]["visible_trends"]))
    sample_seven_trends = " ".join(map(str, by_panel["b"]["visible_trends"]))
    assert "tight panel crop" not in sample_ten_trends.lower()
    assert "tight panel crop" in sample_seven_trends.lower()
    assert _contains(sample_seven_trends, "Sample 7")
    assert _contains(sample_seven_trends, "high Re(Z)")
    for item in value["comparisons"]:
        if re.search(r"\b(?:closer|better)\b", _normal(item.get("claim", ""))):
            assert item.get("uncertain") or _contains(
                item.get("subject", ""),
                "Sample 10",
            )

    def physical_x_max(panel):
        multiplier = str(panel["x_axis"].get("scientific_multiplier") or "")
        exponent = re.search(r"10\s*(?:\^|\*\*)?\s*([-+]?\d+)", multiplier)
        factor = 10 ** int(exponent.group(1)) if exponent else 1
        return max(float(str(tick).replace(",", "")) for tick in panel["x_axis"]["tick_labels"]) * factor

    assert physical_x_max(by_panel["a"]) > physical_x_max(by_panel["b"])
    comparison_claims = [_normal(item["claim"]) for item in value["comparisons"]]
    assert len(comparison_claims) == len(set(comparison_claims))
    scale_comparison = next(
        item for item in value["comparisons"]
        if _contains(item.get("metric", ""), "overall impedance scale")
    )
    assert _contains(scale_comparison["subject"], "Sample 10")
    assert "10^5" in scale_comparison["claim"]
    assert "10^4" in scale_comparison["claim"]
    fit_uncertainties = []
    for item in value["comparisons"]:
        if item.get("uncertain") and (
            _contains(item.get("metric", ""), "fit")
            or _contains(item.get("metric", ""), "model-data deviation")
        ):
            fit_uncertainties.append(item)
    assert len(fit_uncertainties) == 1
    assert _contains(fit_uncertainties[0]["claim"], "too close to distinguish")
    assert not any(
        _contains(note, "fit closeness") for note in value["uncertain_values"]
    )
    assert _normal(answer).count("too close to distinguish") == 1
    initial = debug.get("initial_parsed_json")
    initial_complexities = [
        panel.get("complexity_score") for panel in initial.get("panels", [])
    ] if isinstance(initial, dict) else []
    complexity_errors = [
        error for error in debug.get("initial_validation_errors", [])
        if "complexity" in error.casefold()
    ]
    if complexity_errors:
        assert any(
            not isinstance(score, (int, float))
            or isinstance(score, bool)
            or not 0 <= score <= 1
            for score in initial_complexities
        )
    elif initial_complexities:
        assert all(
            isinstance(score, (int, float))
            and not isinstance(score, bool)
            and 0 <= score <= 1
            for score in initial_complexities
        )
    assert debug.get("validation_error") == ""
    assert value["frequency_direction_evidence"] == []
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


def test_automatic_resolution_hallmarks_semantics_override_previous_pdf():
    index = load_or_build_visual_index()
    bio = "Bioimpedance spectroscopy for characterizing volume-dependent structural.pdf"
    messages = [{
        "role": "assistant",
        "content": "Validated Figure 9 answer",
        "visual_target": {
            "status": "resolved", "target_type": "figure", "target_number": "9",
            "pdf_name": bio, "page_number": 8,
        },
    }]
    result = resolve_visual_target(
        "How does Figure 6 distinguish primary, antagonistic and integrative hallmarks?",
        index,
        selected_pdf=bio,
        conversation_messages=messages,
    )
    assert result.status == "resolved", result.to_dict()
    assert result.pdf_name == "hall marks of aging.pdf"
    assert result.page_number == 46


def test_automatic_resolution_figure_one_uses_genuine_previous_pdf_context():
    index = load_or_build_visual_index()
    first = resolve_visual_target("Compare Groups 1, 2 and 3 in Figure 9.", index)
    messages = [{
        "role": "assistant", "content": "Figure 9 answer",
        "visual_target": first.to_dict(),
    }]
    result = resolve_visual_target(
        "Explain Figure 1.", index, conversation_messages=messages
    )
    assert result.status == "resolved", result.to_dict()
    assert result.pdf_name == first.pdf_name
    assert result.page_number == 2
