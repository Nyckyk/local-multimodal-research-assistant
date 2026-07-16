from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import patch

from services import visual_index
from services.visual_index import extract_page_record, load_or_build_visual_index
from services.visual_fallback import resolve_with_visual_fallback
from services.visual_locator import (
    VisualResolution,
    clear_visual_target_state,
    manual_visual_resolution,
    resolve_visual_target,
    should_activate_automatic_vision,
)
from services.visual_reference_parser import parse_visual_reference
from settings import PAPERS_FOLDER


def _index(*records):
    files = {}
    for pdf_name, page_number, target_type, target_number, caption, match_kind in records:
        file_record = files.setdefault(pdf_name, {"pages": []})
        page = next(
            (item for item in file_record["pages"] if item["pdf_page"] == page_number),
            None,
        )
        if page is None:
            page = {
                "pdf_page": page_number,
                "printed_page_number": None,
                "captions": [],
                "references": [],
                "nearby_text": caption,
            }
            file_record["pages"].append(page)
        row = {
            "target_type": target_type,
            "target_number": target_number,
            "confidence": 0.96 if match_kind == "caption" else 0.48,
        }
        if match_kind == "caption":
            row["caption"] = caption
            page["captions"].append(row)
        else:
            page["references"].append(row)
    return {"version": 2, "files": files}


def _previous(target_type="figure", target_number="9", panel=None, pdf_name="one.pdf", page=8):
    return [{
        "role": "assistant",
        "content": "Prior visual answer",
        "visual_target": {
            "status": "resolved", "target_type": target_type,
            "target_number": target_number, "panel": panel,
            "pdf_name": pdf_name, "pdf_path": str(PAPERS_FOLDER / pdf_name),
            "page_number": page,
        },
    }]


def test_parse_figure_9():
    value = parse_visual_reference("Explain Figure 9.")
    assert (value.target_type, value.target_number, value.panel) == ("figure", "9", None)
    assert value.explicit_reference


def test_parse_fig_13b_as_panel():
    value = parse_visual_reference("What does Fig. 13b show?")
    assert (value.target_number, value.panel) == ("13", "b")


def test_parse_supplement_and_decimal_identifiers():
    assert parse_visual_reference("Explain Figure A.1").target_number == "A.1"
    assert parse_visual_reference("Extract Table B.3").target_number == "B.3"
    assert parse_visual_reference("Read supplementary Fig. S2").target_number == "S2"


def test_parse_panel_references():
    assert parse_visual_reference("Figure 9(a)").panel == "a"
    value = parse_visual_reference("panel c of Figure 9")
    assert (value.target_number, value.panel) == ("9", "c")


def test_unrelated_numbers_are_not_visual_references():
    value = parse_visual_reference("Compare Samples 10 and 7 at 100 Hz")
    assert not value.explicit_reference
    assert value.target_type == "unknown"


def test_remaining_query_preserves_scientific_terms():
    value = parse_visual_reference("Compare BRCA1 and mTOR in Figure 2")
    assert "BRCA1" in value.remaining_query and "mTOR" in value.remaining_query


def test_page_index_constructs_caption_and_printed_page():
    class Rect:
        height = 800

    class Page:
        rect = Rect()

        def get_text(self, kind):
            if kind == "text":
                return "Fig. A.1. Supplement plot\nNearby scientific text\n12"
            return [
                (40, 100, 500, 140, "Fig. A.1. Supplement plot"),
                (40, 760, 60, 780, "12"),
            ]

    record = extract_page_record(Page(), 13)
    assert record["pdf_page"] == 13
    assert record["printed_page_number"] == "12"
    assert record["captions"][0]["target_number"] == "A.1"


def test_cache_invalidates_after_pdf_modification(tmp_path):
    papers = tmp_path / "papers"
    papers.mkdir()
    cache = tmp_path / "cache" / "index.json"
    pdf = papers / "paper.pdf"
    pdf.write_bytes(b"first local PDF bytes")

    def fake_index(path, signature):
        identifier = "2" if path.stat().st_size > 22 else "1"
        return {
            "pdf_filename": path.name,
            "signature": signature,
            "indexed_at": "now",
            "page_count": 1,
            "pages": [{
                "pdf_page": 1,
                "captions": [{
                    "target_type": "figure", "target_number": identifier,
                    "caption": f"Fig. {identifier}.", "confidence": 0.96,
                }],
                "references": [], "nearby_text": "",
            }],
            "error": "",
        }

    with patch.object(visual_index, "_index_pdf", side_effect=fake_index) as index_pdf:
        first = load_or_build_visual_index(papers, cache)
        first_hash = first["files"][pdf.name]["signature"]["sha256"]
        time.sleep(0.01)
        pdf.write_bytes(b"second local PDF bytes that changed")
        second = load_or_build_visual_index(papers, cache)
        third = load_or_build_visual_index(papers, cache)
    assert second["files"][pdf.name]["signature"]["sha256"] != first_hash
    assert second["files"][pdf.name]["pages"][0]["captions"][0]["target_number"] == "2"
    assert third["files"][pdf.name]["signature"] == second["files"][pdf.name]["signature"]
    assert index_pdf.call_count == 2


def test_malformed_pdf_is_recorded_without_stopping_other_files(tmp_path):
    papers = tmp_path / "papers"
    papers.mkdir()
    (papers / "broken.pdf").write_bytes(b"not a PDF")
    value = load_or_build_visual_index(papers, tmp_path / "index.json")
    assert value["files"]["broken.pdf"]["pages"] == []
    assert value["files"]["broken.pdf"]["error"]


def test_exact_identifier_matching_does_not_match_figure_19():
    index = _index(
        ("paper.pdf", 2, "figure", "9", "Fig. 9. Target", "caption"),
        ("paper.pdf", 3, "figure", "19", "Fig. 19. Other", "caption"),
    )
    result = resolve_visual_target("Explain Figure 9", index)
    assert result.status == "resolved" and result.page_number == 2


def test_selected_pdf_is_preference_not_unconditional_truth():
    index = _index(
        ("one.pdf", 1, "figure", "1", "Fig. 1. Alpha", "caption"),
        ("two.pdf", 2, "figure", "1", "Fig. 1. Beta", "caption"),
    )
    preferred = resolve_visual_target("Explain Figure 1", index, selected_pdf="two.pdf")
    assert preferred.pdf_name == "two.pdf"
    named = resolve_visual_target(
        "Explain Figure 1 in one.pdf", index, selected_pdf="two.pdf"
    )
    assert named.pdf_name == "one.pdf"


def test_caption_relevance_disambiguates_duplicate_number():
    index = _index(
        ("aging.pdf", 37, "figure", "1", "Figure 1. Hallmarks of aging", "caption"),
        ("tissue.pdf", 2, "figure", "1", "Figure 1. Adipocyte tissue fluid schematic", "caption"),
    )
    result = resolve_visual_target(
        "Explain the adipocyte tissue fluid schematic in Figure 1", index
    )
    assert result.status == "resolved" and result.pdf_name == "tissue.pdf"


def test_hallmarks_figure_six_semantics_override_previous_bioimpedance_pdf():
    bio = "Bioimpedance spectroscopy.pdf"
    hallmarks = "hall marks of aging.pdf"
    index = _index(
        (bio, 5, "figure", "6", "Fig. 6. Genetic algorithm diagnostic output", "caption"),
        (
            hallmarks, 46, "figure", "6",
            "Figure 6. Primary, antagonistic and integrative hallmarks of aging",
            "caption",
        ),
    )
    result = resolve_visual_target(
        "How does Figure 6 distinguish primary, antagonistic and integrative hallmarks?",
        index,
        conversation_messages=_previous(pdf_name=bio, page=8),
    )
    assert result.status == "resolved"
    assert (result.pdf_name, result.page_number) == (hallmarks, 46)


def test_hallmarks_figure_six_semantics_override_active_sidebar_and_previous_pdf():
    bio = "Bioimpedance spectroscopy.pdf"
    hallmarks = "hall marks of aging.pdf"
    index = _index(
        (bio, 5, "figure", "6", "Fig. 6. Genetic algorithm diagnostics", "caption"),
        (
            hallmarks, 46, "figure", "6",
            "Figure 6. Primary, antagonistic and integrative hallmarks of aging",
            "caption",
        ),
    )
    result = resolve_visual_target(
        "How does Figure 6 distinguish primary, antagonistic and integrative hallmarks?",
        index,
        selected_pdf=bio,
        conversation_messages=_previous(pdf_name=bio, page=8),
    )
    assert (result.status, result.pdf_name, result.page_number) == (
        "resolved", hallmarks, 46
    )


def test_visual_type_comes_from_target_and_caption():
    index = _index(
        ("paper.pdf", 7, "table", "1", "Table 1. Measurements", "caption"),
        ("paper.pdf", 8, "figure", "9", "Fig. 9. Bode graph spectra", "caption"),
        ("paper.pdf", 3, "figure", "3", "Fig. 3. Equivalent circuit", "caption"),
    )
    assert resolve_visual_target("Extract Table 1", index).visual_type == "table"
    assert resolve_visual_target("Explain Figure 9", index).visual_type == "graph"
    assert resolve_visual_target("Explain Figure 3", index).visual_type == "labelled_diagram"


def test_duplicate_figure_one_without_context_is_ambiguous():
    index = _index(
        ("aging.pdf", 37, "figure", "1", "Figure 1. Hallmarks", "caption"),
        ("tissue.pdf", 2, "figure", "1", "Figure 1. Tissue", "caption"),
    )
    result = resolve_visual_target("Explain Figure 1", index)
    assert result.status == "ambiguous"
    assert {row["pdf_name"] for row in result.candidates[:2]} == {"aging.pdf", "tissue.pdf"}


def test_generic_identifier_score_gap_below_configured_margin_is_ambiguous():
    index = _index(
        ("one.pdf", 2, "figure", "1", "Figure 1. First topic", "caption"),
        ("two.pdf", 37, "figure", "1", "Figure 1. Second topic", "caption"),
    )
    with patch(
        "services.visual_locator._embedding_relevance",
        return_value=[0.4, 0.2],
    ):
        result = resolve_visual_target("Explain Figure 1.", index)
    gap = result.candidates[0]["confidence"] - result.candidates[1]["confidence"]
    assert abs(gap - 0.06) < 1e-9
    assert result.status == "ambiguous"
    assert {row["pdf_name"] for row in result.candidates} == {"one.pdf", "two.pdf"}


def test_real_figure_one_explicit_bioimpedance_name_resolves_page_two():
    result = resolve_visual_target(
        "Explain Figure 1 in the bioimpedance paper.",
        load_or_build_visual_index(),
    )
    assert result.status == "resolved", result.to_dict()
    assert result.pdf_name.startswith("Bioimpedance spectroscopy")
    assert result.page_number == 2


def test_real_figure_one_hallmarks_terminology_resolves_page_thirty_seven():
    result = resolve_visual_target(
        "Explain the nine hallmarks in Figure 1.",
        load_or_build_visual_index(),
    )
    assert result.status == "resolved", result.to_dict()
    assert result.pdf_name == "hall marks of aging.pdf"
    assert result.page_number == 37


def test_duplicate_figure_one_uses_prior_pdf_only_with_visual_context():
    index = _index(
        ("aging.pdf", 37, "figure", "1", "Figure 1. Hallmarks", "caption"),
        ("tissue.pdf", 2, "figure", "1", "Figure 1. Tissue", "caption"),
    )
    previous = resolve_visual_target(
        "Explain Figure 1", index,
        conversation_messages=_previous(pdf_name="tissue.pdf", page=8),
    )
    assert previous.status == "resolved" and previous.pdf_name == "tissue.pdf"

    unrelated_history = [{"role": "assistant", "content": "A text-only answer"}]
    cleared = resolve_visual_target(
        "Explain Figure 1", index, conversation_messages=unrelated_history
    )
    assert cleared.status == "ambiguous"


def test_real_hallmarks_figure_six_overrides_previous_bioimpedance_context():
    bio = "Bioimpedance spectroscopy for characterizing volume-dependent structural.pdf"
    result = resolve_visual_target(
        "How does Figure 6 distinguish primary, antagonistic and integrative hallmarks?",
        load_or_build_visual_index(),
        selected_pdf=bio,
        conversation_messages=_previous(pdf_name=bio, page=8),
    )
    assert result.status == "resolved", result.to_dict()
    assert result.pdf_name == "hall marks of aging.pdf"
    assert result.page_number == 46


def test_missing_and_low_confidence_targets_are_not_silently_selected():
    assert resolve_visual_target("Explain Figure 4", _index()).status == "not_found"
    low = _index(("paper.pdf", 3, "figure", "4", "", "reference"))
    result = resolve_visual_target("Explain Figure 4", low)
    assert result.status == "ambiguous"
    assert "confidence" in result.reason


def test_local_vision_fallback_runs_only_after_text_resolution_fails(tmp_path):
    index = {
        "files": {"scan.pdf": {"pages": [{
            "pdf_page": 4, "nearby_text": "", "captions": [], "references": [],
        }]}}
    }
    missing = resolve_visual_target("Explain Figure S2", index)
    with patch(
        "services.visual_fallback._inspect_page",
        return_value={
            "found": True, "target_type": "figure", "target_number": "S2",
            "caption": "Fig. S2. Supplement", "confidence": 0.94,
        },
    ) as inspect:
        resolved = resolve_with_visual_fallback(
            "Explain Figure S2", missing, index,
            papers_folder=tmp_path,
        )
    assert inspect.call_count == 1
    assert resolved.status == "resolved" and resolved.page_number == 4

    already_resolved = VisualResolution(
        status="resolved", target_type="figure", target_number="S2",
        pdf_name="scan.pdf", pdf_path="scan.pdf", page_number=4,
    )
    with patch("services.visual_fallback._inspect_page") as inspect:
        unchanged = resolve_with_visual_fallback(
            "Explain Figure S2", already_resolved, index, papers_folder=tmp_path
        )
    assert unchanged is already_resolved
    inspect.assert_not_called()


def test_panel_followup_reuses_previous_figure():
    index = _index(("one.pdf", 8, "figure", "9", "Fig. 9. Groups", "caption"))
    result = resolve_visual_target(
        "What about panel b?", index, conversation_messages=_previous()
    )
    assert (result.pdf_name, result.page_number, result.target_number, result.panel) == (
        "one.pdf", 8, "9", "b"
    )


def test_there_followup_reuses_previous_visual_target():
    index = _index(("one.pdf", 8, "figure", "9", "Fig. 9. Groups", "caption"))
    result = resolve_visual_target(
        "Which group is highest there?", index,
        conversation_messages=_previous(),
    )
    assert result.status == "resolved"
    assert (result.target_number, result.page_number) == ("9", 8)
    assert result.panel is None


def test_that_table_uses_last_table_only():
    index = _index(("one.pdf", 7, "table", "1", "Table 1. Values", "caption"))
    result = resolve_visual_target(
        "Explain that table", index,
        conversation_messages=_previous("table", "1", pdf_name="one.pdf", page=7),
    )
    assert result.status == "resolved" and result.target_type == "table"


def test_that_table_does_not_skip_over_a_newer_figure():
    index = _index(("one.pdf", 7, "table", "1", "Table 1. Values", "caption"))
    messages = [
        *_previous("table", "1", pdf_name="one.pdf", page=7),
        *_previous("figure", "9", pdf_name="one.pdf", page=8),
    ]
    result = resolve_visual_target(
        "Explain that table", index, conversation_messages=messages
    )
    assert result.status == "not_found"


def test_clear_conversation_resets_visual_target_state():
    state = {
        "last_visual_target": {"target_number": "9"},
        "pending_visual_resolution": {},
        "pending_visual_question": "Question",
        "messages": [1],
    }
    clear_visual_target_state(state)
    assert "last_visual_target" not in state
    assert "pending_visual_resolution" not in state
    assert state["messages"] == [1]


def test_manual_override_wins_and_text_only_does_not_activate_vision(tmp_path):
    manual = manual_visual_resolution(tmp_path / "paper.pdf", 12, "Explain Figure 1")
    assert manual.reason == "Manual PDF/page override" and manual.page_number == 12
    text_resolution = resolve_visual_target("Summarise this paper", _index())
    assert not should_activate_automatic_vision("Summarise this paper", text_resolution)


def test_legacy_files_list_cache_is_supported(tmp_path):
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps({
        "files": [{"pdf_filename": "legacy.pdf", "signature": {}, "pages": []}]
    }), encoding="utf-8")
    loaded = visual_index._load_cache(path)
    assert "legacy.pdf" in loaded["files"]


def test_real_pdf_resolution_mappings(visual_resolution_cases):
    index = load_or_build_visual_index()
    for case in visual_resolution_cases:
        result = resolve_visual_target(case["question"], index)
        assert result.status == "resolved", (case["case_id"], result.to_dict())
        assert result.pdf_name == case["expected_pdf"], case["case_id"]
        assert result.page_number == case["expected_page"], case["case_id"]
        assert result.target_type == case["target_type"]
        assert result.target_number.casefold() == str(case["target_number"]).casefold()


def test_existing_real_vision_questions_resolve_without_manual_page(regression_cases):
    index = load_or_build_visual_index()
    for case in regression_cases:
        if case["expected_response_type"] == "rag_summary":
            continue
        result = resolve_visual_target(case["question"], index)
        assert result.status == "resolved", (case["case_id"], result.to_dict())
        assert result.pdf_name == case["pdf_filename"]
        assert result.page_number == case["pdf_page"]
