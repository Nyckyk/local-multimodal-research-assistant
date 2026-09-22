from __future__ import annotations

import json
from dataclasses import dataclass

from evaluation.runner import execute_run, load_suite, render_report, write_artifacts
from evaluation.scoring import aggregate_runs, normalize_text, numeric_match, score_case


def response(answer="alpha beta", **overrides):
    value = {
        "answer": answer,
        "sources": [{"source": "paper.pdf", "page": 3, "document": "evidence"}],
        "resolution": {
            "status": "resolved", "target_type": "figure", "target_number": "2",
            "page_number": 3, "pdf_name": "paper.pdf",
        },
        "evidence": None,
        "debug": {"final_answer_code_path": "validated", "missing_answer_slots": []},
    }
    value.update(overrides)
    return value


def case(**expect):
    return {"id": "Q", "question": "Question?", "expect": expect}


def test_pass_scoring():
    result = score_case(case(
        document="paper.pdf",
        target={"type": "figure", "number": "2", "page": 3, "status": "resolved"},
        required_terms=["alpha", "beta"],
        required_concepts=[{"id": "relationship", "all_of": ["alpha", "beta"]}],
        debug={"allowed_code_paths": ["validated"], "max_missing_slots": 0},
    ), response())
    assert result["status"] == "PASS" and result["score"] == 1


def test_partial_scoring_for_missing_non_hard_fact():
    result = score_case(case(required_terms=["alpha", "beta", "gamma"]), response())
    assert result["status"] == "PARTIAL"
    assert result["missing"] == ["term:gamma"]


def test_fail_scoring_for_wrong_document():
    result = score_case(case(document="different.pdf", required_terms=["alpha"]), response())
    assert result["status"] == "FAIL"
    assert "document_identity" in result["hard_failures"]


def test_forbidden_contradiction_is_hard_failure():
    result = score_case(case(
        required_terms=["alpha"],
        forbidden_claims=[{"id": "bad_claim", "any_of": ["beta"], "hard": True}],
    ), response())
    assert result["status"] == "FAIL"
    assert result["contradictions"] == ["bad_claim"]


def test_scoped_forbidden_claim_does_not_span_other_domain_sections():
    result = score_case(case(forbidden_claims=[{
        "id": "patient_a549", "any_of": ["a549"],
        "scope_after": "human patient evidence", "hard": True,
    }]), response(answer=(
        "Cell culture evidence: A549 cells. "
        "Human patient evidence: NAFLD liver samples."
    )))
    assert result["status"] == "PASS"


def test_numeric_and_text_normalization():
    assert normalize_text("SA‑β‑Gal") == "sa-beta-gal"
    assert numeric_match("Fewer than expected: less than 30% irradiated cells.", {
        "value": 30, "operator": "lt", "context_any": ["irradiated"],
    })
    assert numeric_match("Pearson r = 0.3862.", {
        "value": 0.3862, "tolerance": 0.00001, "context_any": ["pearson"]
    })
    assert not numeric_match("Pearson r = 0.3682.", {
        "value": 0.3862, "tolerance": 0.00001, "context_any": ["pearson"]
    })


def test_scientific_hyphen_variants_match_without_weakening_model_type():
    result = score_case(case(required_concepts=[{
        "id": "classifier_types",
        "all_of": ["aem.*classification tree", "aerfm.*random forest"],
    }]), response(answer=(
        "AEM is the classification-tree-based model and AERFM is the "
        "random-forest-based model."
    )))
    assert result["status"] == "PASS"


def test_unicode_hyphens_and_harmless_parentheses_normalize_for_matching():
    result = score_case(case(required_concepts=[{
        "id": "classifier_type",
        "all_of": ["classification tree based", "random forest based"],
    }]), response(answer="classification\u2011tree\u2011based; random (forest) based"))
    assert result["status"] == "PASS"


def test_required_mouse_term_accepts_scientific_plural_mice():
    result = score_case(case(required_terms=["mouse"]), response(
        answer="The experiment used young and old mice.",
    ))
    assert result["status"] == "PASS"


def test_human_cohort_term_accepts_explicit_patient_wording():
    result = score_case(case(required_terms=["human"]), response(
        answer="The NAFLD cohort contained 34 patients.",
    ))
    assert result["status"] == "PASS"


def test_number_word_variant_matches_required_numeric_term():
    result = score_case(case(required_terms=["3 plates"]), response(
        answer="Data were derived from at least three plates.",
    ))
    assert result["status"] == "PASS"


def test_debug_evidence_does_not_satisfy_displayed_answer_requirement():
    result = score_case(case(required_terms=["0.9969"]), response(
        answer="Panel G reports a strong correlation.",
        debug={"final_answer_evidence": {"value": "0.9969"}},
    ))
    assert "term:0.9969" in result["missing"]


def test_retrieved_chunks_are_not_scored_as_displayed_answer():
    result = score_case(case(required_terms=["grounded-value"]), response(
        answer="The answer omits the value.",
        sources=[{"source": "paper.pdf", "page": 3, "document": "grounded-value"}],
    ))
    assert "term:grounded-value" in result["missing"]


def test_displayed_value_satisfies_requirement():
    result = score_case(case(required_terms=["0.9969"]), response(
        answer="Panel G reports r = 0.9969.",
    ))
    assert result["status"] == "PASS"


def test_not_found_claim_is_hard_failure_when_selected_evidence_contains_value():
    result = score_case(case(required_terms=["0.7", "hepatocytes"]), response(
        answer="The paper does not mention a circularity threshold.",
        sources=[{"source": "paper.pdf", "page": 17,
                  "document": "A circularity threshold >0.7 selected predominantly hepatocytes."}],
    ))
    assert result["status"] == "FAIL"
    assert "unsupported_not_found" in result["hard_failures"]


def test_semantic_research_problem_equivalent_can_pass():
    result = score_case(case(required_concepts=[{
        "id": "problem",
        "any_of": [r"(?:challenge|difficulty).{0,120}(?:identifying|detecting).{0,80}senescen"],
    }]), response(answer=(
        "The study addresses the challenge of identifying senescent cells because "
        "traditional markers are limited."
    )))
    assert result["status"] == "PASS"


def test_numeric_context_crosses_short_heading_and_bullet_boundary():
    assert numeric_match("**Panel G**\n\n- Pearson correlation: r = 0.9969.", {
        "value": 0.9969, "tolerance": 0.00001, "context_any": ["panel g"],
    })


@dataclass
class FakeResponse:
    payload: dict

    def to_dict(self):
        return self.payload

    def assistant_message(self):
        return {"role": "assistant", "content": self.payload["answer"], "sources": self.payload["sources"]}


class RecordingBackend:
    def __init__(self):
        self.histories = []

    def ask(self, question, *, conversation_messages, **kwargs):
        self.histories.append([dict(row) for row in conversation_messages])
        return FakeResponse(response(answer=f"alpha: {question}"))


def test_fresh_conversation_reset_and_followup_retention():
    suite = {
        "name": "history", "document": "paper.pdf", "tests": [
            {"id": "Q1", "question": "first", "fresh_conversation": True, "expect": {"required_terms": ["alpha"]}},
            {"id": "Q2", "question": "followup", "fresh_conversation": False, "expect": {"required_terms": ["alpha"]}},
            {"id": "Q3", "question": "fresh", "fresh_conversation": True, "expect": {"required_terms": ["alpha"]}},
        ],
    }
    backend = RecordingBackend()
    run = execute_run(suite, backend)
    assert [row["status"] for row in run["results"]] == ["PASS", "PASS", "PASS"]
    assert backend.histories[0] == []
    assert [row["content"] for row in backend.histories[1]] == ["first", "alpha: first"]
    assert backend.histories[2] == []


def test_single_followup_case_executes_declared_setup_but_only_scores_target():
    suite = {
        "name": "history", "document": "paper.pdf", "tests": [
            {"id": "Q10", "question": "setup", "fresh_conversation": True, "expect": {}},
            {"id": "Q11", "question": "followup", "fresh_conversation": False, "setup_case_ids": ["Q10"], "expect": {"required_terms": ["alpha"]}},
        ],
    }
    backend = RecordingBackend()
    run = execute_run(suite, backend, question_id="Q11")
    assert [row["id"] for row in run["results"]] == ["Q11"]
    assert [row["content"] for row in backend.histories[1]] == ["setup", "alpha: setup"]


def test_repeated_run_aggregation():
    aggregate = aggregate_runs([
        {"results": [{"id": "Q1", "status": "PASS"}, {"id": "Q2", "status": "FAIL"}]},
        {"results": [{"id": "Q1", "status": "PASS"}, {"id": "Q2", "status": "PARTIAL"}]},
    ])
    assert aggregate["stable_passes"] == 1
    assert aggregate["flaky_questions"] == ["Q2"]
    assert aggregate["persistent_failures"] == []


def test_output_report_generation(tmp_path):
    suite = {"name": "example", "tests": []}
    runs = [{"run": 1, "duration_seconds": 0.1, "results": [{
        "id": "Q1", "question": "why", "answer": "answer", "status": "PARTIAL",
        "score": 0.8, "passed": ["document"], "missing": ["fact"],
        "contradictions": [], "hard_failures": [], "debug": {},
    }]}]
    report = render_report(suite, runs, {"status": "PASS"})
    assert "Q1 — PARTIAL" in report and "Missing:" in report and "- fact" in report
    output = tmp_path / "result"
    write_artifacts(output, suite, runs, {"status": "PASS"})
    assert {"answers.json", "answers.md", "scores.json", "report.md"}.issubset(
        path.name for path in output.iterdir()
    )
    assert json.loads((output / "scores.json").read_text())["aggregate"]["counts"]["PARTIAL"] == 1


def test_report_does_not_infer_fixture_issue_from_high_partial_score():
    suite = {"name": "example", "tests": []}
    runs = [{"run": 1, "duration_seconds": 0.1, "results": [{
        "id": "Q1", "question": "why", "answer": "answer", "status": "PARTIAL",
        "score": 0.99, "passed": ["document"], "missing": ["displayed_fact"],
        "contradictions": [], "hard_failures": [], "debug": {},
    }]}]
    assert "Possible fixture issue" not in render_report(suite, runs)


def test_senescence_fixture_retains_original_conversation_contract():
    suite = load_suite(__import__("pathlib").Path("tests/evals/senescence_v13.json"))
    assert len(suite["tests"]) == 15
    cases = {row["id"]: row for row in suite["tests"]}
    assert cases["Q11"]["fresh_conversation"] is False
    assert cases["Q11"]["setup_case_ids"] == ["Q10"]
    assert cases["Q15"]["fresh_conversation"] is True
    assert cases["Q15"]["select_document"] is False
    assert "vehicle" in cases["Q09"]["expect"]["required_terms"]
    assert "DMSO" not in cases["Q09"]["expect"]["required_terms"]
