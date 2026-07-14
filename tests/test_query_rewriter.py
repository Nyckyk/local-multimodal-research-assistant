import importlib
import sys
import types
import unittest
from unittest.mock import patch


sys.modules.setdefault("ollama", types.SimpleNamespace(chat=lambda **kwargs: None))
rewriter = importlib.import_module("services.query_rewriter")


HISTORY = [
    {"role": "user", "content": "What does the paper say about Figure 6?"},
    {"role": "assistant", "content": "It describes three groups of hallmarks."},
]


class QueryRewriterTests(unittest.TestCase):
    def test_figure_and_hallmarks_are_retained(self):
        question = (
            "How does Figure 6 distinguish primary, antagonistic and "
            "integrative hallmarks?"
        )
        self.assertEqual(rewriter.rewrite_question(question, HISTORY), question)

    def test_scientific_terms_are_not_replaced_with_synonyms(self):
        question = "How do BRCA1, p53 and mTOR affect DNA repair in Figure 2?"
        self.assertEqual(rewriter.rewrite_question(question, HISTORY), question)

    def test_follow_up_pronoun_can_be_resolved(self):
        question = "How does it affect aging?"
        response = {
            "message": {"content": "How does mTOR affect aging?"}
        }
        with patch.object(rewriter.ollama, "chat", return_value=response):
            rewritten = rewriter.rewrite_question(
                question,
                [{"role": "user", "content": "What does mTOR regulate?"}],
            )
        self.assertEqual(rewritten, "How does mTOR affect aging?")

    def test_rewrite_is_rejected_if_original_term_is_removed(self):
        question = "How does it affect hallmarks?"
        response = {
            "message": {"content": "How does it affect biomolecular interaction features?"}
        }
        with patch.object(rewriter.ollama, "chat", return_value=response):
            rewritten = rewriter.rewrite_question(question, HISTORY)
        self.assertEqual(rewritten, question)

    def test_panel_followup_and_supplement_identifier_are_preserved(self):
        for question in ("What about panel b?", "Compare mTOR in Figure A.1"):
            with self.subTest(question=question):
                self.assertEqual(
                    rewriter.rewrite_question(question, HISTORY), question
                )


if __name__ == "__main__":
    unittest.main()
