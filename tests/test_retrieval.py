import unittest

from rag.retrieval import (
    _source_identity,
    extract_framework_grounding,
    infer_document_type,
    retrieve_context,
)


class FakeCollection:
    def __init__(self):
        self.query_calls = 0

    def count(self):
        return 1

    def query(self, **kwargs):
        self.query_calls += 1
        return {
            "documents": [["Grounded text-only evidence."]],
            "metadatas": [[{"source": "paper.txt", "page": 2, "chunk": 0}]],
            "distances": [[0.1]],
        }


class FakeEmbedder:
    def encode(self, *args, **kwargs):
        return FakeVector()


class FakeVector:
    def tolist(self):
        return [0.1, 0.2]


class FakeReranker:
    def predict(self, pairs):
        return [0.9]


class SummaryCollection:
    def __init__(self):
        self.query_calls = 0
        self.rows = [
            (
                "Abstract\nThis review enumerates a framework of nine hallmarks and "
                "explains their relationships. These hallmarks are: genomic instability, "
                "telomere attrition, epigenetic alterations, loss of proteostasis, "
                "deregulated nutrient-sensing, mitochondrial dysfunction, cellular "
                "senescence, stem cell exhaustion, and altered intercellular communication. ",
                {"source": "review.txt", "pdf": "review.pdf", "page": 1, "chunk": 0},
            ),
            (
                "Introduction\nThe article explains the scope and background of aging research.",
                {"source": "review.txt", "pdf": "review.pdf", "page": 2, "chunk": 0},
            ),
            (
                "Central Framework\nThe review organizes damage, responses and tissue-level "
                "outcomes into a unified conceptual structure.",
                {"source": "review.txt", "pdf": "review.pdf", "page": 4, "chunk": 0},
            ),
            (
                "Proteostasis\nA detailed mechanism involving one narrow pathway.",
                {"source": "review.txt", "pdf": "review.pdf", "page": 7, "chunk": 0},
            ),
            (
                "Interconnections\nThe major themes interact rather than operating as isolated pathways.",
                {"source": "review.txt", "pdf": "review.pdf", "page": 14, "chunk": 0},
            ),
            (
                "Figure 6. The overall framework organizes the hallmarks into three groups.",
                {"source": "review.txt", "pdf": "review.pdf", "page": 12, "chunk": 0},
            ),
            (
                "Conclusions and Perspectives\nPrimary hallmarks are genomic instability, "
                "telomere attrition, epigenetic alterations, and loss of proteostasis. "
                "Antagonistic hallmarks are deregulated nutrient-sensing, mitochondrial "
                "dysfunction, and cellular senescence. Integrative hallmarks are stem cell "
                "exhaustion and altered intercellular communication. Inflammation is a "
                "mechanism discussed within intercellular communication, not an additional "
                "named hallmark. The framework has implications for future research but "
                "causal relationships remain uncertain.",
                {"source": "review.txt", "pdf": "review.pdf", "page": 20, "chunk": 0},
            ),
            (
                "References\nSmith et al. 2010. Jones et al. 2011. Brown et al. 2012. "
                "White et al. 2013. Green et al. 2014. PubMed DOI.",
                {"source": "review.txt", "pdf": "review.pdf", "page": 30, "chunk": 0},
            ),
        ]

    def count(self):
        return len(self.rows)

    def query(self, **kwargs):
        self.query_calls += 1
        return {
            "documents": [[row[0] for row in self.rows]],
            "metadatas": [[row[1] for row in self.rows]],
            "distances": [[0.1] * len(self.rows)],
        }


class SummaryReranker:
    def predict(self, pairs):
        return [
            -7.5 if "narrow pathway" in document else 0.5
            for _, document in pairs
        ]


class NegativeSummaryReranker:
    def predict(self, pairs):
        return [-10.0 - (index * 0.1) for index, _ in enumerate(pairs)]


class FallbackCollection(SummaryCollection):
    def query(self, **kwargs):
        self.query_calls += 1
        if self.query_calls <= 4:
            return {"documents": [[]], "metadatas": [[]], "distances": [[]]}
        rows = self.rows[:5]
        return {
            "documents": [[row[0] for row in rows]],
            "metadatas": [[row[1] for row in rows]],
            "distances": [[0.1] * len(rows)],
        }


class ExpandableSummaryCollection(SummaryCollection):
    def query(self, **kwargs):
        self.query_calls += 1
        rows = self.rows[1:6]
        return {
            "documents": [[row[0] for row in rows]],
            "metadatas": [[row[1] for row in rows]],
            "distances": [[0.1] * len(rows)],
        }

    def get(self, **kwargs):
        return {
            "documents": [row[0] for row in self.rows],
            "metadatas": [row[1] for row in self.rows],
        }


class TextRetrievalTests(unittest.TestCase):
    def test_text_only_rag_retrieval_is_unchanged(self):
        collection = FakeCollection()
        context, sources = retrieve_context(
            question="What does the paper say?",
            previous_question="",
            collection=collection,
            embedder=FakeEmbedder(),
            reranker=FakeReranker(),
        )
        self.assertIn("Grounded text-only evidence.", context)
        self.assertEqual(sources[0]["source"], "paper.txt")
        self.assertEqual(sources[0]["page"], 2)
        self.assertNotIn("DOCUMENT SUMMARY MODE", context)
        self.assertEqual(collection.query_calls, 1)

    def test_summary_excludes_bibliography_and_prioritizes_overview_sections(self):
        collection = SummaryCollection()
        context, sources = retrieve_context(
            question="Summarise this paper.",
            previous_question="",
            collection=collection,
            embedder=FakeEmbedder(),
            reranker=SummaryReranker(),
        )
        pages = [source["page"] for source in sources]
        self.assertIn(1, pages)
        self.assertIn(20, pages)
        self.assertNotIn(30, pages)
        self.assertNotIn(7, pages)
        self.assertIn("Document type: review", context)
        self.assertEqual(collection.query_calls, 4)

    def test_summary_spans_multiple_major_sections(self):
        _, sources = retrieve_context(
            question="What are the main conclusions?",
            previous_question="",
            collection=SummaryCollection(),
            embedder=FakeEmbedder(),
            reranker=SummaryReranker(),
        )
        sections = {source["section"] for source in sources}
        self.assertGreaterEqual(len(sections), 4)
        self.assertTrue({"abstract", "introduction", "conclusion"}.issubset(sections))

    def test_review_is_classified_as_review(self):
        self.assertEqual(
            infer_document_type(["This review synthesizes a conceptual framework."]),
            "review",
        )

    def test_legacy_chunks_without_section_metadata_are_supported(self):
        _, sources = retrieve_context(
            "Give me an overview of this paper.", "",
            SummaryCollection(), FakeEmbedder(), SummaryReranker(),
        )
        self.assertGreaterEqual(len(sources), 5)
        self.assertTrue(all("section" in source for source in sources))

    def test_all_negative_reranker_scores_still_return_results(self):
        _, sources = retrieve_context(
            "Summarise this paper.", "",
            SummaryCollection(), FakeEmbedder(), NegativeSummaryReranker(),
        )
        self.assertGreaterEqual(len(sources), 3)
        self.assertTrue(all(source["score"] < 0 for source in sources))

    def test_pdf_and_text_source_names_share_document_identity(self):
        self.assertEqual(
            _source_identity({"source": "Hallmarks of Aging.txt"}),
            _source_identity({"pdf": "Hallmarks of Aging.pdf"}),
        )

    def test_summary_fallback_runs_after_specialized_search_returns_zero(self):
        collection = FallbackCollection()
        context, sources = retrieve_context(
            "Summarise this paper.", "",
            collection, FakeEmbedder(), SummaryReranker(),
        )
        self.assertTrue(context)
        self.assertGreaterEqual(len(sources), 3)
        self.assertGreater(collection.query_calls, 4)

    def test_summary_expands_selected_document_to_recover_abstract_and_conclusion(self):
        context, sources = retrieve_context(
            "Summarise this paper.", "", ExpandableSummaryCollection(),
            FakeEmbedder(), SummaryReranker(),
        )
        pages = {source["page"] for source in sources}
        self.assertIn(1, pages)
        self.assertIn(20, pages)
        self.assertIn("Canonical named framework item count: 9", context)

    def test_hallmarks_summary_has_five_chunks_across_three_pages(self):
        _, sources = retrieve_context(
            "Summarise the main findings of this paper.", "",
            SummaryCollection(), FakeEmbedder(), SummaryReranker(),
        )
        self.assertGreaterEqual(len(sources), 5)
        self.assertGreaterEqual(len({source["page"] for source in sources}), 3)

    def test_review_framework_categories_are_grounded_in_named_items(self):
        documents = [row[0] for row in SummaryCollection().rows]
        grounding = extract_framework_grounding(documents)
        self.assertIn("loss of proteostasis", grounding["categories"]["primary"])
        self.assertNotIn("inflammation", grounding["named_items"])
        self.assertNotIn(
            "inflammation",
            [item for values in grounding["categories"].values() for item in values],
        )
        context, _ = retrieve_context(
            "Summarise this paper.", "",
            SummaryCollection(), FakeEmbedder(), SummaryReranker(),
        )
        self.assertIn("VALIDATED FRAMEWORK GROUNDING", context)
        self.assertIn('"loss of proteostasis"', context)
        self.assertIn("Canonical named framework item count: 9", context)

    def test_framework_extraction_is_stable_when_candidates_are_page_sorted(self):
        rows = list(reversed(SummaryCollection().rows))
        ordered = [
            document for document, metadata in sorted(
                rows, key=lambda row: (row[1]["page"], row[1]["chunk"])
            )
        ]
        grounding = extract_framework_grounding(ordered)
        self.assertEqual(len(grounding["named_items"]), 9)
        self.assertEqual(len(grounding["categories"]["primary"]), 4)
        self.assertEqual(len(grounding["categories"]["antagonistic"]), 3)
        self.assertEqual(len(grounding["categories"]["integrative"]), 2)


if __name__ == "__main__":
    unittest.main()
