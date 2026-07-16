import importlib
import json
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


class Rect:
    def __init__(self, x0, y0, x1, y1):
        self.x0, self.y0, self.x1, self.y1 = x0, y0, x1, y1

    @property
    def height(self):
        return self.y1 - self.y0

    @property
    def width(self):
        return self.x1 - self.x0


sys.modules.setdefault(
    "fitz", types.SimpleNamespace(Page=object, Rect=Rect, open=lambda path: None)
)
sys.modules.setdefault("ollama", types.SimpleNamespace(chat=lambda **kwargs: None))
vision = importlib.import_module("services.vision_service")


def overview():
    return {
        "headings": [
            {"name": "primary", "label": "Primary hallmarks", "y_center": 0.2, "confidence": 0.95},
            {"name": "antagonistic", "label": "Antagonistic hallmarks", "y_center": 0.5, "confidence": 0.95},
            {"name": "integrative", "label": "Integrative hallmarks", "y_center": 0.8, "confidence": 0.95},
        ]
    }


def regional(region, items):
    return {"region": region, "items": items}


def item(label, y, fully=True, confidence=0.95):
    return {
        "label": label,
        "y_center": y,
        "fully_visible": fully,
        "confidence": confidence,
    }


class VisionPipelineTests(unittest.TestCase):
    def setUp(self):
        self.figure = Rect(0, 0, 100, 100)
        self.clips = [Rect(0, 0, 100, 56), Rect(0, 22, 100, 82), Rect(0, 44, 100, 100)]

    def test_regional_crop_without_heading_is_accepted(self):
        raw = json.dumps(regional("middle", [item("Visible label", 0.5)]))
        result = vision.parse_regional_response(raw, "middle")
        self.assertEqual(result["items"][0]["label"], "Visible label")

    def test_overview_heading_detection_schema(self):
        result = vision.parse_overview_response(json.dumps(overview()))
        self.assertEqual(
            [heading["name"] for heading in result["headings"]],
            ["primary", "antagonistic", "integrative"],
        )

    def test_crop_relative_coordinate_converts_to_page_coordinate(self):
        self.assertEqual(vision.relative_y_to_page(0.5, Rect(0, 20, 100, 60)), 40)

    def test_figure_three_crop_excludes_figure_two_labels_and_right_column(self):
        class Page:
            rect = Rect(0, 0, 1000, 1000)

            def get_text(self, kind):
                if kind == "blocks":
                    return [
                        (40, 300, 480, 390, "Fig. 2. Dispersion regions alpha beta delta gamma"),
                        (40, 700, 480, 770, "Fig. 3. Equivalent circuit representation"),
                        (520, 400, 960, 450, "Fig. 4. Resistance surface"),
                    ]
                return ""

            def get_image_info(self):
                return [{"bbox": (80, 500, 450, 690)}]

        clip = vision._detect_figure_clip(Page(), "Explain Figure 3 topology")
        self.assertGreater(clip.y0, 390)
        self.assertLessEqual(clip.y1, 700)
        self.assertLessEqual(clip.x1, 510)
        neighbouring_labels = {
            "α": (100, 120), "β": (200, 160), "δ": (300, 180), "γ": (400, 200),
        }
        included = [
            label for label, (x, y) in neighbouring_labels.items()
            if clip.x0 <= x <= clip.x1 and clip.y0 <= y <= clip.y1
        ]
        self.assertEqual(included, [])

    def test_inline_figure_reference_is_not_treated_as_caption(self):
        class Page:
            def get_text(self, kind):
                return [
                    (10, 10, 200, 30, "The parameters in Fig. 3 are discussed here."),
                    (10, 200, 200, 230, "Fig. 3. Equivalent circuit representation"),
                ] if kind == "blocks" else ""

        target = vision._target_caption(Page(), "Explain Figure 3")
        self.assertEqual(target[1], "Fig. 3. Equivalent circuit representation")

    def test_six_panel_graph_is_split_into_three_panel_pairs(self):
        text = " ".join(
            f"({label}) Group {(index // 2) + 1} panel"
            for index, label in enumerate("abcdef")
        )
        labels = vision._multi_panel_labels(text)
        clips = vision._graph_panel_pair_clips(Rect(0, 0, 600, 900), len(labels))
        self.assertEqual(labels, list("abcdef"))
        self.assertEqual(len(clips), 3)
        self.assertEqual([(clip.y0, clip.y1) for clip in clips], [(0, 300), (300, 600), (600, 900)])

    def test_small_typed_crop_uses_higher_render_scale(self):
        page = types.SimpleNamespace(rect=Rect(0, 0, 600, 800))
        self.assertEqual(
            vision._typed_render_scale(page, Rect(20, 400, 280, 540)),
            3.0,
        )

    def test_wide_shallow_graph_crop_uses_higher_render_scale(self):
        page = types.SimpleNamespace(rect=Rect(0, 0, 600, 800))
        self.assertEqual(
            vision._typed_render_scale(page, Rect(20, 40, 580, 250)),
            3.0,
        )
        self.assertEqual(
            vision._typed_render_scale(page, Rect(20, 40, 580, 500)),
            1.5,
        )

    def test_two_panel_nyquist_does_not_use_six_panel_path(self):
        self.assertEqual(
            vision._multi_panel_labels("(a) Sample 10 (b) Sample 7"),
            [],
        )
        self.assertEqual(
            vision._two_panel_graph_labels("(a) Sample 10 (b) Sample 7"),
            ["a", "b"],
        )
        clips = vision._side_by_side_panel_clips(Rect(20, 40, 580, 250))
        self.assertEqual(
            [(clip.x0, clip.x1) for clip in clips],
            [(20, 300.0), (300.0, 580)],
        )

    def test_plain_named_sample_comparison_uses_tight_panel_verification(self):
        self.assertTrue(vision._asks_for_two_panel_comparison(
            "Compare Samples 10 and 7 in Figure 13."
        ))
        self.assertTrue(vision._asks_for_two_panel_comparison(
            "Which fit is closer for Samples 10 and 7?"
        ))
        self.assertFalse(vision._asks_for_two_panel_comparison(
            "Explain Sample 10 in Figure 13."
        ))

    def test_overlapping_regions_are_evidence_not_ownership(self):
        regions = [
            regional("top", [item("Alpha", 0.35)]),
            regional("middle", [item("Boundary label", 0.86, False, 0.7)]),
            regional("bottom", [item("Boundary label", 0.52, True, 0.98)]),
        ]
        result, assignments, detections, _ = vision.reconcile_spatial_results(
            overview(), regions, self.clips, self.figure
        )
        self.assertIn("Alpha", result["primary"])
        self.assertIn("Boundary label", result["integrative"])
        self.assertNotIn("Boundary label", result["antagonistic"])
        boundary = next(row for row in assignments if row["label"] == "Boundary label")
        self.assertEqual(boundary["evidence"], ["vision"])
        self.assertEqual(len([d for d in detections if d["label"] == "Boundary label"]), 2)

    def test_text_cross_check_requires_explicit_item_membership(self):
        memberships = vision._explicit_text_memberships(
            "Marker alpha is an antagonistic hallmark. Marker beta is discussed nearby.",
            ["Marker alpha", "Marker beta"],
        )
        self.assertEqual(memberships["marker alpha"]["group"], "antagonistic")
        self.assertNotIn("marker beta", memberships)

    def test_text_cross_check_accepts_explicit_group_descriptor(self):
        memberships = vision._explicit_text_memberships(
            "Cellular senescence is a beneficial compensatory response to damage. ",
            ["Cellular senescence"],
        )
        self.assertEqual(memberships["cellular senescence"]["group"], "antagonistic")

    def test_non_visible_partial_label_is_dropped_when_full_expansion_exists(self):
        observations = {
            "mitochondrial": {
                "label": "Mitochondrial",
                "groups": {"antagonistic": [{"fully_visible": False}]},
            },
            "mitochondrial dysfunction": {
                "label": "Mitochondrial dysfunction",
                "groups": {"antagonistic": [{"fully_visible": True}]},
            },
        }
        vision._discard_partial_label_fragments(observations)
        self.assertNotIn("mitochondrial", observations)
        self.assertIn("mitochondrial dysfunction", observations)

    def test_author_and_page_metadata_are_removed(self):
        cleaned = vision.clean_detected_label("López-Otín et al., Page 46, instability")
        self.assertEqual(cleaned, "instability")

    def test_noisy_ocr_is_recovered_from_document_candidates(self):
        evidence = (
            "Source: paper.txt, page 20, chunk 1\n"
            "These hallmarks are: genomic instability, telomere attrition."
        )
        candidates = vision.extract_candidate_labels(evidence)
        normalized = vision.normalize_visual_label(
            "López-Otín et al., Page 46, instability", candidates
        )
        self.assertEqual(normalized, "Genomic instability")

    def test_ambiguous_partial_candidate_remains_uncertain(self):
        evidence = (
            "Source: paper.txt, page 20, chunk 1\n"
            "These hallmarks are: Genomic instability and Chromosomal instability."
        )
        regions = [
            regional("top", [item("instability", 0.3)]),
            regional("middle", []),
            regional("bottom", []),
        ]
        result, _, _, _ = vision.reconcile_spatial_results(
            overview(), regions, self.clips, self.figure, evidence, 20
        )
        self.assertEqual(result["uncertain"], ["instability"])

    def test_explicit_text_resolves_uncertain_item(self):
        regions = [
            regional("top", []),
            regional("middle", [item("Stem cell exhaustion", 0.63, True, 0.8)]),
            regional("bottom", [item("Stem cell exhaustion", 0.464, True, 0.8)]),
        ]
        evidence = (
            "Source: paper.txt, page 20, chunk 1\n"
            "A third category comprises the integrative hallmarks, "
            "Stem cell exhaustion and Altered intercellular communication."
        )
        result, assignments, _, _ = vision.reconcile_spatial_results(
            overview(), regions, self.clips, self.figure, evidence, 20
        )
        self.assertIn("Stem cell exhaustion", result["integrative"])
        self.assertEqual(result["uncertain"], [])
        assignment = next(row for row in assignments if row["label"] == "Stem cell exhaustion")
        self.assertIn("text", assignment["evidence"])
        self.assertEqual(assignment["source_page"], 20)

    def test_final_json_contains_no_metadata(self):
        with self.assertRaises(vision.StructuredVisionError):
            vision.validate_reconciled_schema({
                "primary": ["Page 46"],
                "antagonistic": [],
                "integrative": [],
                "uncertain": [],
            })

    def test_duplicate_item_is_rejected(self):
        with self.assertRaisesRegex(vision.StructuredVisionError, "appears in both"):
            vision.validate_reconciled_schema({
                "primary": ["Repeated"],
                "antagonistic": ["Repeated"],
                "integrative": [],
                "uncertain": [],
            })

    def test_most_items_in_every_category_is_rejected(self):
        labels = [f"Label {number}" for number in range(9)]
        with self.assertRaisesRegex(vision.StructuredVisionError, "repeated in every"):
            vision.validate_reconciled_schema({
                "primary": labels.copy(),
                "antagonistic": labels.copy(),
                "integrative": labels.copy(),
                "uncertain": [],
            })

    def test_figure_six_regression(self):
        regions = [
            regional("top", [
                item("López-Otín et al., Page 46, instability", 0.18),
                item("Telomere attrition", 0.3),
                item("Epigenetic alterations", 0.42), item("Loss of proteostasis", 0.55),
            ]),
            regional("middle", [
                item("Deregulated nutrient-sensing", 0.48),
                item("Mitochondrial dysfunction", 0.58),
                item("Cellular senescence", 0.68),
                item("Stem cell exhaustion", 0.88, False, 0.65),
            ]),
            regional("bottom", [
                item("Stem cell exhaustion", 0.52),
                item("Altered intercellular communication", 0.68),
            ]),
        ]
        evidence = (
            "Source: paper.txt, page 20, chunk 1\n"
            "The primary hallmarks are Genomic instability, Telomere attrition, "
            "Epigenetic alterations and Loss of proteostasis. "
            "The antagonistic hallmarks are Deregulated nutrient-sensing, "
            "Mitochondrial dysfunction and Cellular senescence. "
            "A third category comprises the integrative hallmarks, "
            "Stem cell exhaustion and Altered intercellular communication."
        )
        result, _, _, _ = vision.reconcile_spatial_results(
            overview(), regions, self.clips, self.figure, evidence, 20
        )
        self.assertEqual(result, {
            "primary": [
                "Genomic instability", "Telomere attrition",
                "Epigenetic alterations", "Loss of proteostasis",
            ],
            "antagonistic": [
                "Deregulated nutrient-sensing", "Mitochondrial dysfunction",
                "Cellular senescence",
            ],
            "integrative": [
                "Stem cell exhaustion", "Altered intercellular communication",
            ],
            "uncertain": [],
        })

    def test_grouping_failure_never_calls_free_form(self):
        fake_path = Path("paper.pdf")
        document = unittest.mock.MagicMock()
        document.__len__.return_value = 1
        fake_page = unittest.mock.MagicMock()
        fake_page.get_text.return_value = ""
        document.load_page.return_value = fake_page
        with patch.object(Path, "exists", return_value=True), patch.object(
            vision.fitz, "open", return_value=document
        ), patch.object(
            vision, "_analyse_grouped_figure",
            side_effect=vision.StructuredVisionError("invalid structured result"),
        ), patch.object(vision, "_ask_vision_model") as free_form:
            with self.assertRaises(vision.StructuredVisionError):
                vision.analyse_pdf_page(fake_path, 1, "Which group contains X?")
        free_form.assert_not_called()
        self.assertIn("Could not verify", vision.COULD_NOT_VERIFY_MESSAGE)


if __name__ == "__main__":
    unittest.main()
