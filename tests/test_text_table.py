from __future__ import annotations

import fitz

from services.text_table import compare_tables, extract_text_table


def _insert_row(page, y, cells):
    for x, value in cells:
        page.insert_text((x, y), value, fontsize=9)


def test_layout_text_table_preserves_blank_cells_and_scientific_notation():
    document = fitz.open()
    page = document.new_page()
    page.insert_text((50, 60), "Table 4", fontsize=10)
    page.insert_text((50, 72), "Synthetic convergence data.", fontsize=9)
    _insert_row(page, 90, [(50, "Mesh"), (180, "Coarse"), (280, "Fine"), (380, "Finer")])
    _insert_row(page, 105, [(50, "Resource"), (180, "10"), (280, "20"), (380, "40")])
    _insert_row(page, 120, [(50, "Metric"), (180, "1.0E-3"), (380, "1.1E-3")])

    table = extract_text_table(page, "4", "Extract Table 4.")
    assert table["columns"] == ["Mesh", "Coarse", "Fine", "Finer"]
    assert table["rows"][1] == ["Metric", "1.0E-3", None, "1.1E-3"]
    assert table["unreadable_cells"] == [{"row": 1, "column": "Fine"}]


def test_layout_text_table_explains_selected_refinement_from_adjacent_columns():
    document = fitz.open()
    page = document.new_page()
    page.insert_text((50, 60), "Table 5", fontsize=10)
    _insert_row(page, 80, [(50, "Mesh"), (180, "Coarse"), (280, "Fine"), (380, "Finer")])
    _insert_row(page, 95, [(50, "Resource"), (180, "10"), (280, "20"), (380, "40")])
    _insert_row(page, 110, [(50, "Metric (u)"), (180, "1.0"), (280, "1.1000"), (380, "1.1001")])

    table = extract_text_table(
        page, "5", "Explain why the authors selected the Fine mesh."
    )
    explanation = table["comparisons"][0]
    assert "changed Metric by only 0.0001 u" in explanation
    assert "Resource increased from 20 to 40" in explanation


def test_table_cross_check_rejects_conflicting_readable_values():
    text = {
        "columns": ["Name", "Value"], "rows": [["A", "1"]],
    }
    matching = {
        "columns": ["Name", "Value"], "rows": [["A", 1]],
    }
    conflicting = {
        "columns": ["Name", "Value"], "rows": [["A", "2"]],
    }
    assert compare_tables(matching, text)["matched"]
    assert not compare_tables(conflicting, text)["matched"]
