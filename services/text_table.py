"""Layout-aware extraction of a contiguous PDF text table."""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation

from services.structured_vision import validate_table
from services.visual_reference_parser import canonical_identifier


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _page_lines(page) -> list[dict]:
    lines = []
    for block in page.get_text("dict").get("blocks", []):
        for line in block.get("lines", []):
            text = _clean("".join(span.get("text", "") for span in line.get("spans", [])))
            if not text:
                continue
            x0, y0, x1, y1 = line["bbox"]
            lines.append({
                "text": text, "x0": float(x0), "x1": float(x1),
                "y0": float(y0), "y1": float(y1),
            })
    return sorted(lines, key=lambda row: (row["y0"], row["x0"]))


def _row_groups(lines: list[dict], tolerance: float = 1.6) -> list[list[dict]]:
    groups: list[list[dict]] = []
    for line in lines:
        if groups and abs(groups[-1][0]["y0"] - line["y0"]) <= tolerance:
            groups[-1].append(line)
        else:
            groups.append([line])
    return [sorted(group, key=lambda row: row["x0"]) for group in groups]


def _decimal(value) -> Decimal | None:
    text = _clean(value).replace(",", "")
    if not re.fullmatch(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?", text):
        return None
    try:
        return Decimal(text)
    except InvalidOperation:
        return None


def _numeric_tail(group: list[dict]) -> int:
    return sum(_decimal(cell["text"]) is not None for cell in group[1:])


def _normal_header(text: str) -> str:
    # A grouped header such as "f = 0.9 GHz" identifies the displayed value;
    # the leaf header supplies the measured variable.
    return re.sub(r"^[A-Za-z]\s*=\s*", "", _clean(text))


def _column_headers(
    header_groups: list[list[dict]], leaf_positions: list[float]
) -> list[str]:
    columns = []
    for position in leaf_positions:
        parts = []
        for group in header_groups:
            preceding = [cell for cell in group if cell["x0"] <= position + 2.0]
            if not preceding:
                continue
            text = _normal_header(preceding[-1]["text"])
            if text and (not parts or text.casefold() != parts[-1].casefold()):
                parts.append(text)
        columns.append(" ".join(parts).strip())
    for index, column in enumerate(columns):
        if not column:
            columns[index] = f"Column {index + 1}"
    # The typed table schema requires unique names. Preserve the visible text
    # and add only a positional suffix when a genuinely repeated leaf remains.
    seen: dict[str, int] = {}
    for index, column in enumerate(columns):
        key = column.casefold()
        seen[key] = seen.get(key, 0) + 1
        if seen[key] > 1:
            columns[index] = f"{column} ({seen[key]})"
    return columns


def _assign_row(group: list[dict], leaf_positions: list[float]) -> list[str | None]:
    row: list[str | None] = [None] * len(leaf_positions)
    for cell in group:
        index = min(
            range(len(leaf_positions)),
            key=lambda position: abs(leaf_positions[position] - cell["x0"]),
        )
        if row[index] is None:
            row[index] = cell["text"]
        else:
            row[index] = f"{row[index]} {cell['text']}".strip()
    return row


def _display_decimal(value: Decimal) -> str:
    rendered = format(value, "f")
    return rendered.rstrip("0").rstrip(".") if "." in rendered else rendered


def _selection_comparison(question: str, columns: list[str], rows: list[list]) -> list[str]:
    if not re.search(r"\b(?:why|reason)\b.*\b(?:select|choose|use|adopt)", question, re.I):
        return []
    question_key = _clean(question).casefold()
    selected = next(
        (
            index for index, column in sorted(
                enumerate(columns[1:], start=1),
                key=lambda item: len(item[1]), reverse=True,
            )
            if _clean(column).casefold() in question_key
        ),
        None,
    )
    if selected is None or selected + 1 >= len(columns):
        return []

    numeric_rows = []
    for row in rows:
        left, right = _decimal(row[selected]), _decimal(row[selected + 1])
        if left is None or right is None:
            continue
        scale = max(abs(left), abs(right), Decimal("1e-30"))
        numeric_rows.append((abs(right - left) / scale, abs(right - left), row))
    if not numeric_rows:
        return []
    _, metric_delta, metric_row = min(numeric_rows, key=lambda item: item[0])
    metric_name = _clean(metric_row[0])
    unit_match = re.search(r"\(([^)]+)\)", metric_name)
    unit = unit_match.group(1).replace("Kg", "kg") if unit_match else ""
    metric_label = re.sub(r"\s*\([^)]*\)\s*$", "", metric_name).strip()
    resources = [
        row for _, _, row in numeric_rows if row is not metric_row
        and _decimal(row[selected + 1]) > _decimal(row[selected])
    ]
    resource_text = ""
    if resources:
        details = [
            f"{_clean(row[0])} increased from {row[selected]} to {row[selected + 1]}"
            for row in resources[:2]
        ]
        resource_text = ", while " + " and ".join(details)
    return [
        f"{columns[selected]} was selected because refinement to "
        f"{columns[selected + 1]} changed {metric_label} by only "
        f"{_display_decimal(metric_delta)}{f' {unit}' if unit else ''}"
        f"{resource_text}."
    ]


def extract_text_table(page, table_number: str, question: str = "") -> dict | None:
    """Parse a visually contiguous text table into the typed table schema."""
    lines = _page_lines(page)
    marker_index = next((
        index for index, line in enumerate(lines)
        if (
            match := re.match(r"^Table\s+([^\s.:]+(?:\.\d+)?)\b", line["text"], re.I)
        ) and canonical_identifier(match.group(1)) == canonical_identifier(table_number)
    ), None)
    if marker_index is None:
        return None
    marker = lines[marker_index]
    following = [line for line in lines if line["y0"] > marker["y0"] + 1.0]
    groups = _row_groups(following)
    start = next((index for index, group in enumerate(groups) if len(group) >= 2), None)
    if start is None:
        return None

    table_groups = []
    previous_y = None
    for group in groups[start:]:
        y0 = group[0]["y0"]
        if previous_y is not None and y0 - previous_y > 19.0:
            break
        if len(group) < 2:
            break
        table_groups.append(group)
        previous_y = y0
    if len(table_groups) < 2:
        return None

    column_count = max(len(group) for group in table_groups)
    full_indices = [
        index for index, group in enumerate(table_groups)
        if len(group) == column_count
    ]
    if not full_indices:
        return None
    first_full = full_indices[0]
    first_is_header = _numeric_tail(table_groups[first_full]) == 0
    data_start = first_full + 1 if first_is_header else first_full
    if data_start >= len(table_groups):
        return None

    reference_group = table_groups[first_full]
    leaf_positions = [cell["x0"] for cell in reference_group]
    header_groups = table_groups[:data_start]
    columns = _column_headers(header_groups, leaf_positions)
    rows = [
        _assign_row(group, leaf_positions)
        for group in table_groups[data_start:]
    ]
    if not rows or all(all(cell is None for cell in row) for row in rows):
        return None

    caption_lines = [
        line["text"] for line in following
        if marker["y0"] + 1.0 <= line["y0"] < table_groups[0][0]["y0"] - 1.0
        and len(_row_groups([line])) == 1
    ]
    title_suffix = " ".join(caption_lines)
    title = f"Table {table_number}"
    if title_suffix:
        title = f"{title}. {title_suffix}"
    units = {}
    for column in columns:
        match = re.search(r"\(([^)]+)\)", column)
        if match:
            units[column] = match.group(1)
    value = {
        "table_number": str(table_number),
        "title": title,
        "columns": columns,
        "rows": rows,
        "units": units,
        "comparisons": _selection_comparison(question, columns, rows),
        "unreadable_cells": [
            {"row": row_index, "column": columns[column_index]}
            for row_index, row in enumerate(rows)
            for column_index, cell in enumerate(row)
            if cell is None
        ],
    }
    return validate_table(value)


def compare_tables(vision: dict, text: dict) -> dict:
    """Cross-check two typed tables without coercing unreadable values."""
    def normal(value) -> str:
        return re.sub(r"\s+", "", str(value or "")).casefold()

    mismatches = []
    if len(vision.get("columns", [])) != len(text.get("columns", [])):
        mismatches.append("column count")
    if len(vision.get("rows", [])) != len(text.get("rows", [])):
        mismatches.append("row count")
    for row_index, (vision_row, text_row) in enumerate(zip(
        vision.get("rows", []), text.get("rows", [])
    )):
        for column_index, (vision_cell, text_cell) in enumerate(zip(vision_row, text_row)):
            if vision_cell is None or text_cell is None:
                continue
            if normal(vision_cell) != normal(text_cell):
                mismatches.append(f"row {row_index}, column {column_index}")
    return {"matched": not mismatches, "mismatches": mismatches}
