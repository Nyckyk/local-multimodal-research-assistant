import json
import math
import re
from difflib import SequenceMatcher
from pathlib import Path
from tempfile import NamedTemporaryFile, mkdtemp

import fitz
import ollama

from settings import VISION_MODEL

from services.structured_vision import (
    analyse_compact_multi_panel_graph,
    analyse_typed_image,
    detect_compact_graph_panel_ids,
    detect_visual_type,
)


VALID_POSITIONS = {"top", "middle", "bottom", "left", "right", "unknown"}
COULD_NOT_VERIFY_MESSAGE = (
    "Could not verify a structured reading of this figure. "
    "No unvalidated classification is shown."
)
VISUAL_IDENTIFIER_PATTERN = r"(?:[A-Za-z]\.)?\d+(?:\.\d+)?|[A-Za-z]\d+"


class StructuredVisionError(ValueError):
    """Raised when a structured vision response cannot be trusted."""


def _render_page_image(
    page: fitz.Page,
    scale: float,
    clip: fitz.Rect | None = None,
    output_path: Path | None = None,
) -> Path:
    """Render a PDF page, or a clipped region of it, to a temporary PNG."""
    pixmap = page.get_pixmap(
        matrix=fitz.Matrix(scale, scale),
        clip=clip,
        alpha=False,
    )

    if output_path is None:
        with NamedTemporaryFile(suffix=".png", delete=False) as temporary_file:
            image_path = Path(temporary_file.name)
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image_path = output_path

    pixmap.save(str(image_path))
    return image_path


def _is_grouping_question(question: str) -> bool:
    text = question.lower()
    keywords = (
        "category", "categories", "group", "groups", "belong",
        "classified", "distinguish", "classification",
    )
    return any(keyword in text for keyword in keywords)


def _figure_number(question: str) -> str | None:
    match = re.search(
        rf"\bfig(?:ure)?\.?\s*({VISUAL_IDENTIFIER_PATTERN})",
        question,
        re.I,
    )
    return match.group(1).lower() if match else None


def _caption_blocks(page: fitz.Page) -> list[tuple[fitz.Rect, str]]:
    captions = []
    for block in page.get_text("blocks"):
        x0, y0, x1, y1, text = block[:5]
        # Inline references such as "the components in Fig. 3" are prose, not
        # captions. Only a line beginning with the identifier is a crop bound.
        if re.search(
            rf"(?im)^\s*fig(?:ure)?\.?\s*(?:{VISUAL_IDENTIFIER_PATTERN})\b",
            str(text),
        ):
            captions.append((fitz.Rect(x0, y0, x1, y1), str(text).strip()))
    return captions


def _target_caption(page: fitz.Page, question: str) -> tuple[fitz.Rect, str] | None:
    figure_number = _figure_number(question)
    if not figure_number:
        return None
    pattern = re.compile(
        rf"\bfig(?:ure)?\.?\s*{re.escape(figure_number)}\b",
        re.I,
    )
    matches = [entry for entry in _caption_blocks(page) if pattern.search(entry[1])]
    return min(matches, key=lambda entry: entry[0].y0) if matches else None


def _horizontal_band(page_rect: fitz.Rect, caption: fitz.Rect) -> tuple[float, float]:
    """Use caption placement to distinguish a column figure from a full-width one."""
    centre = (caption.x0 + caption.x1) / 2
    relative_centre = (centre - page_rect.x0) / page_rect.width
    margin = page_rect.width * 0.02
    if relative_centre < 0.42:
        return page_rect.x0 + margin, page_rect.x0 + page_rect.width * 0.51
    if relative_centre > 0.58:
        return page_rect.x0 + page_rect.width * 0.49, page_rect.x1 - margin
    return page_rect.x0 + margin, page_rect.x1 - margin


def _horizontal_overlap(rect: fitz.Rect, x0: float, x1: float) -> float:
    return max(0.0, min(rect.x1, x1) - max(rect.x0, x0))


def _embedded_figure_rects(
    page: fitz.Page,
    x0: float,
    x1: float,
    top: float,
    bottom: float,
) -> list[fitz.Rect]:
    get_image_info = getattr(page, "get_image_info", None)
    if not callable(get_image_info):
        return []
    candidates = []
    try:
        image_info = get_image_info()
    except Exception:
        return []
    page_area = max(1.0, page.rect.width * page.rect.height)
    for info in image_info:
        bbox = info.get("bbox") if isinstance(info, dict) else None
        if not bbox or len(bbox) != 4:
            continue
        rect = fitz.Rect(*bbox)
        overlap = _horizontal_overlap(rect, x0, x1)
        area = max(0.0, rect.width) * max(0.0, rect.height)
        if (
            overlap >= min(rect.width, x1 - x0) * 0.2
            and rect.y0 >= top - page.rect.height * 0.015
            and rect.y1 <= bottom + page.rect.height * 0.015
            and area >= page_area * 0.005
            and area <= page_area * 0.72
        ):
            candidates.append(rect)
    return candidates


def _detect_figure_clip(page: fitz.Page, question: str) -> fitz.Rect:
    """Isolate the requested figure using its caption, column and image bounds."""
    page_rect = page.rect
    target = _target_caption(page, question)
    if target is None:
        return page_rect

    caption, _ = target
    x0, x1 = _horizontal_band(page_rect, caption)
    previous = [
        rect
        for rect, _ in _caption_blocks(page)
        if rect.y1 <= caption.y0
        and _horizontal_overlap(rect, x0, x1) >= min(rect.width, x1 - x0) * 0.2
    ]
    if previous:
        top = max(rect.y1 for rect in previous) + page_rect.height * 0.008
    else:
        top = page_rect.y0 + page_rect.height * 0.035
    bottom = min(page_rect.y1, caption.y0)

    image_rects = _embedded_figure_rects(page, x0, x1, top, bottom)
    if image_rects:
        padding_x = page_rect.width * 0.01
        padding_y = page_rect.height * 0.008
        clip = fitz.Rect(
            max(x0, min(rect.x0 for rect in image_rects) - padding_x),
            max(top, min(rect.y0 for rect in image_rects) - padding_y),
            min(x1, max(rect.x1 for rect in image_rects) + padding_x),
            min(bottom, max(rect.y1 for rect in image_rects) + padding_y),
        )
    else:
        # Vector figures often have no embedded image object. Caption-to-caption
        # boundaries still isolate the target column without pulling in the
        # preceding or neighbouring figure.
        clip = fitz.Rect(x0, top, x1, bottom)
    return clip if clip.height >= page_rect.height * 0.08 else page_rect


def _detect_table_clip(page: fitz.Page, question: str) -> fitz.Rect:
    page_rect = page.rect
    match = re.search(
        rf"\btable\s*({VISUAL_IDENTIFIER_PATTERN})", question, re.IGNORECASE
    )
    pattern = re.compile(
        rf"^\s*table\s*{re.escape(match.group(1))}\b" if match else r"^\s*table\s*\d+",
        re.IGNORECASE,
    )
    for block in page.get_text("blocks"):
        x0, y0, x1, y1, text = block[:5]
        if pattern.search(str(text)):
            margin = page_rect.width * 0.02
            return fitz.Rect(
                page_rect.x0 + margin,
                max(page_rect.y0, y0 - page_rect.height * 0.015),
                min(page_rect.x1 - margin, max(x1 + margin, page_rect.width * 0.58)),
                min(page_rect.y1, max(y1 + margin, y0 + page_rect.height * 0.34)),
            )
    return page_rect


def _multi_panel_labels(page_text: str) -> list[str]:
    labels = re.findall(
        r"\(([a-z])\)\s*(?:group|sample|panel)\b",
        page_text,
        re.IGNORECASE,
    )
    ordered = []
    for label in labels:
        lowered = label.lower()
        if lowered not in ordered:
            ordered.append(lowered)
    return ordered if len(ordered) >= 4 and len(ordered) % 2 == 0 else []


def _two_panel_graph_labels(page_text: str) -> list[str]:
    """Identify a simple left-to-right pair without using the six-panel path."""
    labels = re.findall(
        r"\(([a-z])\)\s*(?:group|sample|panel)\b",
        page_text,
        re.IGNORECASE,
    )
    ordered = []
    for label in labels:
        lowered = label.lower()
        if lowered not in ordered:
            ordered.append(lowered)
    return ordered if len(ordered) == 2 else []


def _side_by_side_panel_clips(figure_clip: fitz.Rect) -> list[fitz.Rect]:
    midpoint = figure_clip.x0 + figure_clip.width / 2
    return [
        fitz.Rect(figure_clip.x0, figure_clip.y0, midpoint, figure_clip.y1),
        fitz.Rect(midpoint, figure_clip.y0, figure_clip.x1, figure_clip.y1),
    ]


def _graph_panel_pair_clips(figure_clip: fitz.Rect, panel_count: int) -> list[fitz.Rect]:
    row_count = panel_count // 2
    return [
        fitz.Rect(
            figure_clip.x0,
            figure_clip.y0 + figure_clip.height * row / row_count,
            figure_clip.x1,
            figure_clip.y0 + figure_clip.height * (row + 1) / row_count,
        )
        for row in range(row_count)
    ]


def _magnitude_y_axis_label_clip(pair_clip: fitz.Rect) -> fitz.Rect:
    """Return a tight strip containing only the left panel's rotated y label."""
    return fitz.Rect(
        pair_clip.x0,
        pair_clip.y0 + pair_clip.height * 0.08,
        pair_clip.x0 + pair_clip.width * 0.105,
        pair_clip.y0 + pair_clip.height * 0.78,
    )


def _typed_render_scale(page: fitz.Page, clip: fitz.Rect) -> float:
    """Give small diagrams enough pixels for labels and wire junctions."""
    if clip.height < page.rect.height * 0.30:
        return 3.0
    return 1.5


def _region_clips(figure_clip: fitz.Rect, tight: bool = False) -> list[fitz.Rect]:
    """Return overlapping top, middle and bottom bands of a figure."""
    if tight:
        bands = ((0.0, 0.48), (0.27, 0.76), (0.50, 1.0))
    else:
        # Scientific group boundaries often run through a labelled row rather
        # than through whitespace. The generous overlap keeps boundary labels
        # fully visible in both neighbouring observations.
        bands = ((0.0, 0.56), (0.22, 0.82), (0.44, 1.0))
    return [
        fitz.Rect(
            figure_clip.x0,
            figure_clip.y0 + figure_clip.height * start,
            figure_clip.x1,
            figure_clip.y0 + figure_clip.height * end,
        )
        for start, end in bands
    ]


def _build_prompt(question: str, correction: str = "") -> str:
    correction_block = ""
    if correction:
        correction_block = f"""
A previous draft failed validation:
{correction}
Re-read the image and correct that problem. Do not defend the prior answer.
"""

    return f"""
Inspect this research-paper page carefully.

Question:
{question}

Rules:
1. Use only information visibly present on this page.
2. Pay attention to figures, diagrams, tables, labels, headings, arrows and
   spatial layout.
3. Do not use outside knowledge.
4. Distinguish figure labels from caption text and surrounding body text.
5. If the visual evidence is insufficient or unreadable, say so explicitly.
6. Answer directly, clearly and concisely.
7. Mention colours, icons, arrows or other visible features only when they are
   clearly supported by the image.
{correction_block}
"""


def _overview_prompt(question: str, strict: bool = False) -> str:
    retry = "Re-read the heading labels and vertical centres carefully." if strict else ""
    return f"""
Analyse this full grouped scientific diagram overview.

Question: {question}

Detect only the category headings. Do not return item labels. Normalize a clearly
visible heading to primary, antagonistic, or integrative; otherwise use unknown.
y_center is the heading's vertical centre normalized within this overview.
Return exactly one JSON object and no prose:
{{
  "headings": [
    {{
      "name": "primary|antagonistic|integrative|unknown",
      "label": "...",
      "y_center": 0.0,
      "confidence": 0.0
    }}
  ]
}}
{retry}
"""


def _regional_prompt(question: str, position: str) -> str:
    return f"""
Inspect only this {position} crop of a grouped scientific diagram.

Question: {question}

Return every visible item label. Do not look for or require a category heading.
This crop overlaps adjacent crops, so include partial labels but mark them as not
fully visible. y_center is normalized within this crop.

Return exactly one JSON object and no prose:
{{
  "region": "{position}",
  "items": [
    {{
      "label": "...",
      "y_center": 0.0,
      "fully_visible": true,
      "confidence": 0.0
    }}
  ]
}}

Do not emit placeholder text. Return an empty items list if nothing is readable.
"""

def _response_text(response) -> str:
    answer = response["message"].get("content", "").strip()
    if not answer:
        answer = response["message"].get("thinking", "").strip()
    if not answer:
        raise RuntimeError("The vision model returned an empty response.")
    return re.sub(
        r"<think>.*?</think>", "", answer,
        flags=re.DOTALL | re.IGNORECASE,
    ).strip()


def _call_vision_model(
    image_path: Path,
    question: str,
    correction: str = "",
) -> str:
    response = ollama.chat(
        model=VISION_MODEL,
        messages=[{
            "role": "user",
            "content": _build_prompt(question, correction),
            "images": [str(image_path)],
        }],
        options={"temperature": 0, "num_ctx": 8192, "num_predict": 900},
    )
    return _response_text(response)


def _normalise_item(item: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", item.lower()).strip()


def parse_structured_response(raw_response: str, allow_conflicts: bool = False) -> dict:
    """Parse and validate strict JSON returned by the vision model."""
    try:
        result = json.loads(raw_response)
    except (json.JSONDecodeError, TypeError) as error:
        raise StructuredVisionError("The model did not return valid JSON.") from error

    if not isinstance(result, dict) or set(result) != {
        "groups", "uncertain_items", "confidence"
    }:
        raise StructuredVisionError("The JSON does not match the required schema.")
    if not isinstance(result["groups"], list) or not result["groups"]:
        raise StructuredVisionError("At least one group is required.")
    if not isinstance(result["uncertain_items"], list) or not all(
        isinstance(item, str) and item.strip() for item in result["uncertain_items"]
    ):
        raise StructuredVisionError("uncertain_items must be a list of strings.")
    confidence = result["confidence"]
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise StructuredVisionError("confidence must be numeric.")
    if not 0 <= confidence <= 1:
        raise StructuredVisionError("confidence must be between 0 and 1.")

    owners: dict[str, str] = {}
    for group in result["groups"]:
        if not isinstance(group, dict) or set(group) != {"name", "position", "items"}:
            raise StructuredVisionError("A group does not match the required schema.")
        if not isinstance(group["name"], str) or not group["name"].strip():
            raise StructuredVisionError("Every group must have a name.")
        if group["position"] not in VALID_POSITIONS:
            raise StructuredVisionError("A group has an invalid position.")
        if not isinstance(group["items"], list) or not all(
            isinstance(item, str) and item.strip() for item in group["items"]
        ):
            raise StructuredVisionError("Group items must be strings.")
        for item in group["items"]:
            key = _normalise_item(item)
            if not allow_conflicts and key in owners and owners[key] != group["name"]:
                raise StructuredVisionError(
                    f"Item {item!r} appears in mutually exclusive groups "
                    f"{owners[key]!r} and {group['name']!r}."
                )
            owners[key] = group["name"]

    uncertain = {_normalise_item(item) for item in result["uncertain_items"]}
    overlap = uncertain.intersection(owners)
    if overlap and not allow_conflicts:
        raise StructuredVisionError(
            "An item appears both in a group and in uncertain_items."
        )
    return result


def parse_overview_response(raw_response: str) -> dict:
    try:
        result = json.loads(raw_response)
    except (json.JSONDecodeError, TypeError) as error:
        raise StructuredVisionError("The overview response was not valid JSON.") from error
    if not isinstance(result, dict) or set(result) != {"headings"}:
        raise StructuredVisionError("The overview response has an invalid schema.")
    if not isinstance(result["headings"], list) or not result["headings"]:
        raise StructuredVisionError("The overview did not contain category headings.")
    seen = set()
    for heading in result["headings"]:
        if not isinstance(heading, dict) or set(heading) != {
            "name", "label", "y_center", "confidence"
        }:
            raise StructuredVisionError("An overview heading has an invalid schema.")
        if heading["name"] not in {"primary", "antagonistic", "integrative", "unknown"}:
            raise StructuredVisionError("An overview heading has an invalid canonical name.")
        if heading["name"] != "unknown" and heading["name"] in seen:
            raise StructuredVisionError("The overview contains duplicate category headings.")
        if not isinstance(heading["label"], str) or not heading["label"].strip():
            raise StructuredVisionError("An overview heading label is empty.")
        for field in ("y_center", "confidence"):
            value = heading[field]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
                raise StructuredVisionError(f"Overview {field} must be between 0 and 1.")
        seen.add(heading["name"])
    return result


def _call_overview_model(image_path: Path, question: str, strict=False) -> dict:
    response = ollama.chat(
        model=VISION_MODEL,
        messages=[{
            "role": "user",
            "content": _overview_prompt(question, strict),
            "images": [str(image_path)],
        }],
        format="json",
        options={"temperature": 0, "num_ctx": 8192, "num_predict": 600},
    )
    return parse_overview_response(_response_text(response))


def parse_regional_response(raw_response: str, expected_position: str) -> dict:
    try:
        result = json.loads(raw_response)
    except (json.JSONDecodeError, TypeError) as error:
        raise StructuredVisionError("A regional response was not valid JSON.") from error
    if not isinstance(result, dict) or set(result) != {"region", "items"}:
        raise StructuredVisionError("A regional response has an invalid schema.")
    if result["region"] != expected_position:
        raise StructuredVisionError("A regional response returned the wrong region.")
    if not isinstance(result["items"], list):
        raise StructuredVisionError("Regional items must be a list.")
    for item in result["items"]:
        if not isinstance(item, dict) or set(item) != {
            "label", "y_center", "fully_visible", "confidence"
        }:
            raise StructuredVisionError("A regional item has an invalid schema.")
        if not isinstance(item["label"], str) or not item["label"].strip():
            raise StructuredVisionError("A regional item must have a name.")
        if not isinstance(item["fully_visible"], bool):
            raise StructuredVisionError("Regional fully_visible must be boolean.")
        for field in ("y_center", "confidence"):
            value = item[field]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
                raise StructuredVisionError(f"Regional {field} must be between 0 and 1.")
    return result


def _call_regional_model(image_path: Path, question: str, position: str) -> dict:
    response = ollama.chat(
        model=VISION_MODEL,
        messages=[{
            "role": "user",
            "content": _regional_prompt(question, position),
            "images": [str(image_path)],
        }],
        format="json",
        options={"temperature": 0, "num_ctx": 8192, "num_predict": 900},
    )
    return parse_regional_response(_response_text(response), position)


def _call_verification_model(image_paths: list[Path], labels: list[str]) -> list[dict]:
    prompt = f"""
Inspect the overview and overlapping top, middle and bottom crops. Resolve only
these disputed labels: {json.dumps(labels)}

For each label answer: "Which heading is this label directly under?" Use the
label's connector/alignment and distance to the visible group headings. Do not
classify any other labels. Return exactly this JSON and no prose:
{{"items": [{{"item": "...", "heading": "...", "confidence": 0.0}}]}}
"""
    response = ollama.chat(
        model=VISION_MODEL,
        messages=[{
            "role": "user", "content": prompt,
            "images": [str(path) for path in image_paths],
        }],
        format="json",
        options={"temperature": 0, "num_ctx": 8192, "num_predict": 700},
    )
    try:
        result = json.loads(_response_text(response))
    except json.JSONDecodeError as error:
        raise StructuredVisionError("Verification response was malformed.") from error
    if not isinstance(result, dict) or set(result) != {"items"} or not isinstance(result["items"], list):
        raise StructuredVisionError("Verification response has an invalid schema.")
    requested = {_normalise_item(label) for label in labels}
    for item in result["items"]:
        if not isinstance(item, dict) or set(item) != {"item", "heading", "confidence"}:
            raise StructuredVisionError("A verification item has an invalid schema.")
        if _normalise_item(item["item"]) not in requested:
            raise StructuredVisionError("Verification introduced an unrequested item.")
        if canonical_group_name(item["heading"]) is None:
            raise StructuredVisionError("Verification returned an unknown heading.")
        if not isinstance(item["confidence"], (int, float)) or not 0 <= item["confidence"] <= 1:
            raise StructuredVisionError("Verification confidence is invalid.")
    return result["items"]


def canonical_group_name(name: str) -> str | None:
    """Map equivalent headings and descriptors to one of three group keys."""
    text = _normalise_item(name)
    if "primary" in text or "causes of damage" in text:
        return "primary"
    if "antagonistic" in text or "compensatory" in text or "responses to damage" in text:
        return "antagonistic"
    if (
        "integrative" in text
        or "culprits of the phenotype" in text
        or "culprits of phenotype" in text
        or "end result" in text
    ):
        return "integrative"
    return None


def _is_actual_item(item: str) -> bool:
    text = _normalise_item(item)
    placeholders = {
        "", "none", "unknown", "unreadable", "not visible", "no items",
        "no readable items", "no confidently readable items",
    }
    if text in placeholders or text.startswith("no confidently readable"):
        return False
    if METADATA_PATTERN.search(item):
        return False
    if len(text.split()) > 8 or re.search(r"[.!?]\s+", item):
        return False
    # A group heading or generic group description is not an item label.
    return canonical_group_name(item) is None


METADATA_PATTERN = re.compile(
    r"\b(?:et\s+al\.?|page\s+\d+|journal|author\s+manuscript|manuscript|"
    r"copyright|publisher|published|doi|pmc|volume|vol\.?\s*\d+|figure\s+\d+)\b",
    re.IGNORECASE,
)


def clean_detected_label(label: str) -> str:
    """Remove common page/header metadata while preserving label fragments."""
    parts = re.split(r"[,;|\n]+", label)
    kept = [part.strip(" .:-") for part in parts if not METADATA_PATTERN.search(part)]
    cleaned = " ".join(part for part in kept if part)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .:-")
    return cleaned


def _candidate_display_label(label: str) -> str:
    """Restore sentence-case display when PDF enumeration text is all lowercase."""
    if label and label == label.lower():
        return label[0].upper() + label[1:]
    return label


def _evidence_blocks(text: str, default_page: int | None = None) -> list[tuple[int | None, str]]:
    pattern = re.compile(
        r"(?:^|\n\n)Source:.*?page\s+(\d+).*?\n(.*?)(?=\n\nSource:|\Z)",
        re.IGNORECASE | re.DOTALL,
    )
    blocks = [(int(match.group(1)), match.group(2)) for match in pattern.finditer(text)]
    return blocks or [(default_page, text)]


def extract_candidate_labels(
    text: str,
    default_page: int | None = None,
) -> list[dict]:
    """Extract document-stated list items following a canonical group heading."""
    candidates: dict[str, dict] = {}
    for page, block in _evidence_blocks(text, default_page):
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", block):
            group = canonical_group_name(sentence)
            enumeration = re.search(
                r"\bhallmarks?(?:\s+described[^:]*)?\s+(?:are|include|includes)\s*:?\s*(.*)",
                sentence,
                re.IGNORECASE,
            )
            if enumeration:
                for phrase in re.split(r"\s*,\s*|\s+and\s+", enumeration.group(1)):
                    label = _candidate_display_label(clean_detected_label(phrase))
                    words = _normalise_item(label).split()
                    if 1 < len(words) <= 7 and _is_actual_item(label):
                        candidates.setdefault(_normalise_item(label), {
                            "label": label,
                            "group": group,
                            "source_page": page,
                            "sentence": sentence.strip(),
                        })
            if not group:
                continue
            direct = re.search(
                rf"^(.+?)\s+(?:is|are)\s+(?:an?\s+)?{group}\s+hallmarks?\b",
                sentence.strip(),
                re.IGNORECASE,
            )
            if direct:
                label = clean_detected_label(direct.group(1))
                words = _normalise_item(label).split()
                if 1 < len(words) <= 7 and _is_actual_item(label):
                    candidates.setdefault(_normalise_item(label), {
                        "label": label,
                        "group": group,
                        "source_page": page,
                        "sentence": sentence.strip(),
                    })
            match = re.search(
                rf"\b{group}\b[^,.]*?\bhallmarks?\b(.*)", sentence,
                re.IGNORECASE,
            )
            if not match:
                continue
            tail = re.sub(
                r"^[\s,:;-]*(?:are|is|include|includes|comprise|comprises|"
                r"consist\s+of|consists\s+of)?\s*",
                "",
                match.group(1),
                flags=re.IGNORECASE,
            )
            for phrase in re.split(r"\s*,\s*|\s+and\s+", tail):
                label = clean_detected_label(phrase)
                words = _normalise_item(label).split()
                if 1 < len(words) <= 7 and _is_actual_item(label):
                    candidates.setdefault(_normalise_item(label), {
                        "label": label,
                        "group": group,
                        "source_page": page,
                        "sentence": sentence.strip(),
                    })
    return list(candidates.values())


def normalize_visual_label(label: str, candidates: list[dict]) -> str:
    """Repair noisy OCR only by matching a label stated in document evidence."""
    cleaned = clean_detected_label(label)
    if not cleaned:
        return ""
    clean_tokens = set(_normalise_item(cleaned).split())
    if not clean_tokens:
        return ""
    exact_matches = [
        candidate for candidate in candidates
        if _normalise_item(candidate["label"]) == _normalise_item(cleaned)
    ]
    if len(exact_matches) == 1:
        return exact_matches[0]["label"]
    suffix_matches = [
        candidate for candidate in candidates
        if _normalise_item(candidate["label"]).endswith(
            f" {_normalise_item(cleaned)}"
        )
    ]
    if len(suffix_matches) == 1:
        return suffix_matches[0]["label"]
    subset_matches = [
        candidate for candidate in candidates
        if clean_tokens.issubset(set(_normalise_item(candidate["label"]).split()))
    ]
    if len(subset_matches) == 1:
        return subset_matches[0]["label"]

    scored = []
    for candidate in candidates:
        candidate_key = _normalise_item(candidate["label"])
        candidate_tokens = set(candidate_key.split())
        overlap = len(clean_tokens & candidate_tokens) / max(len(clean_tokens), len(candidate_tokens))
        similarity = SequenceMatcher(None, _normalise_item(cleaned), candidate_key).ratio()
        scored.append((0.6 * overlap + 0.4 * similarity, candidate["label"]))
    scored.sort(reverse=True)
    if scored and scored[0][0] >= 0.68 and (
        len(scored) == 1 or scored[0][0] - scored[1][0] >= 0.12
    ):
        return scored[0][1]
    return cleaned


def ambiguous_candidate_match(label: str, candidates: list[dict]) -> bool:
    """Return true when a partial OCR fragment matches multiple candidates."""
    cleaned = clean_detected_label(label)
    key = _normalise_item(cleaned)
    tokens = set(key.split())
    if not tokens or any(_normalise_item(candidate["label"]) == key for candidate in candidates):
        return False
    suffix_matches = [
        candidate for candidate in candidates
        if _normalise_item(candidate["label"]).endswith(f" {key}")
    ]
    if len(suffix_matches) > 1:
        return True
    if len(suffix_matches) == 1:
        return False
    subset_matches = [
        candidate for candidate in candidates
        if tokens.issubset(set(_normalise_item(candidate["label"]).split()))
    ]
    if len(subset_matches) > 1:
        return True
    scored = []
    for candidate in candidates:
        candidate_key = _normalise_item(candidate["label"])
        candidate_tokens = set(candidate_key.split())
        overlap = len(tokens & candidate_tokens) / max(len(tokens), len(candidate_tokens))
        similarity = SequenceMatcher(None, key, candidate_key).ratio()
        scored.append(0.6 * overlap + 0.4 * similarity)
    scored.sort(reverse=True)
    return len(scored) > 1 and scored[0] >= 0.68 and scored[0] - scored[1] < 0.12


def validate_reconciled_schema(result: dict) -> dict:
    required = {"primary", "antagonistic", "integrative", "uncertain"}
    if not isinstance(result, dict) or set(result) != required:
        raise StructuredVisionError("Reconciliation did not produce the final schema.")
    category_sets = {
        group_name: {
            _normalise_item(item)
            for item in result[group_name]
            if isinstance(item, str)
        }
        for group_name in ("primary", "antagonistic", "integrative")
    }
    detected = set().union(*category_sets.values())
    repeated_everywhere = set.intersection(*category_sets.values())
    if detected and len(repeated_everywhere) >= max(2, math.ceil(len(detected) * 0.7)):
        raise StructuredVisionError(
            "Most detected labels were repeated in every mutually exclusive category."
        )

    owners: dict[str, str] = {}
    for group_name in ("primary", "antagonistic", "integrative", "uncertain"):
        items = result[group_name]
        if not isinstance(items, list) or not all(
            isinstance(item, str) and _is_actual_item(item) for item in items
        ):
            raise StructuredVisionError(f"Final {group_name} items are invalid.")
        for item in items:
            key = _normalise_item(item)
            if key in owners:
                raise StructuredVisionError(
                    f"Final item {item!r} appears in both {owners[key]} and {group_name}."
                )
            owners[key] = group_name
    return result


def relative_y_to_page(y_center: float, crop) -> float:
    """Convert a crop-relative vertical centre to PDF page coordinates."""
    return crop.y0 + y_center * crop.height


def _explicit_text_memberships(
    evidence_text: str,
    labels: list[str],
    default_page: int | None = None,
) -> dict[str, dict]:
    """Accept membership only from a document sentence that explicitly states it."""
    candidates = extract_candidate_labels(evidence_text, default_page)
    by_label = {
        _normalise_item(candidate["label"]): candidate
        for candidate in candidates
        if candidate.get("group")
    }
    label_keys = {_normalise_item(label): label for label in labels}
    for page, block in _evidence_blocks(evidence_text, default_page):
        # PDF text commonly wraps a single sentence over several physical lines.
        prose = re.sub(r"(?<![.!?])\n", " ", block)
        for sentence in re.split(r"(?<=[.!?])\s+", prose):
            group = canonical_group_name(sentence)
            if not group:
                continue
            sentence_key = _normalise_item(sentence)
            for key, label in label_keys.items():
                if key and re.search(rf"\b{re.escape(key)}\b", sentence_key):
                    by_label[key] = {
                        "label": label,
                        "group": group,
                        "source_page": page,
                        "sentence": sentence.strip(),
                    }
    return {
        key: by_label[key]
        for label in labels
        if (key := _normalise_item(label)) in by_label
    }


def _discard_partial_label_fragments(observations: dict[str, dict]) -> None:
    """Drop an incomplete OCR fragment when one full visible expansion exists."""
    for short_key, short_record in list(observations.items()):
        short_detections = [
            detection
            for evidence in short_record["groups"].values()
            for detection in evidence
        ]
        if any(detection["fully_visible"] for detection in short_detections):
            continue
        short_tokens = set(short_key.split())
        expansions = []
        for long_key, long_record in observations.items():
            if long_key == short_key or not short_tokens < set(long_key.split()):
                continue
            long_detections = [
                detection
                for evidence in long_record["groups"].values()
                for detection in evidence
            ]
            if any(detection["fully_visible"] for detection in long_detections):
                expansions.append(long_key)
        if len(expansions) == 1:
            del observations[short_key]


def reconcile_spatial_results(
    overview: dict,
    regional: list[dict],
    region_clips: list,
    figure_clip,
    page_text: str = "",
    source_page: int | None = None,
    candidate_text: str = "",
) -> tuple[dict, dict, list[dict], list[dict]]:
    """Reconcile overlapping item crops against overview headings in page space."""
    headings = []
    for heading in overview["headings"]:
        if heading["name"] == "unknown":
            continue
        headings.append({
            **heading,
            "page_y": relative_y_to_page(heading["y_center"], figure_clip),
        })
    if not headings:
        raise StructuredVisionError("No validated overview headings were available.")
    headings.sort(key=lambda heading: heading["page_y"])

    boundaries = [figure_clip.y0]
    boundaries.extend(
        (headings[index]["page_y"] + headings[index + 1]["page_y"]) / 2
        for index in range(len(headings) - 1)
    )
    boundaries.append(figure_clip.y1)
    bands = {
        heading["name"]: (boundaries[index], boundaries[index + 1])
        for index, heading in enumerate(headings)
    }

    candidates = extract_candidate_labels(candidate_text or page_text, source_page)
    observations: dict[str, dict] = {}
    ambiguous_items: set[str] = set()
    detections = []
    for response, crop in zip(regional, region_clips):
        crop_center = (crop.y0 + crop.y1) / 2
        region_group = min(headings, key=lambda heading: abs(heading["page_y"] - crop_center))["name"]
        for item in response["items"]:
            ambiguous_match = ambiguous_candidate_match(item["label"], candidates)
            normalized_label = normalize_visual_label(item["label"], candidates)
            if not _is_actual_item(normalized_label):
                continue
            page_y = relative_y_to_page(item["y_center"], crop)
            distances = {
                heading["name"]: abs(page_y - heading["page_y"]) / figure_clip.height
                for heading in headings
            }
            nearest = min(distances, key=distances.get)
            band_start, band_end = bands[nearest]
            band_center = (band_start + band_end) / 2
            half_band = max((band_end - band_start) / 2, 1e-6)
            centrality = max(0.0, 1.0 - abs(page_y - band_center) / half_band)
            proximity = max(0.0, 1.0 - min(distances[nearest] * 2, 1.0))
            score = (
                0.35 * float(item["fully_visible"])
                + 0.25 * proximity
                + 0.20 * centrality
                + 0.20 * item["confidence"]
                + (0.05 if region_group == nearest else 0.0)
            )
            detection = {
                "raw_label": item["label"],
                "label": normalized_label,
                "region": response["region"],
                "crop_relative_y": item["y_center"],
                "page_y": round(page_y, 3),
                "heading_distances": {
                    name: round(distance, 4) for name, distance in distances.items()
                },
                "nearest_heading": nearest,
                "fully_visible": item["fully_visible"],
                "confidence": item["confidence"],
                "score": round(score, 4),
            }
            detections.append(detection)
            key = _normalise_item(normalized_label)
            if ambiguous_match:
                ambiguous_items.add(key)
            record = observations.setdefault(key, {"label": normalized_label, "groups": {}})
            evidence = record["groups"].setdefault(nearest, [])
            evidence.append(detection)

    _discard_partial_label_fragments(observations)
    final = {"primary": [], "antagonistic": [], "integrative": [], "uncertain": []}
    sources = {}
    confidences = {}
    for key, record in observations.items():
        ranked = sorted(
            (
                (
                    group,
                    max(item["score"] for item in evidence)
                    + 0.05 * (len({item["region"] for item in evidence}) - 1),
                )
                for group, evidence in record["groups"].items()
            ),
            key=lambda pair: pair[1],
            reverse=True,
        )
        if (
            key in ambiguous_items
            or not ranked
            or (len(ranked) > 1 and ranked[0][1] - ranked[1][1] < 0.08)
        ):
            final["uncertain"].append(record["label"])
            sources[key] = ["vision"]
            confidences[key] = ranked[0][1] if ranked else 0.0
        else:
            final[ranked[0][0]].append(record["label"])
            sources[key] = ["vision"]
            confidences[key] = ranked[0][1]

    text_memberships = _explicit_text_memberships(
        candidate_text or page_text,
        [record["label"] for record in observations.values()],
        source_page,
    )
    for group in ("primary", "antagonistic", "integrative"):
        for item in final[group]:
            key = _normalise_item(item)
            if text_memberships.get(key, {}).get("group") == group:
                sources[key] = ["vision", "text"]
    for item in final["uncertain"][:]:
        key = _normalise_item(item)
        text_match = text_memberships.get(key)
        if text_match:
            final["uncertain"].remove(item)
            final[text_match["group"]].append(item)
            sources[key] = ["vision", "text"]
            confidences[key] = max(confidences.get(key, 0.0), 0.85)

    final = validate_reconciled_schema(final)
    assignments = []
    for group in ("primary", "antagonistic", "integrative", "uncertain"):
        for label in final[group]:
            key = _normalise_item(label)
            text_match = text_memberships.get(key, {})
            assignments.append({
                "label": label,
                "group": group,
                "evidence": sources.get(key, ["vision"]),
                "source_page": text_match.get("source_page", source_page),
                "confidence": round(min(confidences.get(key, 0.0), 1.0), 3),
            })

    return final, assignments, detections, headings


def format_structured_answer(result: dict, question: str) -> str:
    """Create a direct answer without text-model reinterpretation."""
    question_key = _normalise_item(question)
    matches = []
    for group_name in ("primary", "antagonistic", "integrative"):
        for item in result[group_name]:
            item_key = _normalise_item(item)
            if item_key and item_key in question_key:
                matches.append((item, group_name))
    if len(matches) == 1:
        item, group_name = matches[0]
        return (
            f"**{item} belongs to {group_name.title()}.**\n\n"
            f"It is visibly placed in the {group_name} group of the diagram."
        )
    lines = [
        f"- **{group_name.title()}:** " + ", ".join(result[group_name])
        for group_name in ("primary", "antagonistic", "integrative")
        if result[group_name]
    ]
    if result["uncertain"]:
        lines.append("- **Uncertain:** " + ", ".join(result["uncertain"]))
    return "**The diagram shows these groups:**\n\n" + "\n".join(lines)


def _ask_vision_model(image_path: Path, question: str) -> str:
    return _call_vision_model(image_path=image_path, question=question)


def _rect_coordinates(rect: fitz.Rect) -> dict:
    return {name: round(getattr(rect, name), 2) for name in ("x0", "y0", "x1", "y1")}


def _analyse_grouped_figure(
    page: fitz.Page,
    question: str,
    debug_info: dict | None = None,
    text_evidence: str = "",
    source_page: int | None = None,
    candidate_text: str = "",
) -> str:
    last_error: Exception | None = None
    save_crops = bool(debug_info and debug_info.get("_save_crops"))
    for tight in (False, True):
        image_paths: list[Path] = []
        try:
            figure_clip = _detect_figure_clip(page, question)
            positions = ("overview", "top", "middle", "bottom")
            clips = [figure_clip, *_region_clips(figure_clip, tight=tight)]
            debug_folder = (
                Path(mkdtemp(prefix="research_vision_crops_"))
                if save_crops
                else None
            )
            image_paths = [
                _render_page_image(
                    page,
                    1.5,
                    clip,
                    debug_folder / f"{position}.png" if debug_folder else None,
                )
                for position, clip in zip(positions, clips)
            ]
            if debug_info is not None:
                debug_info.clear()
                if save_crops:
                    debug_info.update({
                        "crop_folder": str(debug_folder),
                        "crops": [
                        {
                            "name": position,
                            "path": str(path),
                            "coordinates": _rect_coordinates(clip),
                        }
                        for position, path, clip in zip(positions, image_paths, clips)
                        ],
                    })
                debug_info["tight_retry"] = tight
            overview = _call_overview_model(image_paths[0], question, strict=tight)
            regional = [
                _call_regional_model(path, question, position)
                for position, path in zip(positions[1:], image_paths[1:])
            ]
            page_text = page.get_text("text")
            combined_evidence = (
                f"Source: selected PDF, page {source_page}\n{page_text}\n\n{text_evidence}"
            )
            result, assignments, detections, headings = reconcile_spatial_results(
                overview,
                regional,
                clips[1:],
                figure_clip,
                combined_evidence,
                source_page,
                candidate_text,
            )
            if debug_info is not None:
                debug_info["overview_headings"] = headings
                debug_info["detections"] = detections
                debug_info["assignments"] = assignments
                debug_info["normalized_json"] = result
                debug_info["final_answer_path"] = "validated_structured_vision"
            return format_structured_answer(result, question)
        except Exception as error:
            last_error = error
        finally:
            if not save_crops:
                for image_path in image_paths:
                    image_path.unlink(missing_ok=True)
    raise StructuredVisionError(
        f"Structured figure analysis failed after validation retry: {last_error}"
    ) from last_error


def _analyse_typed_page(
    page: fitz.Page,
    question: str,
    visual_type: str,
    text_evidence: str,
    debug_info: dict | None,
) -> str:
    target_caption = _target_caption(page, question) if visual_type != "table" else None
    clip = (
        _detect_table_clip(page, question)
        if visual_type == "table"
        else _detect_figure_clip(page, question)
    )
    save_crops = bool(debug_info and debug_info.get("_save_crops"))
    debug_folder = (
        Path(mkdtemp(prefix="research_vision_typed_"))
        if save_crops
        else None
    )
    page_text = page.get_text("text")
    caption_evidence = (
        f"TARGET FIGURE CAPTION (authoritative for this crop):\n{target_caption[1]}\n\n"
        if target_caption
        else ""
    )
    combined_evidence = (
        f"{caption_evidence}PAGE TEXT CROSS-CHECK:\n{page_text}\n\n"
        f"RETRIEVED TEXT CROSS-CHECK:\n{text_evidence}"
    )

    panel_labels = _multi_panel_labels(page_text) if visual_type == "graph" else []
    panel_detection_raw = ""
    panel_detection_path = None
    caption_text = target_caption[1] if target_caption else ""
    if visual_type == "graph" and not panel_labels and re.search(
        r"\bbode\b", caption_text, re.IGNORECASE
    ):
        panel_detection_path = _render_page_image(
            page,
            1.0,
            clip,
            debug_folder / "panel_detection.png" if debug_folder else None,
        )
        try:
            panel_labels, panel_detection_raw = detect_compact_graph_panel_ids(
                panel_detection_path
            )
        finally:
            if not save_crops:
                panel_detection_path.unlink(missing_ok=True)
    if panel_labels:
        pair_clips = _graph_panel_pair_clips(clip, len(panel_labels))
        panel_groups = [
            panel_labels[index:index + 2]
            for index in range(0, len(panel_labels), 2)
        ]
        image_paths = [
            _render_page_image(
                page,
                2.5,
                pair_clip,
                (
                    debug_folder / f"panels_{group[0]}_{group[-1]}.png"
                    if debug_folder else None
                ),
            )
            for pair_clip, group in zip(pair_clips, panel_groups)
        ]
        axis_label_clips = [
            _magnitude_y_axis_label_clip(pair_clip) for pair_clip in pair_clips
        ]
        axis_label_image_paths = [
            _render_page_image(
                page,
                5.0,
                axis_clip,
                (
                    debug_folder / f"magnitude_y_axis_{group[0]}.png"
                    if debug_folder else None
                ),
            )
            for axis_clip, group in zip(axis_label_clips, panel_groups)
        ]
        if debug_info is not None:
            debug_info.clear()
            debug_info.update({
                "target_figure": _figure_number(question),
                "target_caption": target_caption[1] if target_caption else "",
                "crop_coordinates": _rect_coordinates(clip),
                "panel_groups": panel_groups,
                "panel_detection_response": panel_detection_raw,
            })
            if save_crops:
                debug_info.update({
                    "crop_folder": str(debug_folder),
                    "crops": ([{
                        "name": "panel_detection",
                        "path": str(panel_detection_path),
                        "coordinates": _rect_coordinates(clip),
                    }] if panel_detection_path else []) + [
                        {
                            "name": f"panels_{group[0]}_{group[-1]}",
                            "path": str(path),
                            "coordinates": _rect_coordinates(pair_clip),
                        }
                        for path, pair_clip, group in zip(image_paths, pair_clips, panel_groups)
                    ] + [
                        {
                            "name": f"magnitude_y_axis_{group[0]}",
                            "path": str(path),
                            "coordinates": _rect_coordinates(axis_clip),
                        }
                        for path, axis_clip, group in zip(
                            axis_label_image_paths, axis_label_clips, panel_groups
                        )
                    ],
                })
        try:
            return analyse_compact_multi_panel_graph(
                image_paths=image_paths,
                panel_groups=panel_groups,
                figure_number=_figure_number(question) or "",
                question=question,
                evidence_text=combined_evidence,
                axis_label_image_paths=axis_label_image_paths,
                debug_info=debug_info,
            )
        finally:
            if not save_crops:
                for path in image_paths:
                    path.unlink(missing_ok=True)
                for path in axis_label_image_paths:
                    path.unlink(missing_ok=True)

    image_path = _render_page_image(
        page,
        _typed_render_scale(page, clip),
        clip,
        debug_folder / f"{visual_type}.png" if debug_folder else None,
    )
    fit_verification_images: list[tuple[str, Path]] = []
    fit_verification_clips = []
    two_panel_labels = (
        _two_panel_graph_labels(page_text) if visual_type == "graph" else []
    )
    asks_for_two_panel_fit = bool(
        re.search(
            r"\b(?:samples?|groups?)\s+[^.?!]{0,40}"
            r"\b(?:and|versus|vs\.?)\b",
            question,
            re.IGNORECASE,
        )
        and re.search(
            r"\b(?:closer|better|fit|deviation)\b",
            question,
            re.IGNORECASE,
        )
    )
    if not two_panel_labels and asks_for_two_panel_fit:
        two_panel_labels = ["panel_1", "panel_2"]
    if (
        len(two_panel_labels) == 2
        and clip.width > clip.height * 1.35
        and re.search(r"\bnyquist\b", combined_evidence, re.IGNORECASE)
        and re.search(
            r"\b(?:closer|better)\b.{0,30}\bfit\b|\bfit\b.{0,30}\bcloser\b",
            question,
            re.IGNORECASE,
        )
    ):
        fit_verification_clips = _side_by_side_panel_clips(clip)
        fit_verification_images = [
            (
                label,
                _render_page_image(
                    page,
                    3.0,
                    panel_clip,
                    (
                        debug_folder / f"fit_panel_{label}.png"
                        if debug_folder else None
                    ),
                ),
            )
            for label, panel_clip in zip(
                two_panel_labels, fit_verification_clips
            )
        ]
    if debug_info is not None:
        debug_info.clear()
        debug_info.update({
            "target_figure": _figure_number(question),
            "target_caption": target_caption[1] if target_caption else "",
            "crop_coordinates": _rect_coordinates(clip),
        })
        if save_crops:
            debug_info.update({
                "crop_folder": str(debug_folder),
                "crops": [{
                    "name": visual_type,
                    "path": str(image_path),
                    "coordinates": _rect_coordinates(clip),
                }, *[
                    {
                        "name": f"fit_panel_{label}",
                        "path": str(path),
                        "coordinates": _rect_coordinates(panel_clip),
                    }
                    for (label, path), panel_clip in zip(
                        fit_verification_images, fit_verification_clips
                    )
                ]],
            })
    try:
        return analyse_typed_image(
            image_path=image_path,
            visual_type=visual_type,
            question=question,
            evidence_text=combined_evidence,
            debug_info=debug_info,
            fit_verification_images=fit_verification_images,
        )
    finally:
        if not save_crops:
            image_path.unlink(missing_ok=True)
            for _, path in fit_verification_images:
                path.unlink(missing_ok=True)


def analyse_pdf_page(
    pdf_path: Path,
    page_number: int,
    question: str,
    debug_info: dict | None = None,
    text_evidence: str = "",
) -> str:
    """Analyse one PDF page locally, with structured grouping when useful."""
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    document = fitz.open(pdf_path)
    try:
        if page_number < 1 or page_number > len(document):
            raise ValueError(
                f"Invalid page number. The PDF contains {len(document)} pages."
            )
        page = document.load_page(page_number - 1)
        page_text = page.get_text("text")
        visual_type = detect_visual_type(question, page_text)

        # Graphs and tables must be dispatched before the legacy grouping
        # keyword check because their legends/rows commonly contain "groups".
        # Non-grouping diagrams use their own relationship schema as well.
        if visual_type in {"graph", "table"} or (
            visual_type == "labelled_diagram" and not _is_grouping_question(question)
        ):
            return _analyse_typed_page(
                page,
                question,
                visual_type,
                text_evidence,
                debug_info,
            )

        if _is_grouping_question(question):
            # Grouping answers must never fall through to unvalidated prose.
            # _analyse_grouped_figure performs both normal and tighter-crop
            # structured attempts before raising a validation error.
            candidate_text = "\n\n".join(
                f"Source: {pdf_path.name}, page {index + 1}\n{pdf_page.get_text('text')}"
                for index, pdf_page in enumerate(document)
            )
            return _analyse_grouped_figure(
                page,
                question,
                debug_info,
                text_evidence,
                page_number,
                candidate_text,
            )

        last_error: Exception | None = None
        for scale in (1.0, 0.75, 0.5):
            image_path = _render_page_image(page, scale)
            try:
                return _ask_vision_model(image_path, question)
            except Exception as error:
                last_error = error
                message = str(error).lower()
                is_context_error = (
                    "context size" in message
                    or "exceed_context_size" in message
                    or "exceeds the available context" in message
                )
                if not is_context_error:
                    raise
            finally:
                image_path.unlink(missing_ok=True)

        raise RuntimeError(
            "Vision analysis failed because the rendered page still exceeded "
            "the model context window after automatic downscaling."
        ) from last_error
    finally:
        document.close()
