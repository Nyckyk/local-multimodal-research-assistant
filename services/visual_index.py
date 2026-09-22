"""Build and cache a local text-first index of figures and tables in PDFs."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import fitz

from services.visual_reference_parser import canonical_identifier
from settings import PAPERS_FOLDER, VISUAL_INDEX_PATH


INDEX_VERSION = 6
_IDENTIFIER = r"(?:[A-Za-z]\.)?\d+(?:\.\d+)?|[A-Za-z]\d+"
_FIGURE_CAPTION_PATTERN = re.compile(
    rf"^\s*(?P<kind>fig(?:ure)?\.?)\s*(?P<number>{_IDENTIFIER})(?:\s*[.:]\s*|\s+)(?P<title>.+)$",
    re.IGNORECASE,
)
_TABLE_ONLY_PATTERN = re.compile(
    rf"^\s*(?P<kind>table)\s*(?P<number>{_IDENTIFIER})\s*[.:]?\s*$",
    re.IGNORECASE,
)
_TABLE_CAPTION_PATTERN = re.compile(
    rf"^\s*(?P<kind>table)\s*(?P<number>{_IDENTIFIER})(?:\s*[.:]\s*|\s+)"
    r"(?P<title>.+)$",
    re.IGNORECASE,
)
_REFERENCE_VERBS = re.compile(
    r"^(?:reports?|represents?|presents?|shows?|provides?|illustrates?|displays?|is|was)\b",
    re.IGNORECASE,
)
_REFERENCE_PATTERN = re.compile(
    rf"\b(?P<kind>fig(?:ure)?\.?|table)\s*(?P<number>{_IDENTIFIER})\b",
    re.IGNORECASE,
)
_EQUATION_NUMBER_PATTERN = re.compile(
    r"(?m)^\s*\((?P<number>\d+(?:\.\d+)?[a-z]?)\)\s*$",
    re.IGNORECASE,
)


def _file_signature(path: Path) -> dict:
    stat = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": digest.hexdigest(),
    }


def _quick_signature_matches(path: Path, cached: dict) -> bool:
    try:
        stat = path.stat()
    except OSError:
        return False
    signature = cached.get("signature", {}) if isinstance(cached, dict) else {}
    return (
        signature.get("size") == stat.st_size
        and signature.get("mtime_ns") == stat.st_mtime_ns
        and bool(signature.get("sha256"))
    )


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _equation_records(text: str) -> list[dict]:
    """Index displayed equation numbers and their immediate text context."""
    records = []
    seen = set()
    for match in _EQUATION_NUMBER_PATTERN.finditer(text or ""):
        number = match.group("number")
        key = canonical_identifier(number)
        if key in seen:
            continue
        seen.add(key)
        start = max(0, match.start() - 900)
        end = min(len(text), match.end() + 900)
        records.append({
            "target_type": "equation",
            "target_number": number,
            "caption": _clean_text(text[start:end])[:1800],
            "confidence": 0.92,
        })
    return records


def _printed_page_number(page: fitz.Page) -> str | None:
    blocks = sorted(page.get_text("blocks"), key=lambda row: (row[1], row[0]))
    if not blocks:
        return None
    page_height = page.rect.height
    edge_text = [
        _clean_text(block[4])
        for block in blocks
        if block[1] <= page_height * 0.08 or block[3] >= page_height * 0.92
    ]
    for text in edge_text:
        match = re.fullmatch(r"(?:page\s+)?([ivxlcdm]+|\d{1,4})", text, re.I)
        if match:
            return match.group(1)
        labelled = re.search(r"\bpage\s+(\d{1,4})\b", text, re.I)
        if labelled:
            return labelled.group(1)
    return None


def _caption_blocks(page: fitz.Page) -> list[dict]:
    raw_blocks = sorted(page.get_text("blocks"), key=lambda row: (row[1], row[0]))
    blocks = [
        {"x0": row[0], "y0": row[1], "x1": row[2], "y1": row[3], "text": _clean_text(row[4])}
        for row in raw_blocks if _clean_text(row[4])
    ]
    captions = []
    for index, block in enumerate(blocks):
        figure_match = _FIGURE_CAPTION_PATTERN.match(block["text"])
        match = (
            figure_match
            if figure_match and not _REFERENCE_VERBS.match(figure_match.group("title"))
            else _TABLE_ONLY_PATTERN.match(block["text"])
        )
        if not match:
            table_match = _TABLE_CAPTION_PATTERN.match(block["text"])
            if table_match and not _REFERENCE_VERBS.match(table_match.group("title")):
                match = table_match
        if not match:
            continue
        caption_parts = [block["text"]]
        # Some PDFs put only "Table 1" or "Fig. A.1" in one text block.
        # Join a close following block, but never scan across the whole page.
        if len(caption_parts[0].split()) <= 4 and index + 1 < len(blocks):
            following = blocks[index + 1]
            vertical_gap = following["y0"] - block["y1"]
            if -3 <= vertical_gap <= page.rect.height * 0.035:
                caption_parts.append(following["text"])

        connected = [block]
        changed = True
        while changed:
            changed = False
            for candidate in blocks:
                if candidate in connected or candidate["text"] in caption_parts:
                    continue
                if _FIGURE_CAPTION_PATTERN.match(candidate["text"]) or _TABLE_CAPTION_PATTERN.match(candidate["text"]):
                    continue
                touches = any(
                    (
                        min(candidate["y1"], current["y1"])
                        - max(candidate["y0"], current["y0"])
                    ) >= min(candidate["y1"] - candidate["y0"], current["y1"] - current["y0"]) * 0.35
                    or (
                        abs(candidate["x0"] - current["x0"]) <= getattr(page.rect, "width", 600) * 0.04
                        and -3 <= candidate["y0"] - current["y1"] <= page.rect.height * 0.012
                    )
                    for current in connected
                )
                if touches and not re.match(
                    r"^(?:Article\s+https?://|Nature\s+Communications)", candidate["text"], re.I
                ):
                    connected.append(candidate)
                    changed = True
        connected.sort(key=lambda candidate: (candidate["x0"], candidate["y0"]))
        caption_parts.extend(
            candidate["text"] for candidate in connected
            if candidate["text"] not in caption_parts
        )

        # Journal captions commonly continue in an adjacent column. When a
        # caption begins in the page header band, collect every geometrically
        # connected header-band block in column order. This preserves long
        # multi-panel captions without swallowing the Results text below.
        if block["y0"] <= page.rect.height * 0.25:
            header_blocks = [
                candidate for candidate in blocks
                if candidate["y0"] <= page.rect.height * 0.25
                and candidate["y1"] <= page.rect.height * 0.30
                and candidate["text"] not in caption_parts
                and not re.match(r"^(?:Article\s+https?://|Nature\s+Communications)", candidate["text"], re.I)
                and not _FIGURE_CAPTION_PATTERN.match(candidate["text"])
                and not _TABLE_CAPTION_PATTERN.match(candidate["text"])
            ]
            header_blocks.sort(key=lambda candidate: (candidate["x0"], candidate["y0"]))
            caption_parts.extend(candidate["text"] for candidate in header_blocks)
        caption = _clean_text(" ".join(caption_parts))
        captions.append({
            "target_type": "table" if match.group("kind").casefold().startswith("table") else "figure",
            "target_number": match.group("number"),
            "caption": caption,
            "full_caption": caption,
            "short_caption": caption[:360].rsplit(" ", 1)[0] + ("…" if len(caption) > 360 else ""),
            "caption_y0": float(block["y0"]),
            "confidence": 0.96,
        })
    return captions


def extract_page_record(page: fitz.Page, page_number: int) -> dict:
    text = page.get_text("text") or ""
    captions = _caption_blocks(page)
    equations = _equation_records(text)
    caption_keys = {
        (row["target_type"], canonical_identifier(row["target_number"]))
        for row in captions
    }
    references = []
    for match in _REFERENCE_PATTERN.finditer(text):
        target_type = "table" if match.group("kind").casefold().startswith("table") else "figure"
        key = (target_type, canonical_identifier(match.group("number")))
        if key in caption_keys:
            continue
        if key not in {
            (row["target_type"], canonical_identifier(row["target_number"]))
            for row in references
        }:
            references.append({
                "target_type": target_type,
                "target_number": match.group("number"),
                "confidence": 0.48,
            })
    return {
        "pdf_page": int(page_number),
        "printed_page_number": _printed_page_number(page),
        "figure_identifiers": [
            row["target_number"] for row in [*captions, *references]
            if row["target_type"] == "figure"
        ],
        "table_identifiers": [
            row["target_number"] for row in [*captions, *references]
            if row["target_type"] == "table"
        ],
        "captions": captions,
        "references": references,
        "equations": equations,
        "nearby_text": _clean_text(text)[:6000],
        "confidence": max([row["confidence"] for row in captions] or [0.25]),
    }


def _index_pdf(path: Path, signature: dict) -> dict:
    indexed_at = datetime.now(timezone.utc).isoformat()
    try:
        with fitz.open(path) as document:
            pages = [
                extract_page_record(page, index + 1)
                for index, page in enumerate(document)
            ]
            for page_index, page_record in enumerate(pages):
                page = document[page_index]
                current_images = sum(
                    rect.get_area() for image in page.get_images(full=True)
                    for rect in page.get_image_rects(image[0])
                ) / max(1.0, page.rect.get_area())
                previous_images = 0.0
                if page_index:
                    previous_page = document[page_index - 1]
                    previous_images = sum(
                        rect.get_area() for image in previous_page.get_images(full=True)
                        for rect in previous_page.get_image_rects(image[0])
                    ) / max(1.0, previous_page.rect.get_area())
                for caption in page_record["captions"]:
                    caption["caption_page_number"] = page_index + 1
                    caption["visual_page_number"] = page_index + 1
                    if (
                        caption["target_type"] == "figure"
                        and page_index
                        and caption.get("caption_y0", page.rect.height) <= page.rect.height * 0.16
                        and previous_images >= max(0.08, current_images * 1.5)
                    ):
                        caption["visual_page_number"] = page_index
        return {
            "pdf_filename": path.name,
            "signature": signature,
            "indexed_at": indexed_at,
            "page_count": len(pages),
            "pages": pages,
            "error": "",
        }
    except Exception as error:
        return {
            "pdf_filename": path.name,
            "signature": signature,
            "indexed_at": indexed_at,
            "page_count": 0,
            "pages": [],
            "error": str(error),
        }


def _load_cache(cache_path: Path) -> dict:
    if not cache_path.exists():
        return {"version": INDEX_VERSION, "files": {}}
    try:
        value = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"version": INDEX_VERSION, "files": {}}
    # Legacy indexes used a top-level documents/files list. Treat their rows as
    # reusable only when they already contain the current page schema.
    if isinstance(value.get("files"), list):
        value["files"] = {
            row.get("pdf_filename", ""): row
            for row in value["files"] if isinstance(row, dict) and row.get("pdf_filename")
        }
    if not isinstance(value.get("files"), dict):
        value["files"] = {}
    value.setdefault("version", 0)
    return value


def load_or_build_visual_index(
    papers_folder: Path = PAPERS_FOLDER,
    cache_path: Path = VISUAL_INDEX_PATH,
    force: bool = False,
    progress_callback=None,
) -> dict:
    """Reuse unchanged PDFs and rebuild only added/modified files."""
    papers_folder = Path(papers_folder)
    cache_path = Path(cache_path)
    cached = _load_cache(cache_path)
    schema_changed = cached.get("version") != INDEX_VERSION
    current_files = sorted(papers_folder.glob("*.pdf"), key=lambda path: path.name.casefold())
    updated_files = {}
    changed = (
        force or schema_changed
        or set(cached["files"]) != {path.name for path in current_files}
    )
    for position, path in enumerate(current_files, start=1):
        old = cached["files"].get(path.name, {})
        if not force and not schema_changed and _quick_signature_matches(path, old):
            updated_files[path.name] = old
        else:
            signature = _file_signature(path)
            updated_files[path.name] = _index_pdf(path, signature)
            changed = True
        if progress_callback:
            progress_callback(position, len(current_files), path.name)
    value = {
        "version": INDEX_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "files": updated_files,
    }
    if changed or not cache_path.exists():
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
        temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
        temporary.replace(cache_path)
    else:
        value["generated_at"] = cached.get("generated_at", value["generated_at"])
    return value


def visual_index_needs_rebuild(
    papers_folder: Path = PAPERS_FOLDER,
    cache_path: Path = VISUAL_INDEX_PATH,
) -> bool:
    cached = _load_cache(Path(cache_path))
    if cached.get("version") != INDEX_VERSION:
        return True
    paths = sorted(Path(papers_folder).glob("*.pdf"), key=lambda path: path.name.casefold())
    if set(cached.get("files", {})) != {path.name for path in paths}:
        return True
    return any(
        not _quick_signature_matches(path, cached["files"].get(path.name, {}))
        for path in paths
    )


def flatten_visual_targets(index: dict, papers_folder: Path = PAPERS_FOLDER) -> list[dict]:
    targets = []
    for filename, file_record in index.get("files", {}).items():
        for page in file_record.get("pages", []):
            for caption in page.get("captions", []):
                targets.append({
                    **caption,
                    "pdf_name": filename,
                    "pdf_path": str(Path(papers_folder) / filename),
                    "page_number": int(caption.get("visual_page_number", page["pdf_page"])),
                    "visual_page_number": int(caption.get("visual_page_number", page["pdf_page"])),
                    "caption_page_number": int(caption.get("caption_page_number", page["pdf_page"])),
                    "printed_page_number": page.get("printed_page_number"),
                    "nearby_text": page.get("nearby_text", ""),
                    "match_kind": "caption",
                })
            for reference in page.get("references", []):
                targets.append({
                    **reference,
                    "caption": "",
                    "pdf_name": filename,
                    "pdf_path": str(Path(papers_folder) / filename),
                    "page_number": int(page["pdf_page"]),
                    "printed_page_number": page.get("printed_page_number"),
                    "nearby_text": page.get("nearby_text", ""),
                    "match_kind": "reference",
                })
            for equation in page.get("equations", []):
                targets.append({
                    **equation,
                    "pdf_name": filename,
                    "pdf_path": str(Path(papers_folder) / filename),
                    "page_number": int(page["pdf_page"]),
                    "printed_page_number": page.get("printed_page_number"),
                    "nearby_text": page.get("nearby_text", ""),
                    "match_kind": "equation",
                })
    return targets
