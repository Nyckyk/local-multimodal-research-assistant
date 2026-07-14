"""Conservative local-vision fallback for text-unresolved visual references."""

from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile

import fitz
import ollama

from services.structured_vision import parse_json_response
from services.visual_locator import VisualResolution, choose_visual_type
from services.visual_reference_parser import VisualReference, canonical_identifier
from settings import PAPERS_FOLDER, VISION_MODEL


FALLBACK_CONFIDENCE = 0.85
MAX_FALLBACK_PAGES = 24


def _response_text(response) -> str:
    if isinstance(response, dict):
        return str(response.get("message", {}).get("content", "")).strip()
    message = getattr(response, "message", None)
    return str(getattr(message, "content", "") or "").strip()


def _inspect_page(
    pdf_path: Path, page_number: int, target_type: str, target_number: str,
) -> dict:
    image_path = None
    try:
        with fitz.open(pdf_path) as document:
            page = document.load_page(page_number - 1)
            pixmap = page.get_pixmap(matrix=fitz.Matrix(1.0, 1.0), alpha=False)
            with NamedTemporaryFile(suffix=".png", delete=False) as temporary:
                image_path = Path(temporary.name)
            pixmap.save(str(image_path))
        prompt = f"""
Inspect this single PDF page only for an explicitly visible {target_type}
identifier {target_number}. Do not infer from scientific content. A match
requires the printed caption/heading identifier itself to be visible.
Return JSON only:
{{"found": false, "target_type": "{target_type}",
  "target_number": "{target_number}", "caption": "", "confidence": 0.0}}
"""
        response = ollama.chat(
            model=VISION_MODEL,
            messages=[{
                "role": "user", "content": prompt, "images": [str(image_path)]
            }],
            format="json",
            options={"temperature": 0, "num_ctx": 8192, "num_predict": 220},
        )
        value = parse_json_response(_response_text(response))
        return value
    finally:
        if image_path:
            image_path.unlink(missing_ok=True)


def _candidate_pages(
    index: dict, base: VisualResolution, selected_pdf: Path | str | None,
) -> list[tuple[str, int]]:
    selected_name = Path(selected_pdf).name if selected_pdf else None
    candidates = []
    for candidate in base.candidates:
        candidates.append((candidate["pdf_name"], int(candidate["page_number"])))
    for filename, file_record in index.get("files", {}).items():
        if selected_name and filename.casefold() != selected_name.casefold():
            continue
        for page in file_record.get("pages", []):
            if len(str(page.get("nearby_text", "")).strip()) < 120:
                candidates.append((filename, int(page["pdf_page"])))
    if selected_name and not candidates:
        record = index.get("files", {}).get(selected_name, {})
        candidates.extend(
            (selected_name, int(page["pdf_page"]))
            for page in record.get("pages", [])
        )
    return list(dict.fromkeys(candidates))[:MAX_FALLBACK_PAGES]


def resolve_with_visual_fallback(
    question: str,
    base_resolution: VisualResolution,
    index: dict,
    selected_pdf: Path | str | None = None,
    papers_folder: Path = PAPERS_FOLDER,
    debug_info: dict | None = None,
) -> VisualResolution:
    """Use local vision only after text resolution is absent/low-confidence."""
    eligible = base_resolution.status == "not_found" or (
        base_resolution.status == "ambiguous"
        and "confidence" in base_resolution.reason.casefold()
    )
    if not eligible or not base_resolution.target_number:
        return base_resolution
    pages = _candidate_pages(index, base_resolution, selected_pdf)
    hits, errors = [], []
    for filename, page_number in pages:
        try:
            value = _inspect_page(
                Path(papers_folder) / filename,
                page_number,
                base_resolution.target_type,
                base_resolution.target_number,
            )
        except Exception as error:
            errors.append(f"{filename}, page {page_number}: {error}")
            continue
        confidence = value.get("confidence")
        if (
            value.get("found") is True
            and isinstance(confidence, (int, float))
            and confidence >= FALLBACK_CONFIDENCE
            and str(value.get("target_type", "")).casefold()
            == base_resolution.target_type.casefold()
            and canonical_identifier(value.get("target_number"))
            == canonical_identifier(base_resolution.target_number)
        ):
            hits.append({
                "pdf_name": filename,
                "page_number": page_number,
                "caption": str(value.get("caption", "")),
                "confidence": float(confidence),
            })
    if debug_info is not None:
        debug_info.update({
            "attempted": True,
            "candidate_pages": [
                {"pdf_name": name, "page_number": page} for name, page in pages
            ],
            "hits": hits,
            "errors": errors,
        })
    if not hits:
        return base_resolution
    hits.sort(key=lambda row: row["confidence"], reverse=True)
    if len(hits) > 1 and hits[0]["confidence"] - hits[1]["confidence"] < 0.08:
        return VisualResolution(
            status="ambiguous",
            target_type=base_resolution.target_type,
            target_number=base_resolution.target_number,
            panel=base_resolution.panel,
            confidence=round(hits[0]["confidence"], 3),
            candidate_count=len(hits),
            reason="Multiple similarly confident local-vision matches",
            candidates=hits,
            reference=base_resolution.reference,
        )
    top = hits[0]
    reference = VisualReference(
        target_type=base_resolution.target_type,
        target_number=base_resolution.target_number,
        panel=base_resolution.panel,
        explicit_reference=True,
    )
    return VisualResolution(
        status="resolved",
        pdf_path=str(Path(papers_folder) / top["pdf_name"]),
        pdf_name=top["pdf_name"],
        page_number=top["page_number"],
        target_type=base_resolution.target_type,
        target_number=base_resolution.target_number,
        panel=base_resolution.panel,
        caption=top["caption"],
        visual_type=choose_visual_type(reference, question, top["caption"]),
        confidence=round(top["confidence"], 3),
        candidate_count=len(hits),
        reason="Exact identifier verified by local vision fallback",
        candidates=hits,
        reference=base_resolution.reference,
    )
