import hashlib
import io
import re
from pathlib import Path

from pypdf import PdfReader

from settings import CHUNK_OVERLAP, CHUNK_SIZE, PAPERS_FOLDER, TEXT_FOLDER


def safe_filename(filename: str) -> str:
    cleaned = re.sub(
        r'[<>:"/\\|?*]',
        "_",
        Path(filename).name,
    )
    return cleaned.strip()


def extract_pdf_pages(pdf_bytes: bytes) -> list[tuple[int, str]]:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    pages: list[tuple[int, str]] = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()

        if text:
            pages.append((page_number, text))

    return pages


def save_extracted_text(
    pdf_stem: str,
    pages: list[tuple[int, str]],
) -> Path:
    output_path = TEXT_FOLDER / f"{pdf_stem}.txt"

    sections = [
        f"--- Page {page_number} ---\n{page_text}"
        for page_number, page_text in pages
    ]

    output_path.write_text(
        "\n\n".join(sections),
        encoding="utf-8",
    )

    return output_path


def split_into_chunks(text: str) -> list[str]:
    chunks: list[str] = []
    start = 0

    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = end - CHUNK_OVERLAP

    return chunks


def index_pdf(
    collection,
    embedder,
    pdf_filename: str,
    pdf_bytes: bytes,
) -> dict:
    filename = safe_filename(pdf_filename)

    if not filename.lower().endswith(".pdf"):
        raise ValueError("The uploaded file is not a PDF.")

    pdf_path = PAPERS_FOLDER / filename
    pdf_path.write_bytes(pdf_bytes)

    pages = extract_pdf_pages(pdf_bytes)

    if not pages:
        raise ValueError(
            "No readable text was found. "
            "The PDF may be scanned or image-only."
        )

    text_path = save_extracted_text(
        pdf_path.stem,
        pages,
    )

    collection.delete(
        where={"source": text_path.name}
    )

    documents: list[str] = []
    metadatas: list[dict] = []
    ids: list[str] = []

    source_hash = hashlib.sha256(
        filename.lower().encode("utf-8")
    ).hexdigest()[:16]

    for page_number, page_text in pages:
        page_chunks = split_into_chunks(page_text)

        for chunk_number, chunk in enumerate(page_chunks):
            documents.append(chunk)

            metadatas.append(
                {
                    "source": text_path.name,
                    "pdf": filename,
                    "page": page_number,
                    "chunk": chunk_number,
                }
            )

            ids.append(
                f"{source_hash}"
                f"_page_{page_number}"
                f"_chunk_{chunk_number}"
            )

    if not documents:
        raise ValueError(
            "The PDF did not produce usable text chunks."
        )

    embeddings = embedder.encode(
        documents,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).tolist()

    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    return {
        "filename": filename,
        "pages": len(pages),
        "chunks": len(documents),
    } 