import re
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

TEXT_FOLDER = Path("data/extracted_text")
DB_PATH = "data/chroma"
COLLECTION_NAME = "papers"

CHUNK_SIZE = 1600
CHUNK_OVERLAP = 300


def split_pages(text: str) -> list[tuple[int, str]]:
    """Split text using markers produced by extract_text.py."""
    parts = re.split(r"\s*--- Page (\d+) ---\s*", text)

    pages = []

    # parts format: [text_before_marker, page_number, page_text, ...]
    for index in range(1, len(parts), 2):
        page_number = int(parts[index])
        page_text = parts[index + 1].strip()

        if page_text:
            pages.append((page_number, page_text))

    return pages


def chunk_page(text: str) -> list[str]:
    """Create overlapping chunks without producing tiny empty pieces."""
    chunks = []
    start = 0

    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end == len(text):
            break

        start = end - CHUNK_OVERLAP

    return chunks


def main() -> None:
    client = chromadb.PersistentClient(path=DB_PATH)

    # Rebuild the collection so stale or duplicate chunks are removed.
    try:
        client.delete_collection(COLLECTION_NAME)
        print("Removed old index.")
    except Exception:
        pass

    collection = client.create_collection(COLLECTION_NAME)
    model = SentenceTransformer("all-MiniLM-L6-v2")

    text_files = list(TEXT_FOLDER.glob("*.txt"))

    if not text_files:
        print("No extracted text files found.")
        return

    total_chunks = 0

    for txt_file in text_files:
        text = txt_file.read_text(encoding="utf-8", errors="ignore")
        pages = split_pages(text)

        ids = []
        documents = []
        metadatas = []

        for page_number, page_text in pages:
            page_chunks = chunk_page(page_text)

            for chunk_number, chunk in enumerate(page_chunks):
                ids.append(
                    f"{txt_file.stem}_page_{page_number}_chunk_{chunk_number}"
                )
                documents.append(chunk)
                metadatas.append(
                    {
                        "source": txt_file.name,
                        "page": page_number,
                        "chunk": chunk_number,
                    }
                )

        embeddings = model.encode(
            documents,
            normalize_embeddings=True,
            show_progress_bar=True,
        ).tolist()

        collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        total_chunks += len(documents)
        print(f"Indexed {txt_file.name}: {len(documents)} chunks")

    print(f"Done. Indexed {total_chunks} chunks.")


if __name__ == "__main__":
    main() 