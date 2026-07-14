from pathlib import Path
import fitz  # PyMuPDF

papers_folder = Path("papers")
output_folder = Path("data/extracted_text")
output_folder.mkdir(parents=True, exist_ok=True)

pdf_files = list(papers_folder.rglob("*.pdf"))

for pdf_path in pdf_files:
    print(f"Reading: {pdf_path.name}")

    try:
        doc = fitz.open(pdf_path)
        full_text = []

        for page_num, page in enumerate(doc, start=1):
            text = page.get_text()
            full_text.append(f"\n\n--- Page {page_num} ---\n\n{text}")

        output_file = output_folder / f"{pdf_path.stem}.txt"

        with open(output_file, "w", encoding="utf-8") as f:
            f.write("\n".join(full_text))

        print(f"Saved text to: {output_file}")

    except Exception as e:
        print(f"Error reading {pdf_path.name}: {e}")

print("Done.")