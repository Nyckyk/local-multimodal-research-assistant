from pathlib import Path
from pypdf import PdfReader
import csv

papers_folder = Path("papers")
output_file = Path("data/pdf_index.csv")

rows = []

for pdf_path in papers_folder.rglob("*.pdf"):
    try:
        reader = PdfReader(str(pdf_path))
        metadata = reader.metadata

        title = metadata.title if metadata and metadata.title else ""

        rows.append({
            "filename": pdf_path.name,
            "path": str(pdf_path),
            "title": title,
            "pages": len(reader.pages),
        })

    except Exception as e:
        rows.append({
            "filename": pdf_path.name,
            "path": str(pdf_path),
            "title": "",
            "pages": "",
            "error": str(e),
        })

output_file.parent.mkdir(exist_ok=True)

with open(output_file, "w", newline="", encoding="utf-8") as f:
    fieldnames = ["filename", "path", "title", "pages"]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"Indexed {len(rows)} PDFs.")
print(f"Saved to {output_file}")