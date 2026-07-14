from pathlib import Path

from services.vision_service import analyse_pdf_page


PDF_PATH = Path("papers/hall marks of aging.pdf")
PAGE_NUMBER = 46
QUESTION = (
    "Which category contains mitochondrial dysfunction in Figure 6? "
    "Reply with the category and one brief piece of visible evidence."
)

try:
    answer = analyse_pdf_page(
        pdf_path=PDF_PATH,
        page_number=PAGE_NUMBER,
        question=QUESTION,
    )
    print("Vision answer:")
    print(answer)
except Exception as error:
    print(f"Vision test failed: {error}")
    raise
