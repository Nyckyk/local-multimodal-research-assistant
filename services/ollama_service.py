import ollama

from settings import OLLAMA_MODEL


def generate_answer(
    question: str,
    context: str,
    conversation_history: list[dict],
) -> str:
    prompt = f"""
You are a cautious research assistant answering questions about research papers.

Use only the supplied evidence. The evidence may include:
- extracted PDF text; and
- a labelled visual analysis produced from a selected PDF page.

EVIDENCE RULES:
1. Treat labelled visual analysis as valid evidence about figures, diagrams,
   tables, labels, colours and spatial layout.
2. For questions about a figure, table, diagram or page layout, prefer clear
   visual evidence over incomplete extracted text.
3. Do not reject a clear visual result merely because the same classification
   is not written as a sentence in the extracted text.
4. When text and visual evidence genuinely conflict, state the conflict.
5. Do not use outside knowledge or invent details.
6. Cite only source names and pages present in the supplied evidence.
7. Answer directly and concisely.
8. If neither source of evidence answers the question, say the evidence is
   insufficient.
9. Mention colours, icons, arrows or other visible features only when the
   supplied visual evidence explicitly supports them.
10. When evidence is marked DOCUMENT SUMMARY MODE, follow its document-type
    framing. Describe reviews as reviews or frameworks, not as original
    experimental studies, unless the evidence explicitly reports experiments.
11. When VALIDATED FRAMEWORK GROUNDING is present, use its category membership
    exactly. Do not promote mechanisms discussed inside a section into the
    canonical named-item list.

Supplied evidence:
{context}

Question:
{question}

Before answering, verify that the conclusion is supported by either explicit PDF
text or the labelled visual analysis.
"""

    messages = [
        {
            "role": "system",
            "content": (
                "You are an evidence-grounded research assistant. A visual "
                "analysis of a selected PDF page is valid evidence for questions "
                "about figures, tables, diagrams and page layout."
            ),
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]

    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=messages,
        think=False,
        options={
            "temperature": 0,
        },
    )

    answer = response["message"].get("content", "").strip()

    if not answer:
        answer = response["message"].get("thinking", "").strip()

    if not answer:
        return "The model returned an empty response."

    return answer
