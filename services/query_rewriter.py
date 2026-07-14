import ollama
import re

from settings import OLLAMA_MODEL


REFERENCE_WORDS = {
    "it", "its", "that", "this", "these", "those", "they", "them",
    "their", "former", "latter", "previous", "above", "below", "there",
}
QUESTION_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "does", "do",
    "for", "from", "how", "in", "is", "of", "on", "the", "to", "what",
    "when", "where", "which", "who", "why", "with",
}


def _tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9][A-Za-z0-9_.+\-/]*", text)


def _needs_context_resolution(question: str) -> bool:
    words = {token.lower() for token in _tokens(question)}
    return bool(words.intersection(REFERENCE_WORDS))


def _preserves_original_terms(question: str, rewritten: str) -> bool:
    rewritten_words = {token.lower() for token in _tokens(rewritten)}
    protected = {
        token.lower()
        for token in _tokens(question)
        if token.lower() not in REFERENCE_WORDS | QUESTION_WORDS
    }
    return protected.issubset(rewritten_words)


def rewrite_question(
    question: str,
    conversation_history: list[dict],
) -> str:
    if not conversation_history or not _needs_context_resolution(question):
        return question

    history_text = "\n".join(
        f"{message['role']}: {message['content']}"
        for message in conversation_history[-4:]
    )

    prompt = f"""
Rewrite the user's latest question as one complete standalone research query.

Use the conversation only to resolve vague references such as:
- it
- that
- number 6
- the previous mechanism
- the second paper

Rules:
- Preserve the user's actual intent.
- Preserve every scientific and technical term verbatim. Never replace a term
  with a broader synonym or explanatory paraphrase.
- Preserve figure numbers, table numbers, protein names, gene names,
  abbreviations, hyphenation and domain-specific terminology exactly.
- Rewrite only the vague reference needed to make the question standalone.
- Do not answer the question.
- Do not add page numbers, figure numbers, citations, authors or claims.
- Do not assume where the answer appears.
- Return only the rewritten query.

Conversation:
{history_text}

Latest question:
{question}
"""

    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        think=False,
    )

    rewritten = response["message"].get("content", "").strip()

    if not rewritten or not _preserves_original_terms(question, rewritten):
        return question

    return rewritten
