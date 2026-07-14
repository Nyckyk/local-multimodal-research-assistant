import chromadb
import ollama
from sentence_transformers import CrossEncoder, SentenceTransformer

DB_PATH = "data/chroma"
COLLECTION_NAME = "papers"
OLLAMA_MODEL = "qwen3.5:9b"

INITIAL_RESULTS = 20
FINAL_RESULTS = 5
MAX_HISTORY_MESSAGES = 6

client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_collection(COLLECTION_NAME)

embedder = SentenceTransformer("all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

conversation_history = []

print("Local Research Assistant")
print("Commands: exit, clear\n")

while True:
    question = input("You: ").strip()

    if not question:
        continue

    if question.lower() in {"exit", "quit"}:
        break

    if question.lower() == "clear":
        conversation_history.clear()
        print("\nConversation memory cleared.\n")
        continue

    # Improve retrieval for follow-up questions such as:
    # "Explain number 6" or "What evidence supports that?"
    if conversation_history:
        previous_user_question = next(
            (
                message["content"]
                for message in reversed(conversation_history)
                if message["role"] == "user"
            ),
            "",
        )

        retrieval_query = (
            f"Previous question: {previous_user_question}\n"
            f"Current follow-up question: {question}"
        )
    else:
        retrieval_query = question

    query_embedding = embedder.encode(
        retrieval_query,
        normalize_embeddings=True,
    ).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=INITIAL_RESULTS,
        include=["documents", "metadatas", "distances"],
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    if not documents:
        print("\nAssistant:\nNo relevant information was found.\n")
        continue

    pairs = [
        [retrieval_query, document]
        for document in documents
    ]

    reranker_scores = reranker.predict(pairs)

    ranked_results = sorted(
        zip(reranker_scores, documents, metadatas),
        key=lambda item: float(item[0]),
        reverse=True,
    )

    selected_results = []
    seen_pages = set()

    for score, document, metadata in ranked_results:
        page_key = (
            metadata.get("source", "Unknown source"),
            metadata.get("page", "Unknown page"),
        )

        if page_key in seen_pages:
            continue

        selected_results.append((score, document, metadata))
        seen_pages.add(page_key)

        if len(selected_results) >= FINAL_RESULTS:
            break

    context_parts = []

    for _, document, metadata in selected_results:
        source = metadata.get("source", "Unknown source")
        page = metadata.get("page", "Unknown page")

        context_parts.append(
            f"Source: {source}, page {page}\n{document}"
        )

    context = "\n\n".join(context_parts)

    current_prompt = f"""
Use only the supplied research-paper context to answer the question.

Rules:
- Answer the current question directly.
- Use previous conversation messages only to understand follow-up references.
- Do not invent information missing from the supplied context.
- Cite relevant source pages in the answer.
- If the answer is absent, say it could not be found.

Research-paper context:
{context}

Current question:
{question}
"""

    messages = [
        {
            "role": "system",
            "content": (
                "You are a careful research assistant. "
                "Your answers must be grounded in the supplied papers."
            ),
        }
    ]

    messages.extend(conversation_history[-MAX_HISTORY_MESSAGES:])

    messages.append(
        {
            "role": "user",
            "content": current_prompt,
        }
    )

    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=messages,
        think=False,
    )

    answer = response["message"].get("content", "").strip()

    if not answer:
        answer = response["message"].get("thinking", "").strip()

    if not answer:
        answer = "The model returned an empty response."

    print("\nAssistant:\n")
    print(answer)

    print("\nTop retrieved sources:")

    for score, _, metadata in selected_results:
        source = metadata.get("source", "Unknown source")
        page = metadata.get("page", "Unknown page")

        print(
            f"- {source}, page {page} "
            f"(score: {float(score):.3f})"
        )

    print()

    conversation_history.append(
        {
            "role": "user",
            "content": question,
        }
    )

    conversation_history.append(
        {
            "role": "assistant",
            "content": answer,
        }
    ) 