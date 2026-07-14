from sentence_transformers import CrossEncoder, SentenceTransformer

from settings import EMBEDDING_MODEL, RERANKER_MODEL


def load_embedder():
    return SentenceTransformer(
        EMBEDDING_MODEL
    )


def load_reranker():
    return CrossEncoder(
        RERANKER_MODEL
    ) 