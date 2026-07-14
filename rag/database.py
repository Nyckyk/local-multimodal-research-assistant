import chromadb

from settings import COLLECTION_NAME, DB_PATH


def get_collection():
    client = chromadb.PersistentClient(
        path=str(DB_PATH)
    )

    return client.get_or_create_collection(
        name=COLLECTION_NAME
    ) 