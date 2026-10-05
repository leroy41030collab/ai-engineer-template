from pathlib import Path

from langchain_community.vectorstores import FAISS

from src.rag.embeddings import get_embeddings


VECTOR_STORE_PATH = Path("data/processed/faiss")


def build_vector_store(documents):
    vector_store = FAISS.from_documents(documents, get_embeddings())
    VECTOR_STORE_PATH.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(VECTOR_STORE_PATH))
    return vector_store


def load_vector_store():
    if not VECTOR_STORE_PATH.exists():
        return None

    return FAISS.load_local(
        str(VECTOR_STORE_PATH),
        get_embeddings(),
        allow_dangerous_deserialization=True,
    )


def get_or_build_vector_store(documents):
    vector_store = load_vector_store()

    if vector_store is not None:
        return vector_store

    return build_vector_store(documents)


def search_vector_store(vector_store, query: str, k: int = 4):
    return vector_store.similarity_search(query, k=k)
