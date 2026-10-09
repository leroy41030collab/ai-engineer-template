import hashlib
import json
from pathlib import Path

from langchain_community.vectorstores import FAISS

from src.rag.embeddings import get_embeddings


VECTOR_STORE_PATH = Path("data/processed/faiss")
MANIFEST_PATH = VECTOR_STORE_PATH / "manifest.json"


def _documents_fingerprint(documents) -> str:
    payload = [
        {
            "content": document.page_content,
            "metadata": document.metadata,
        }
        for document in documents
    ]

    serialized = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        default=str,
    )

    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _write_manifest(fingerprint: str) -> None:
    VECTOR_STORE_PATH.mkdir(parents=True, exist_ok=True)

    MANIFEST_PATH.write_text(
        json.dumps({"fingerprint": fingerprint}, indent=2),
        encoding="utf-8",
    )


def build_vector_store(documents):
    VECTOR_STORE_PATH.mkdir(parents=True, exist_ok=True)

    vector_store = FAISS.from_documents(
        documents,
        get_embeddings(),
    )

    vector_store.save_local(str(VECTOR_STORE_PATH))
    return vector_store


def load_vector_store():
    if not (VECTOR_STORE_PATH / "index.faiss").exists():
        return None

    if not (VECTOR_STORE_PATH / "index.pkl").exists():
        return None

    return FAISS.load_local(
        str(VECTOR_STORE_PATH),
        get_embeddings(),
        allow_dangerous_deserialization=True,
    )


def get_or_build_vector_store(documents, force_rebuild: bool = False):
    fingerprint = _documents_fingerprint(documents)

    if not force_rebuild and MANIFEST_PATH.exists():
        try:
            manifest = json.loads(
                MANIFEST_PATH.read_text(encoding="utf-8")
            )

            if manifest.get("fingerprint") == fingerprint:
                vector_store = load_vector_store()

                if vector_store is not None:
                    return vector_store
        except (OSError, ValueError, TypeError):
            pass

    vector_store = build_vector_store(documents)
    _write_manifest(fingerprint)

    return vector_store


def search_vector_store(vector_store, query: str, k: int = 4):
    return vector_store.similarity_search(query, k=k)


def search_with_scores(vector_store, query: str, k: int = 4):
    return vector_store.similarity_search_with_score(query, k=k)