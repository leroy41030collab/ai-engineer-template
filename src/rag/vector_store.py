import hashlib
import json
from pathlib import Path
import re

from rank_bm25 import BM25Okapi

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

def hybrid_search_with_scores(vector_store, query: str, k: int = 4):
    total_docs = getattr(
        getattr(vector_store, "index", None),
        "ntotal",
        k,
    )

    semantic_results = vector_store.similarity_search_with_score(
        query,
        k=max(k, total_docs),
    )

    if not semantic_results:
        return []

    documents = [document for document, _ in semantic_results]

    def tokenize(text: str) -> list[str]:
        return re.findall(r"\w+", text.lower())

    tokenized_documents = [
        tokenize(document.page_content)
        for document in documents
    ]
    query_tokens = tokenize(query)

    if not query_tokens:
        return semantic_results[:k]

    bm25 = BM25Okapi(tokenized_documents)
    lexical_scores = bm25.get_scores(query_tokens)

    # Conteggio dei termini della query presenti in ogni documento.
    # Aiuta quando BM25 assegna punteggi zero, ad esempio su corpus piccoli.
    query_terms = set(query_tokens)
    lexical_overlap = [
        len(query_terms.intersection(set(tokens)))
        for tokens in tokenized_documents
    ]

    semantic_ranks = {
        index: index + 1
        for index in range(len(semantic_results))
    }

    lexical_order = sorted(
        range(len(documents)),
        key=lambda index: (
            float(lexical_scores[index]),
            lexical_overlap[index],
        ),
        reverse=True,
    )

    relevant_lexical_results = [
        index
        for index in lexical_order
        if lexical_scores[index] > 0 or lexical_overlap[index] > 0
    ]

    lexical_ranks = {
        index: rank
        for rank, index in enumerate(
            relevant_lexical_results,
            start=1,
        )
    }

    candidates = set(range(min(k, len(documents))))
    candidates.update(relevant_lexical_results[:k])

    def reciprocal_rank(index: int) -> float:
        score = 1 / (60 + semantic_ranks[index])

        if index in lexical_ranks:
            score += 1 / (60 + lexical_ranks[index])

        return score

    ranked_candidates = sorted(
        candidates,
        key=reciprocal_rank,
        reverse=True,
    )

    return [
        semantic_results[index]
        for index in ranked_candidates[:k]
    ]