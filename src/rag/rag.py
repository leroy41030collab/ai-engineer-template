from langchain_core.messages import HumanMessage, SystemMessage

from src.llm.models import get_chat_model
from src.rag.ingestion import load_documents, split_documents
from src.rag.vector_store import (
    get_or_build_vector_store,
    search_vector_store,
    search_with_scores,
)


def build_rag():
    documents = load_documents()
    chunks = split_documents(documents)
    return get_or_build_vector_store(chunks)


def retrieve_documents(question: str, vector_store, k: int = 4):
    return search_vector_store(vector_store, question, k=k)


def retrieve_with_scores(question: str, vector_store, k: int = 4):
    return search_with_scores(vector_store, question, k=k)


def answer_question(question: str, vector_store) -> dict:
    results = retrieve_with_scores(question, vector_store)

    context_parts = []
    sources = []

    for document, score in results:
        source = document.metadata.get("source", "Fonte sconosciuta")
        filename = document.metadata.get("filename", source)
        chunk_index = document.metadata.get("chunk_index")

        context_parts.append(document.page_content)

        sources.append(
            {
                "source": source,
                "filename": filename,
                "chunk_index": chunk_index,
                "score": round(float(score), 4),
            }
        )

    context = "\n\n".join(context_parts)

    messages = [
        SystemMessage(
            content=(
                "Rispondi usando esclusivamente il contesto fornito. "
                "Se il contesto non contiene informazioni sufficienti, "
                "dichiara di non avere informazioni sufficienti. "
                "Non inventare fonti o riferimenti.\n\n"
                f"CONTESTO:\n{context}"
            )
        ),
        HumanMessage(content=question),
    ]

    response = get_chat_model().invoke(messages)

    return {
        "answer": response.content,
        "sources": sources,
    }