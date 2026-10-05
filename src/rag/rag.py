from langchain_core.messages import HumanMessage, SystemMessage

from src.llm.models import get_chat_model
from src.rag.ingestion import load_documents, split_documents
from src.rag.vector_store import get_or_build_vector_store, search_vector_store


def build_rag():
    documents = load_documents()
    chunks = split_documents(documents)
    return get_or_build_vector_store(chunks)


def answer_question(question: str, vector_store) -> str:
    documents = search_vector_store(vector_store, question)

    context = "\\n\\n".join(
        document.page_content for document in documents
    )

    messages = [
        SystemMessage(
            content=(
                "Rispondi usando esclusivamente il contesto fornito. "
                "Se il contesto non contiene informazioni sufficienti, "
                "dichiara di non avere informazioni sufficienti.\\n\\n"
                f"CONTESTO:\\n{context}"
            )
        ),
        HumanMessage(content=question),
    ]

    response = get_chat_model().invoke(messages)
    return response.content
