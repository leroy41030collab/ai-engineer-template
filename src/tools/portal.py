
from langchain_core.tools import tool

from src.rag.rag import answer_question, build_rag


@tool
def get_portal_status() -> str:
    """Restituisce lo stato attuale del portale."""
    return "Sorbara e Dintorni è online e operativo."


@tool
def search_portal_knowledge(question: str) -> str:
    """Cerca nei documenti indicizzati del portale per rispondere a domande
    sui suoi contenuti, servizi e informazioni documentate.
    Usa questo strumento quando la risposta richiede informazioni dai documenti.
    """
    vector_store = build_rag()
    result = answer_question(question, vector_store)

    answer = result["answer"]
    sources = result.get("sources", [])

    if not sources:
        return answer

    source_lines = [
        f"- {source['filename']} (chunk {source['chunk_index']}, "
        f"score {source['score']})"
        for source in sources
    ]

    return f"{answer}\n\nFonti:\n" + "\n".join(source_lines)