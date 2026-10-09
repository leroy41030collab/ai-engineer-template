from langchain_core.documents import Document

from src.rag.ingestion import load_documents, split_documents
from src.rag.rag import answer_question
from src.rag.vector_store import (
    get_or_build_vector_store,
    search_with_scores,
)


def test_load_documents_reads_txt_and_metadata(tmp_path):
    file_path = tmp_path / "knowledge.txt"
    file_path.write_text(
        "Il portale raccoglie eventi e servizi.",
        encoding="utf-8",
    )

    documents = load_documents(str(tmp_path))

    assert len(documents) == 1
    assert "eventi e servizi" in documents[0].page_content
    assert documents[0].metadata["filename"] == "knowledge.txt"
    assert documents[0].metadata["file_type"] == "txt"


def test_split_documents_preserves_metadata():
    document = Document(
        page_content="Informazioni sul portale. " * 100,
        metadata={"source": "example.txt", "filename": "example.txt"},
    )

    chunks = split_documents([document])

    assert len(chunks) > 1
    assert all(chunk.metadata["source"] == "example.txt" for chunk in chunks)
    assert all("chunk_index" in chunk.metadata for chunk in chunks)


def test_search_with_scores_returns_documents_and_scores():
    expected_document = Document(
        page_content="Il portale raccoglie eventi.",
        metadata={"filename": "example.txt"},
    )

    class FakeVectorStore:
        def similarity_search_with_score(self, query, k):
            assert query == "eventi"
            assert k == 2
            return [(expected_document, 0.25)]

    results = search_with_scores(FakeVectorStore(), "eventi", k=2)

    assert len(results) == 1
    document, score = results[0]
    assert document.page_content == "Il portale raccoglie eventi."
    assert score == 0.25


def test_force_rebuild_rebuilds_vector_store(monkeypatch):
    expected_store = object()
    build_calls = []

    def fake_build(documents):
        build_calls.append(documents)
        return expected_store

    def fake_load():
        return object()

    monkeypatch.setattr(
        "src.rag.vector_store.build_vector_store",
        fake_build,
    )
    monkeypatch.setattr(
        "src.rag.vector_store.load_vector_store",
        fake_load,
    )

    documents = ["documento di prova"]

    result = get_or_build_vector_store(
        documents,
        force_rebuild=True,
    )

    assert result is expected_store
    assert build_calls == [documents]


def test_answer_question_ignores_irrelevant_documents(monkeypatch):
    from types import SimpleNamespace

    relevant = Document(
        page_content="Il portale pubblica eventi.",
        metadata={"filename": "eventi.txt"},
    )
    irrelevant = Document(
        page_content="Informazioni non pertinenti.",
        metadata={"filename": "altro.txt"},
    )

    monkeypatch.setattr(
        "src.rag.rag.settings.rag_max_distance",
        1.0,
    )
    monkeypatch.setattr(
        "src.rag.rag.retrieve_with_scores",
        lambda question, vector_store: [
            (relevant, 0.5),
            (irrelevant, 1.5),
        ],
    )
    monkeypatch.setattr(
        "src.rag.rag.get_chat_model",
        lambda: SimpleNamespace(
            invoke=lambda messages: SimpleNamespace(
                content="Il portale pubblica eventi."
            )
        ),
    )

    result = answer_question("Quali contenuti pubblica?", object())

    assert result["answer"] == "Il portale pubblica eventi."
    assert len(result["sources"]) == 1
    assert result["sources"][0]["filename"] == "eventi.txt"


def test_answer_question_does_not_call_llm_without_relevant_context(
    monkeypatch,
):
    irrelevant = Document(
        page_content="Informazioni non pertinenti.",
        metadata={"filename": "altro.txt"},
    )

    monkeypatch.setattr(
        "src.rag.rag.settings.rag_max_distance",
        1.0,
    )
    monkeypatch.setattr(
        "src.rag.rag.retrieve_with_scores",
        lambda question, vector_store: [(irrelevant, 1.5)],
    )

    def unexpected_llm_call():
        raise AssertionError("Il modello non dovrebbe essere interrogato.")

    monkeypatch.setattr(
        "src.rag.rag.get_chat_model",
        unexpected_llm_call,
    )

    result = answer_question("Dove sono i parcheggi?", object())

    assert "informazioni sufficienti" in result["answer"]
    assert result["sources"] == []