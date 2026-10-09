from langchain_core.documents import Document

from src.rag.ingestion import load_documents, split_documents
from src.rag.rag import answer_question
from src.rag.vector_store import hybrid_search_with_scores
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


def test_force_rebuild_rebuilds_vector_store(monkeypatch, tmp_path):
    import src.rag.vector_store as vector_store_module

    expected_store = object()
    build_calls = []

    def fake_build(documents):
        build_calls.append(documents)
        return expected_store

    def fake_load():
        return object()

    monkeypatch.setattr(
        vector_store_module,
        "VECTOR_STORE_PATH",
        tmp_path,
    )
    monkeypatch.setattr(
        vector_store_module,
        "MANIFEST_PATH",
        tmp_path / "manifest.json",
    )
    monkeypatch.setattr(
        vector_store_module,
        "build_vector_store",
        fake_build,
    )
    monkeypatch.setattr(
        vector_store_module,
        "load_vector_store",
        fake_load,
    )

    documents = [
        Document(
            page_content="Documento di prova",
            metadata={"filename": "prova.txt"},
        )
    ]

    result = get_or_build_vector_store(
        documents,
        force_rebuild=True,
    )

    assert result is expected_store
    assert build_calls == [documents]
    assert (tmp_path / "manifest.json").exists()

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


def test_reuses_vector_store_when_documents_are_unchanged(
    monkeypatch,
    tmp_path,
):
    import json
    import src.rag.vector_store as module

    documents = [
        Document(
            page_content="Il portale pubblica eventi.",
            metadata={"filename": "eventi.txt"},
        )
    ]

    existing_store = object()
    build_calls = []

    monkeypatch.setattr(module, "VECTOR_STORE_PATH", tmp_path)
    monkeypatch.setattr(
        module,
        "MANIFEST_PATH",
        tmp_path / "manifest.json",
    )
    monkeypatch.setattr(
        module,
        "load_vector_store",
        lambda: existing_store,
    )
    monkeypatch.setattr(
        module,
        "build_vector_store",
        lambda docs: build_calls.append(docs),
    )

    fingerprint = module._documents_fingerprint(documents)

    (tmp_path / "manifest.json").write_text(
        json.dumps({"fingerprint": fingerprint}),
        encoding="utf-8",
    )

    result = module.get_or_build_vector_store(documents)

    assert result is existing_store
    assert build_calls == []


def test_rebuilds_vector_store_when_documents_change(
    monkeypatch,
    tmp_path,
):
    import json
    import src.rag.vector_store as module

    old_documents = [
        Document(
            page_content="Il portale pubblica eventi.",
            metadata={"filename": "eventi.txt"},
        )
    ]

    new_documents = [
        Document(
            page_content="Il portale pubblica eventi e servizi.",
            metadata={"filename": "eventi.txt"},
        )
    ]

    expected_store = object()
    build_calls = []

    def fake_build(documents):
        build_calls.append(documents)
        return expected_store

    monkeypatch.setattr(module, "VECTOR_STORE_PATH", tmp_path)
    monkeypatch.setattr(
        module,
        "MANIFEST_PATH",
        tmp_path / "manifest.json",
    )
    monkeypatch.setattr(
        module,
        "load_vector_store",
        lambda: object(),
    )
    monkeypatch.setattr(
        module,
        "build_vector_store",
        fake_build,
    )

    old_fingerprint = module._documents_fingerprint(old_documents)

    (tmp_path / "manifest.json").write_text(
        json.dumps({"fingerprint": old_fingerprint}),
        encoding="utf-8",
    )

    result = module.get_or_build_vector_store(new_documents)

    assert result is expected_store
    assert build_calls == [new_documents]

    updated_manifest = json.loads(
        (tmp_path / "manifest.json").read_text(encoding="utf-8")
    )
    assert updated_manifest["fingerprint"] == (
        module._documents_fingerprint(new_documents)
    )

def test_hybrid_search_combines_semantic_and_keyword_results():
    from langchain_core.documents import Document

    semantic_document = Document(
        page_content="Informazioni generali sul territorio.",
        metadata={"filename": "generale.txt"},
    )
    keyword_document = Document(
        page_content="Il servizio XZ-42 è disponibile a Sorbara.",
        metadata={"filename": "servizio.txt"},
    )

    class FakeVectorStore:
        index = type("FakeIndex", (), {"ntotal": 2})()

        def similarity_search_with_score(self, query, k):
            assert query == "XZ-42 Sorbara"
            assert k == 2
            return [
                (semantic_document, 0.2),
                (keyword_document, 0.4),
            ]

    results = hybrid_search_with_scores(
        FakeVectorStore(),
        "XZ-42 Sorbara",
        k=2,
    )

    assert len(results) == 2
    assert all(isinstance(score, float) for _, score in results)
    assert {doc.metadata["filename"] for doc, _ in results} == {
        "generale.txt",
        "servizio.txt",
    }
    assert results[0][0].metadata["filename"] == "servizio.txt"