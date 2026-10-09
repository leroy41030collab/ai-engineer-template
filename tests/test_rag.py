from langchain_core.documents import Document

from src.rag.ingestion import load_documents, split_documents
from src.rag.vector_store import search_with_scores


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