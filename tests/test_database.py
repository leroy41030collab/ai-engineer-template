from src.database.repository import create_document, get_document


def test_create_and_get_document():
    document = create_document(
        source="test",
        content="Contenuto di test.",
    )

    loaded = get_document(document.id)

    assert loaded is not None
    assert loaded.source == "test"
    assert loaded.content == "Contenuto di test."