from sqlalchemy import select

from src.database.connection import engine
from src.database.models import Document


def create_document(source: str, content: str) -> Document:
    with engine.begin() as connection:
        result = connection.execute(
            Document.__table__.insert().returning(Document.id),
            {
                "source": source,
                "content": content,
            },
        )

        document_id = result.scalar_one()

    return Document(
        id=document_id,
        source=source,
        content=content,
    )


def get_document(document_id: int) -> Document | None:
    with engine.connect() as connection:
        result = connection.execute(
            select(Document).where(Document.id == document_id)
        )

        row = result.one_or_none()

    if row is None:
        return None

    return Document(
        id=row.id,
        source=row.source,
        content=row.content,
    )