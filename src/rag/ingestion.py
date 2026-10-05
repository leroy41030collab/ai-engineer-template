from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def load_documents(path: str = "data/raw") -> list[Document]:
    documents = []

    for file_path in Path(path).glob("*.txt"):
        text = file_path.read_text(encoding="utf-8")
        documents.append(Document(page_content=text, metadata={"source": str(file_path)}))

    return documents


def split_documents(documents: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )
    return splitter.split_documents(documents)
