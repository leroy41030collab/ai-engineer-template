from langchain_openai import OpenAIEmbeddings

from src.config.settings import settings


def get_embeddings():
    return OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=settings.openai_api_key,
    )
