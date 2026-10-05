from langchain_openai import ChatOpenAI

from src.config.settings import settings


def get_chat_model():
    return ChatOpenAI(
        model=settings.openai_model,
        api_key=settings.openai_api_key,
        temperature=0,
    )
