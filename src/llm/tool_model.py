from src.llm.models import get_chat_model
from src.tools.portal import get_portal_status, search_portal_knowledge


def get_model_with_tools():
    model = get_chat_model()
    return model.bind_tools([
        get_portal_status,
        search_portal_knowledge,
    ])
