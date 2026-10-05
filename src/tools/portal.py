from langchain_core.tools import tool


@tool
def get_portal_status() -> str:
    """Restituisce lo stato attuale del portale."""
    return "Sorbara e Dintorni è online e operativo."
