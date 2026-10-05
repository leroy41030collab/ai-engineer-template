from src.tools.portal import get_portal_status


def test_get_portal_status():
    result = get_portal_status.invoke({})

    assert isinstance(result, str)
    assert "operativo" in result.lower()
