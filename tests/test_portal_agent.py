from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.agents.portal_agent import build_portal_agent


def test_portal_agent_uses_tool_and_returns_answer():
    tool_call = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "get_portal_status",
                "args": {},
                "id": "call_test",
            }
        ],
    )

    final_answer = AIMessage(
        content="Il portale è online e operativo."
    )

    mock_model = type('MockModel', (), {
        'invoke': lambda self, messages: tool_call if len(messages) == 1 else final_answer
    })()

    with patch('src.agents.portal_agent.get_model_with_tools', return_value=mock_model):
        agent = build_portal_agent()

        result = agent.invoke({
            'messages': [HumanMessage(content='Qual è lo stato del portale?')]
        })

    assert result['messages'][-1].content == 'Il portale è online e operativo.'
