from typing import Annotated
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from src.llm.tool_model import get_model_with_tools
from src.tools.portal import get_portal_status, search_portal_knowledge


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


def build_portal_agent():
    tools = [get_portal_status, search_portal_knowledge]
    model = get_model_with_tools()
    tool_node = ToolNode(tools, handle_tool_errors=True)

    def agent(state: AgentState):
        response = model.invoke(state["messages"])
        return {"messages": [response]}

    def should_continue(state: AgentState):
        last_message = state["messages"][-1]

        if last_message.tool_calls:
            return "tool"

        return END

    graph_builder = StateGraph(AgentState)

    graph_builder.add_node("agent", agent)
    graph_builder.add_node("tool", tool_node)

    graph_builder.add_edge(START, "agent")

    graph_builder.add_conditional_edges(
        "agent",
        should_continue,
        {"tool": "tool", END: END},
    )

    graph_builder.add_edge("tool", "agent")

    return graph_builder.compile()