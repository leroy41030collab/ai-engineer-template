from langchain_core.messages import HumanMessage

from src.agents.portal_agent import build_portal_agent


def main():
    agent = build_portal_agent()

    result = agent.invoke({
        "messages": [
            HumanMessage(content="Qual è lo stato del portale?")
        ]
    })

    print("\\n--- RISPOSTA ---")
    print(result["messages"][-1].content)


if __name__ == "__main__":
    main()
