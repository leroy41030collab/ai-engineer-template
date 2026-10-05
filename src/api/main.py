from fastapi import FastAPI
from pydantic import BaseModel

from src.agents.portal_agent import build_portal_agent
from langchain_core.messages import HumanMessage

app = FastAPI(
    title="AI Engineer Template",
    version="1.0.0",
)


class AskRequest(BaseModel):
    question: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask")
def ask(request: AskRequest):
    agent = build_portal_agent()

    result = agent.invoke({
        "messages": [
            HumanMessage(content=request.question)
        ]
    })

    return {
        "answer": result["messages"][-1].content
    }