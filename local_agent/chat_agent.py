from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import Tool
from langchain_ollama import ChatOllama
from langchain.agents import create_agent

from local_agent.config import Settings


@dataclass(frozen=True)
class LocalGraphAgent:
    """Small wrapper to keep the old AgentExecutor-like surface.

    - server calls: agent.invoke({"input": "..."})
    - CLI calls:    agent.run("...")
    """

    graph: Any

    def invoke(self, inputs: dict[str, Any], *, config: dict[str, Any] | None = None) -> dict[str, Any]:
        text = inputs.get("input")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Expected inputs={'input': <non-empty str>}")

        result = self.graph.invoke(
            {"messages": [HumanMessage(content=text)]},
            config=config,
        )
        return result

    def run(self, query: str) -> str:
        result = self.invoke({"input": query})
        messages = result.get("messages") or []
        for m in reversed(messages):
            if isinstance(m, AIMessage) and m.content:
                return str(m.content)
        return ""


def create_chat_agent(vectorstore, settings: Settings) -> LocalGraphAgent:
    def search_docs(query: str) -> str:
        results = vectorstore.similarity_search(query, k=3)
        return "\n\n".join([r.page_content for r in results])

    search_tool = Tool(
        name="LocalDocumentSearch",
        func=search_docs,
        description="Searches local documents for relevant information",
    )

    llm = ChatOllama(
        model=settings.ollama_chat_model,
        temperature=0,
        base_url=settings.ollama_base_url,
        streaming=True,
    )

    graph = create_agent(llm, tools=[search_tool])
    return LocalGraphAgent(graph=graph)

