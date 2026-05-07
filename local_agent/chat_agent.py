from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import Tool
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from local_agent.config import Settings


class AgentState(TypedDict, total=False):
    messages: list[Any]
    plan: str


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

    llm_with_tools = llm.bind_tools([search_tool])

    def planner(state: AgentState) -> AgentState:
        # Keep the plan short; it will be fed into later nodes as context.
        planner_msg = llm.invoke(
            [
                SystemMessage(
                    content=(
                        "You are a planner for a local-document QA assistant.\n"
                        "Write a short plan (3-6 bullets) for how to answer the user.\n"
                        "If local docs are likely relevant, include a step to search them.\n"
                        "Do not answer the question yet; output only the plan."
                    )
                ),
                *state.get("messages", []),
            ]
        )
        plan_text = (getattr(planner_msg, "content", "") or "").strip()
        return {"plan": plan_text}

    def executor(state: AgentState) -> AgentState:
        plan = state.get("plan", "").strip()
        exec_system = (
            "You are the executor. Use tools when needed to gather facts.\n"
            "When you are ready to write the final answer, do NOT answer yet; just stop calling tools.\n"
        )
        if plan:
            exec_system += f"\nPlan:\n{plan}\n"

        msg = llm_with_tools.invoke([SystemMessage(content=exec_system), *state.get("messages", [])])
        return {"messages": [msg]}

    def synthesizer(state: AgentState) -> AgentState:
        plan = state.get("plan", "").strip()
        synth_system = (
            "You are the synthesizer. Produce the final answer to the user.\n"
            "Use tool results already present in the conversation. Be concise and accurate.\n"
        )
        if plan:
            synth_system += f"\n(Plan used)\n{plan}\n"

        msg = llm.invoke([SystemMessage(content=synth_system), *state.get("messages", [])])
        return {"messages": [msg]}

    tools_node = ToolNode([search_tool])

    g = StateGraph(AgentState)
    g.add_node("planner", planner)
    g.add_node("executor", executor)
    g.add_node("tools", tools_node)
    g.add_node("synthesizer", synthesizer)

    g.add_edge(START, "planner")
    g.add_edge("planner", "executor")
    g.add_conditional_edges("executor", tools_condition, {"tools": "tools", "__end__": "synthesizer"})
    g.add_edge("tools", "executor")
    g.add_edge("synthesizer", END)

    graph = g.compile()
    return LocalGraphAgent(graph=graph)

