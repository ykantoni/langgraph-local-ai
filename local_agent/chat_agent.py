from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import Tool
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

import logging

from local_agent.config import Settings

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    question: str
    plan: list[str]
    current_step: int
    intermediate_results: list[str]
    final_answer: str


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

        result = self.graph.invoke({
                "question": text,
                "plan": [],
                "current_step": 0,
                "intermediate_results": [],
                "final_answer": ""
            },
            config=config,
        )
        return result

    def run(self, query: str) -> str:
        result = self.invoke({"input": query})
        messages = result.get("intermediate_results") or ""
        return messages


def create_chat_agent(vectorstore, settings: Settings) -> LocalGraphAgent:

    def search_docs(query: str) -> str:
        logger.info("search_docs: start (query_chars=%d)", len(query or ""))
        results = vectorstore.similarity_search(query, k=3)
        out = "\n\n".join([r.page_content for r in results])
        logger.info(
            "search_docs: done (results=%d out_chars=%d)",
            len(results or []),
            len(out),
        )
        return out

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
        logger.info("planner: start (question_chars=%d)", len(state.get("question", "") or ""))
        prompt = f"""
            Create a short 3 steps step-by-step plan to answer the following question:
            {state['question']}

            Use steps like:
            - always search local docs
            - summarize the information
            - combine the information

            Return as a numbered list. Keep the plan short and concise.
            """
        plan_text = llm.invoke(prompt).content

        steps = [s.strip() for s in plan_text.split("\n") if s.strip()]
        logger.info("planner: done (steps=%d plan_chars=%d)", len(steps), len(plan_text or ""))

        return {
            **state,
            "plan": steps,
            "current_step": 0,
            "intermediate_results": []
        }

    def executor(state: AgentState) -> AgentState:
        step_idx = int(state.get("current_step", 0) or 0)
        plan_len = len(state.get("plan") or [])
        step = (state.get("plan") or [""])[step_idx] if step_idx < plan_len else ""
        if step_idx > 0:
            prior = "\n\n".join(state.get("intermediate_results") or [])
            if prior:
                step = f"{step}\n\nPrior context:\n{prior}"

        logger.info("executor: start (step=%d/%d step_chars=%d)", step_idx + 1, plan_len, len(step or ""))

        if step_idx == 0:
            result = search_docs(state["question"])
        else:
            result = llm.invoke(step).content
        logger.info("executor: done (result_chars=%d)", len(result or ""))

        return {
            **state,
            "intermediate_results": state["intermediate_results"] + [result],
            "current_step": step_idx + 1,
        }

    def should_continue(state: AgentState):
        if state["current_step"] >= len(state["plan"]):
            return "synthesizer"
        return "executor"

    def synthesizer(state: AgentState) -> AgentState:
        context = "\n\n".join(state["intermediate_results"])
        logger.info(
            "synthesizer: start (question_chars=%d context_chars=%d steps=%d)",
            len(state.get("question", "") or ""),
            len(context),
            len(state.get("plan") or []),
        )

        prompt = f"""
        Answer the question using the context below.

        Question:
        {state['question']}

        Context:
        {context}
        """

        answer = llm.invoke(prompt).content
        logger.info("synthesizer: done (answer_chars=%d)", len(answer or ""))

        return {
            **state,
            "final_answer": answer
        }

    tools_node = ToolNode([search_tool])

    g = StateGraph(AgentState)

    g.add_node("planner", planner)
    g.add_node("executor", executor)
    g.add_node("synthesizer", synthesizer)

    g.set_entry_point("planner")

    g.add_edge("planner", "executor")
    g.add_conditional_edges("executor", should_continue)
    g.add_edge("synthesizer", END)

    graph = g.compile()
    return LocalGraphAgent(graph=graph)

