from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypedDict

import re

from langchain_core.tools import Tool
from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph

import logging

from local_agent.config import Settings

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    question: str
    plan: list[str]
    current_step: int
    intermediate_results: list[str]
    final_answer: str
    retrieval_query: str
    critic_iters: int
    critic_verdict: str


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
                "final_answer": "",
                "retrieval_query": "",
                "critic_iters": 0,
                "critic_verdict": "",
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

    def planner(state: AgentState) -> AgentState:
        logger.info("planner: start (question_chars=%d)", len(state.get("question", "") or ""))
        prompt = f"""
            Given that context contains result of local documents search create a short 3 steps step-by-step plan to answer the following question:
            {state['question']}

            Use steps like:
            - analyze the question and context
            - summarize the information
            - combine the information

            Output exactly 3 steps in this EXACT format (including the ### lines):

            1) <step text>
            ###
            2) <step text>
            ###
            3) <step text>
            ###

            Rules:
            - The ONLY separator is a line containing exactly ### (three hash marks).
            - There MUST be a ### line after step 1, after step 2, and after step 3.
            - Do not add any extra text before 1) or after the last ###.
            Keep the plan short and concise.
            """
        plan_text = llm.invoke(prompt).content

        raw = (plan_text or "").strip()

        # Primary parse: split on ### separators (works when ### appears after each step).
        steps = [s.strip() for s in raw.split("###") if s.strip()]

        # Fallback: if the model only put ### at the end, recover steps from a numbered list.
        if len(steps) <= 1:
            # Capture "1) ...", "2) ...", "3) ..." blocks even if they wrap lines.
            matches = re.findall(r"(?ms)^\s*\d+\)\s*(.+?)(?=^\s*\d+\)\s*|\Z)", raw)
            recovered = [m.strip() for m in matches if m.strip()]
            if len(recovered) >= 2:
                steps = recovered[:3]

        logger.info("planner: done (steps=%d plan_chars=%d)", len(steps), len(plan_text or ""))

        return {
            **state,
            "plan": steps,
            "current_step": 0,
            "intermediate_results": [],
            "retrieval_query": "",
        }

    def retrieve(state: AgentState) -> AgentState:
        """Always run local document search once after planning (or after critic revision)."""
        q = (state.get("retrieval_query") or "").strip() or (state.get("question", "") or "")
        logger.info("retrieve: start (query_chars=%d)", len(q))
        docs = search_docs(q)
        logger.info("retrieve: done (docs_chars=%d)", len(docs or ""))
        return {
            **state,
            "intermediate_results": (state.get("intermediate_results") or []) + [docs],
        }

    def executor(state: AgentState) -> AgentState:
        step_idx = int(state.get("current_step", 0) or 0)
        plan_len = len(state.get("plan") or [])
        step = (state.get("plan") or [""])[step_idx] if step_idx < plan_len else ""
        prior = "\n\n".join(state.get("intermediate_results") or [])
        step = f"{step}. Keep the answer short and concise.\n\nPrior context:\n{prior}"

        logger.info("executor: start (step=%d/%d step_chars=%d)", step_idx + 1, plan_len, len(step or ""))

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
        Answer the question using the context below. Keep the anwser short and concise.

        Question:
        {state['question']}

        Context:
        {context}
        """

        answer = llm.invoke(prompt).content
        logger.info("synthesizer: done (answer_chars=%d)", len(answer or ""))

        return {
            **state,
            "final_answer": answer,
        }

    def _parse_critic_output(raw: str) -> tuple[str, str]:
        """Return (verdict 'PASS'|'REVISE', refined_search_query or '')."""
        t = (raw or "").strip()
        if not t:
            return "PASS", ""
        rq_m = re.search(r"(?im)^REFINED_QUERY:\s*(.+)$", t)
        refined = rq_m.group(1).strip() if rq_m else ""
        if re.search(r"(?im)^\s*PASS\s*$", t) or re.match(r"(?is)^\s*PASS\s*(\n|$)", t):
            return "PASS", refined
        if re.search(r"(?im)^\s*REVISE\s*:", t) or "REVISE:" in t.upper():
            return "REVISE", refined
        return "PASS", refined

    def critic(state: AgentState) -> AgentState:
        iters = int(state.get("critic_iters", 0) or 0)
        if iters >= 2:
            logger.info("critic: hard-stop (max revisions reached)")
            return {**state, "critic_verdict": "PASS"}

        answer = state.get("final_answer", "") or ""
        question = state.get("question", "") or ""
        context = "\n\n".join(state.get("intermediate_results") or [])

        logger.info(
            "critic: start (iters=%d answer_chars=%d)",
            iters,
            len(answer),
        )

        prompt = f"""You are a strict critic for the assistant's final answer.

Question:
{question}

Retrieved / intermediate context (may be partial):
{context[:8000]}

Final answer:
{answer}

If the answer is adequate (grounded, complete enough for the question, clear), respond with exactly:
PASS

If the answer is weak (vague, missing key facts from context, or off-topic), respond with:
REVISE: <one short paragraph: what is wrong and what to fix>

Then on a new line:
REFINED_QUERY: <a single line search query for local document search to gather better evidence>

Do not include anything else outside this format."""

        raw = llm.invoke(prompt).content
        verdict, refined = _parse_critic_output(str(raw or ""))
        logger.info(
            "critic: done (verdict=%s refined_chars=%d)",
            verdict,
            len(refined),
        )

        if verdict == "PASS":
            return {**state, "critic_verdict": "PASS"}

        rq = refined.strip() or question
        return {
            **state,
            "critic_verdict": "REVISE",
            "critic_iters": iters + 1,
            "retrieval_query": rq,
            "intermediate_results": [],
            "current_step": 0,
        }

    def route_after_critic(state: AgentState) -> str:
        v = (state.get("critic_verdict") or "").strip()
        if v == "PASS":
            return END
        return "retrieve"

    g = StateGraph(AgentState)

    g.add_node("planner", planner)
    g.add_node("retrieve", retrieve)
    g.add_node("executor", executor)
    g.add_node("synthesizer", synthesizer)
    g.add_node("critic", critic)

    g.set_entry_point("planner")

    g.add_edge("planner", "retrieve")
    g.add_edge("retrieve", "executor")
    g.add_conditional_edges("executor", should_continue)
    g.add_edge("synthesizer", "critic")
    g.add_conditional_edges(
        "critic",
        route_after_critic,
        {END: END, "retrieve": "retrieve"},
    )

    graph = g.compile()
    return LocalGraphAgent(graph=graph)

