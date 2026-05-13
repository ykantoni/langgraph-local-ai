from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import re
from dataclasses import dataclass
from typing import Any, Iterable, TypedDict

from langchain_core.tools import BaseTool, Tool
from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph

from local_agent.config import Settings

logger = logging.getLogger(__name__)


def _run_tool_sync(tool: BaseTool, args: dict[str, Any] | str) -> str:
    """Invoke a LangChain tool synchronously regardless of sync/async impl.

    MCP-backed tools (``StructuredTool`` with only ``coroutine``) raise
    ``NotImplementedError`` from ``invoke``; in that case we drive the async
    path via ``asyncio.run``, isolating it to a worker thread if we happen to
    be inside a running event loop.
    """
    try:
        return str(tool.invoke(args))
    except NotImplementedError:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return str(asyncio.run(tool.ainvoke(args)))
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            return str(ex.submit(asyncio.run, tool.ainvoke(args)).result())


def _find_tool(tools: Iterable[BaseTool], name: str) -> BaseTool | None:
    target = (name or "").strip()
    if not target:
        return None
    for t in tools:
        if t.name == target:
            return t
    # Best-effort match by suffix to tolerate "<server>_search_docs" prefixing.
    for t in tools:
        if t.name.endswith("_" + target) or t.name.endswith(target):
            return t
    return None


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


def create_chat_agent(
    vectorstore,
    settings: Settings,
    tools: list[BaseTool] | None = None,
    tools_supported: bool = False,
) -> LocalGraphAgent:
    """Build the planner / retrieve / executor / synthesizer / critic graph.

    ``tools`` are external LangChain tools (typically loaded from MCP servers).
    When ``tools_supported`` is true AND ``tools`` is non-empty, the ``retrieve``
    node asks the LLM to choose and invoke one of those tools (tool-calling
    path). Otherwise it falls back to the original behaviour of calling FAISS
    similarity search on ``vectorstore`` directly (direct-retrieval path).
    """

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

    # Always create the in-process local-search Tool so the fallback path (and
    # backwards-compatible tests) keep working even when MCP is disabled.
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

    use_tool_calling = bool(tools_supported and tools)
    llm_with_tools = None
    if use_tool_calling:
        try:
            llm_with_tools = llm.bind_tools(list(tools))
            logger.info(
                "chat_agent: tool-calling enabled (tools=%s)",
                [t.name for t in tools],
            )
        except Exception as e:
            logger.warning(
                "chat_agent: bind_tools failed (%s); falling back to direct retrieval",
                e,
            )
            use_tool_calling = False
            llm_with_tools = None
    else:
        logger.info(
            "chat_agent: tool-calling disabled (tools_supported=%s tools_count=%d) — using direct retrieval",
            tools_supported,
            len(tools or []),
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

    def retrieve_direct(state: AgentState) -> AgentState:
        """Run local document search once after planning (or after critic revision)."""
        q = (state.get("retrieval_query") or "").strip() or (state.get("question", "") or "")
        logger.info("retrieve(direct): start (query_chars=%d)", len(q))
        docs = search_docs(q)
        logger.info("retrieve(direct): done (docs_chars=%d)", len(docs or ""))
        return {
            **state,
            "intermediate_results": (state.get("intermediate_results") or []) + [docs],
        }

    def retrieve_via_tools(state: AgentState) -> AgentState:
        """Ask the tool-capable LLM to call one of the available MCP tools.

        On any failure (no tool call returned, unknown tool name, runtime
        error from the tool), we fall back to in-process FAISS search so the
        critic loop still has evidence to work with.
        """
        q = (state.get("retrieval_query") or "").strip() or (state.get("question", "") or "")
        logger.info("retrieve(tools): start (query_chars=%d)", len(q))

        tool_names = ", ".join(t.name for t in (tools or []))
        prompt = (
            "You have access to the following tools that can search local "
            f"project documents: {tool_names}.\n\n"
            "Decide which single tool to call and invoke it with the most "
            "appropriate search query for the user's question. "
            "Do not answer directly; only call the tool.\n\n"
            f"User question / refined query:\n{q}"
        )

        outputs: list[str] = []
        try:
            assert llm_with_tools is not None  # guarded by use_tool_calling
            msg = llm_with_tools.invoke(prompt)
            tool_calls = list(getattr(msg, "tool_calls", None) or [])
            logger.info("retrieve(tools): llm produced %d tool_call(s)", len(tool_calls))

            for tc in tool_calls:
                name = (tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", "")) or ""
                args = (tc.get("args") if isinstance(tc, dict) else getattr(tc, "args", {})) or {}
                if isinstance(args, str):
                    args = {"query": args}
                if "query" not in args and q:
                    args.setdefault("query", q)
                target = _find_tool(tools or [], name)
                if target is None:
                    logger.warning("retrieve(tools): unknown tool name %r", name)
                    continue
                try:
                    out = _run_tool_sync(target, args)
                    if out:
                        outputs.append(out)
                except Exception as e:
                    logger.warning("retrieve(tools): tool %r raised: %s", target.name, e)
        except Exception as e:
            logger.warning("retrieve(tools): llm tool-call failed (%s); using direct search", e)

        if not outputs:
            logger.info("retrieve(tools): no tool output, falling back to direct search")
            outputs.append(search_docs(q))

        logger.info(
            "retrieve(tools): done (chunks=%d total_chars=%d)",
            len(outputs),
            sum(len(o or "") for o in outputs),
        )
        return {
            **state,
            "intermediate_results": (state.get("intermediate_results") or []) + outputs,
        }

    retrieve = retrieve_via_tools if use_tool_calling else retrieve_direct

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

