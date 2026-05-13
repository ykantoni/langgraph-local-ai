"""Load settings, vector store, and chat agent (shared by CLI and HTTP server)."""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import logging
import urllib.error
import urllib.request
from typing import Any, Coroutine

from local_agent.chat_agent import create_chat_agent
from local_agent.config import Settings, load_settings
from local_agent.vectorstore import SentenceTransformerEmbeddings, get_or_create_vectorstore


logger = logging.getLogger(__name__)


def _run_coro_sync(coro: Coroutine[Any, Any, Any]) -> Any:
    """Run an async coroutine to completion from a sync caller.

    - Outside any event loop (CLI path): drive via ``asyncio.run`` directly.
    - Inside a running event loop (FastAPI lifespan): isolate to a worker
      thread with its own loop so ``asyncio.run()`` does not collide with
      the caller's loop.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(asyncio.run, coro).result()


def setup_logging(log_level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )


def model_supports_tools(base_url: str, model: str, *, timeout: float = 5.0) -> bool:
    """Probe Ollama ``/api/show`` to see if the chat model advertises tool support.

    Returns ``False`` on any error (network, JSON, missing capabilities) so the
    agent falls back to the direct-retrieval path safely.
    """
    if not base_url or not model:
        return False
    try:
        req = urllib.request.Request(
            f"{base_url.rstrip('/')}/api/show",
            data=json.dumps({"name": model}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, ValueError, OSError) as e:
        logger.warning("model_supports_tools: probe failed for %s (%s)", model, e)
        return False

    caps = payload.get("capabilities") or []
    normalized = {str(c).lower() for c in caps}
    supported = "tools" in normalized
    logger.info(
        "model_supports_tools: model=%s capabilities=%s tools_supported=%s",
        model,
        sorted(normalized),
        supported,
    )
    return supported


def load_mcp_tools(settings: Settings) -> list:
    """Load LangChain tools from MCP servers declared in ``settings``.

    Returns an empty list when MCP is disabled, no servers are configured, or
    on any startup error (logged). The agent then runs in fallback mode.
    """
    if not settings.mcp_enabled:
        return []
    raw = (settings.mcp_servers_json or "").strip()
    if not raw:
        logger.warning("load_mcp_tools: MCP_ENABLED=1 but MCP_SERVERS_JSON is empty")
        return []
    try:
        connections = json.loads(raw)
    except json.JSONDecodeError as e:
        logger.error("load_mcp_tools: invalid MCP_SERVERS_JSON: %s", e)
        return []
    if not isinstance(connections, dict) or not connections:
        logger.warning("load_mcp_tools: MCP_SERVERS_JSON must be a non-empty object")
        return []

    try:
        from langchain_mcp_adapters.client import MultiServerMCPClient
    except ImportError as e:
        logger.error(
            "load_mcp_tools: required package not importable (%s). "
            "Run `pip install -r requirements.txt` inside the active venv "
            "(sys.executable=%s).",
            e,
            __import__("sys").executable,
        )
        return []

    async def _get_tools_async():
        # Construct the client inside the coroutine so all per-call state
        # lives on the loop that actually awaits it (avoids cross-thread
        # state issues when we hop to a worker thread under FastAPI's
        # lifespan loop).
        client = MultiServerMCPClient(connections)
        return await client.get_tools()

    try:
        tools = _run_coro_sync(_get_tools_async())
        logger.info(
            "load_mcp_tools: loaded %d tool(s) from %d MCP server(s): %s",
            len(tools),
            len(connections),
            [getattr(t, "name", "?") for t in tools],
        )
        return tools
    except Exception as e:
        logger.error("load_mcp_tools: failed to fetch tools (%s)", e)
        return []


def load_chat_agent():
    """Build embeddings, FAISS store, MCP tools, and LangGraph agent.

    Call once at process startup.
    """
    settings = load_settings()
    setup_logging(settings.log_level)

    embeddings = SentenceTransformerEmbeddings(
        model_name=settings.st_embed_model,
        encode_batch_size=settings.st_encode_batch,
        device=settings.st_device,
    )
    vectorstore = get_or_create_vectorstore(settings, embeddings)

    tools = load_mcp_tools(settings)
    tools_supported = bool(tools) and model_supports_tools(
        settings.ollama_base_url, settings.ollama_chat_model
    )

    agent = create_chat_agent(
        vectorstore,
        settings,
        tools=tools,
        tools_supported=tools_supported,
    )
    return agent, settings
