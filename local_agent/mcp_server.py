"""MCP server exposing the local FAISS document search as an MCP tool.

Run as a stdio MCP server:

    python -m local_agent.mcp_server

Or over Streamable HTTP (default 127.0.0.1:8765/mcp; override with
``MCP_HTTP_HOST`` / ``MCP_HTTP_PORT`` / ``MCP_HTTP_PATH``):

    $env:MCP_TRANSPORT="streamable-http"
    python -m local_agent.mcp_server

The server registers a single tool, ``search_docs(query, k=3)``, which performs a
similarity search against the project's FAISS vector store (built from
``DOCS_DIR``). All settings come from environment variables via
``local_agent.config.load_settings`` so the agent process and this MCP server
can share configuration.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from mcp.server.fastmcp import FastMCP

from local_agent.config import Settings, load_settings
from local_agent.vectorstore import SentenceTransformerEmbeddings, get_or_create_vectorstore

logger = logging.getLogger(__name__)

_vectorstore: Any | None = None
_settings: Settings | None = None


def _get_vectorstore() -> Any:
    """Lazily build embeddings + FAISS store on first tool call."""
    global _vectorstore, _settings
    if _vectorstore is not None:
        return _vectorstore

    _settings = load_settings()
    logging.basicConfig(
        level=getattr(logging, _settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    logger.info("mcp_server: building embeddings and vector store")
    embeddings = SentenceTransformerEmbeddings(
        model_name=_settings.st_embed_model,
        encode_batch_size=_settings.st_encode_batch,
        device=_settings.st_device,
    )
    _vectorstore = get_or_create_vectorstore(_settings, embeddings)
    logger.info("mcp_server: vector store ready")
    return _vectorstore


def build_mcp_server(name: str = "local-docs") -> FastMCP:
    """Create the FastMCP instance with the ``search_docs`` tool registered.

    Factored out so tests can construct the server and inspect/register tools
    without spawning a subprocess. HTTP host/port/path come from environment
    variables (``MCP_HTTP_HOST`` / ``MCP_HTTP_PORT`` / ``MCP_HTTP_PATH``) so
    the same module can run as either a stdio child process or a stand-alone
    HTTP service without code changes.
    """
    mcp = FastMCP(
        name,
        host=os.environ.get("MCP_HTTP_HOST", "127.0.0.1"),
        port=int(os.environ.get("MCP_HTTP_PORT", "8765")),
        streamable_http_path=os.environ.get("MCP_HTTP_PATH", "/mcp"),
    )

    @mcp.tool()
    def search_docs(query: str, k: int = 3) -> str:
        """Search the local FAISS document index and return top-k passages.

        Args:
            query: Natural-language search query.
            k: Number of top passages to return (default 3).
        """
        logger.info("search_docs: start (query_chars=%d k=%d)", len(query or ""), k)
        vs = _get_vectorstore()
        results = vs.similarity_search(query or "", k=max(1, int(k)))
        out = "\n\n".join((r.page_content or "") for r in results)
        logger.info(
            "search_docs: done (results=%d out_chars=%d)",
            len(results or []),
            len(out),
        )
        return out

    return mcp


def main() -> None:
    transport = os.environ.get("MCP_TRANSPORT", "stdio").lower()
    if transport not in ("stdio", "sse", "streamable-http"):
        raise ValueError(
            f"Unsupported MCP_TRANSPORT={transport!r}. "
            "Expected one of: stdio, sse, streamable-http"
        )
    server = build_mcp_server()
    server.run(transport=transport)  # type: ignore[arg-type]


if __name__ == "__main__":
    main()
