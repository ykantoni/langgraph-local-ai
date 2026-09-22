"""Launch mcp-local-rag as an MCP server (stdio or streamable HTTP).

mcp-local-rag (https://github.com/shinpr/mcp-local-rag) indexes documents under
``BASE_DIR`` into LanceDB and exposes tools such as ``query_documents``.

This module does not implement RAG itself. For streamable HTTP it runs
``supergateway``, which bridges the stdio-only npm package to
``http://127.0.0.1:8765/mcp`` by default.

Stdio (agent spawns a short-lived process per tool call):

    python -m local_agent.mcp_server

Streamable HTTP (long-lived; embeddings stay warm between tool calls):

    $env:MCP_TRANSPORT="streamable-http"
    python -m local_agent.mcp_server

Requires Node.js 22+ with ``npx`` on PATH.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys

from local_agent.config import load_settings

logger = logging.getLogger(__name__)


def _truthy(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _require_npx() -> None:
    if shutil.which("npx") is None:
        raise RuntimeError(
            "npx was not found on PATH. Install Node.js 22+ to run mcp-local-rag."
        )


def _npx_executable() -> str:
    """Resolve an npx executable path that subprocess can start on Windows."""
    candidate = shutil.which("npx")
    if candidate:
        return candidate
    candidate = shutil.which("npx.cmd")
    if candidate:
        return candidate
    return "npx"


def build_local_rag_env(settings) -> dict[str, str]:
    """Environment variables consumed by mcp-local-rag."""
    base_dir = os.path.abspath(settings.docs_dir)
    rag_root = os.environ.get(
        "MCP_RAG_DB_PATH",
        os.path.join(settings.faiss_index_dir, "mcp_local_rag"),
    )
    db_path = os.environ.get(
        "MCP_RAG_LANCEDB_PATH",
        os.path.join(rag_root, "lancedb"),
    )
    cache_dir = os.environ.get(
        "MCP_RAG_CACHE_DIR",
        os.path.join(rag_root, "models"),
    )
    return {
        "BASE_DIR": base_dir,
        "DB_PATH": db_path,
        "CACHE_DIR": cache_dir,
    }


def build_sync_command(base_dir: str) -> list[str]:
    return [_npx_executable(), "-y", "mcp-local-rag", "sync", base_dir]


def build_stdio_command() -> list[str]:
    return [_npx_executable(), "-y", "mcp-local-rag"]


def build_streamable_http_command(
    *,
    port: int,
    path: str,
    stdio_command: str,
    stateful: bool,
    session_timeout_ms: int | None,
) -> list[str]:
    cmd = [
        _npx_executable(),
        "-y",
        "supergateway",
        "--stdio",
        stdio_command,
        "--outputTransport",
        "streamableHttp",
        "--port",
        str(port),
        "--streamableHttpPath",
        path,
    ]
    if stateful:
        cmd.append("--stateful")
    if session_timeout_ms is not None:
        cmd.extend(["--sessionTimeout", str(session_timeout_ms)])
    return cmd


def run_sync(env: dict[str, str], base_dir: str) -> int:
    logger.info("mcp_server: syncing documents into LanceDB (base_dir=%s)", base_dir)
    try:
        result = subprocess.run(
            build_sync_command(base_dir),
            env=env,
            check=False,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "Failed to start npx. Ensure Node.js 22+ is installed and npx is on PATH."
        ) from exc
    if result.returncode != 0:
        logger.warning(
            "mcp_server: sync exited with code %s (index may be stale or empty)",
            result.returncode,
        )
    return result.returncode


def run_stdio(env: dict[str, str]) -> int:
    logger.info("mcp_server: starting mcp-local-rag (stdio)")
    try:
        return subprocess.run(build_stdio_command(), env=env).returncode
    except FileNotFoundError as exc:
        raise RuntimeError(
            "Failed to start npx. Ensure Node.js 22+ is installed and npx is on PATH."
        ) from exc


def run_streamable_http(env: dict[str, str]) -> int:
    port = int(os.environ.get("MCP_HTTP_PORT", "8765"))
    path = os.environ.get("MCP_HTTP_PATH", "/mcp")
    stdio_command = os.environ.get("MCP_LOCAL_RAG_COMMAND", "npx -y mcp-local-rag")
    stateful = _truthy(os.environ.get("MCP_HTTP_STATEFUL"), default=True)
    session_timeout_raw = os.environ.get("MCP_HTTP_SESSION_TIMEOUT_MS", "").strip()
    session_timeout_ms = int(session_timeout_raw) if session_timeout_raw else None

    logger.info(
        "mcp_server: starting streamable HTTP at http://127.0.0.1:%s%s",
        port,
        path,
    )
    cmd = build_streamable_http_command(
        port=port,
        path=path,
        stdio_command=stdio_command,
        stateful=stateful,
        session_timeout_ms=session_timeout_ms,
    )
    try:
        return subprocess.run(cmd, env=env).returncode
    except FileNotFoundError as exc:
        raise RuntimeError(
            "Failed to start npx/supergateway. Ensure Node.js 22+ is installed and npx is on PATH."
        ) from exc


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    _require_npx()

    settings = load_settings()
    env = os.environ.copy()
    env.update(build_local_rag_env(settings))

    if _truthy(os.environ.get("MCP_AUTO_SYNC"), default=True):
        run_sync(env, env["BASE_DIR"])

    transport = (os.environ.get("MCP_TRANSPORT") or "stdio").strip().lower()
    if transport in {"streamable-http", "streamable_http", "http"}:
        code = run_streamable_http(env)
    else:
        code = run_stdio(env)

    if code != 0:
        sys.exit(code)


if __name__ == "__main__":
    main()
