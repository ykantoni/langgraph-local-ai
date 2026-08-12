# Local docs LangChain agent

This project runs a CLI agent that answers questions using local `.txt` files under `./docs`, powered by a local Ollama server.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Start Ollama and pull models:

```powershell
$env:OLLAMA_BASE_URL="http://localhost:11434"
ollama pull granite3.3
ollama pull nomic-embed-text
```

## Run

```powershell
cd frontend
npm run build
cd ..
python -m local_agent.server
```

Optional model overrides:

```powershell
$env:OLLAMA_CHAT_MODEL="granite3.3"
$env:OLLAMA_EMBED_MODEL="nomic-embed-text"
python .\agent.py
```

## Notes

- Put one or more `.txt` files in a `docs` folder at the project root.
- If `docs` is missing or empty, the script now exits with a clear error message.

## MCP integration

The local FAISS document search is also exposed as a [Model Context Protocol](https://modelcontextprotocol.io)
tool, so the agent (and any other MCP-compatible client, e.g. Cursor or Claude
Desktop) can call it through a standard transport.

### Run the MCP server

Stdio mode (one short-lived subprocess per tool call, simplest setup):

```powershell
python -m local_agent.mcp_server
```

Streamable-HTTP mode (long-lived process, FAISS/embeddings stay warm between
tool calls):

```powershell
$env:MCP_TRANSPORT="streamable-http"
# optional, defaults shown:
# $env:MCP_HTTP_HOST="127.0.0.1"
# $env:MCP_HTTP_PORT="8765"
# $env:MCP_HTTP_PATH="/mcp"
python -m local_agent.mcp_server
```

The server registers one tool: `search_docs(query: str, k: int = 3) -> str`.
It reads the same `DOCS_DIR` / `FAISS_*` / `ST_*` environment variables as the
agent, so the FAISS index is shared between the two processes.

### Make the agent use MCP

Two patterns:

**A. Stdio (agent spawns the MCP server)** — simplest, but every tool call
spawns a fresh subprocess that has to reload Sentence-Transformers + FAISS:

```powershell
$env:MCP_ENABLED="1"
$env:MCP_SERVERS_JSON='{"local-docs": {"command": "python", "args": ["-m", "local_agent.mcp_server"], "transport": "stdio"}}'
```

**B. Streamable HTTP (out-of-band, persistent MCP server)** — start the MCP
server once in its own terminal, then point the agent at it. The vector store
stays warm between calls:

```powershell
# Terminal 1: the MCP server
& .\.venv\Scripts\Activate.ps1
$env:MCP_TRANSPORT="streamable-http"
python -m local_agent.mcp_server   # binds 127.0.0.1:8765/mcp

# Terminal 2: the agent
& .\.venv\Scripts\Activate.ps1
$env:MCP_ENABLED="1"
$env:MCP_SERVERS_JSON='{"local-docs": {"transport": "streamable_http", "url": "http://127.0.0.1:8765/mcp"}}'
python -m local_agent.server
```

# cmd
```
set MCP_ENABLED=1
set "MCP_SERVERS_JSON={"local-docs":{"transport":"streamable_http","url":"http://127.0.0.1:8765/mcp"}}"
python -m local_agent.server
```

Note the MCP server binds **port 8765** by default — different from the
agent's FastAPI port (8000). Override with `MCP_HTTP_HOST` / `MCP_HTTP_PORT`
on the MCP server, and update the URL in `MCP_SERVERS_JSON` to match.

On startup, `local_agent.runtime.load_chat_agent` will:

1. Spawn the configured MCP server(s) and load their tools via
   `langchain-mcp-adapters`.
2. Probe Ollama's `/api/show` endpoint to check whether
   `OLLAMA_CHAT_MODEL` advertises the `tools` capability.
3. If both succeed, use the **tool-calling retrieve path**: the LLM is
   bound to the MCP tools and chooses which one to call inside the
   `retrieve` node of the planner / critic graph.
4. Otherwise, fall back to the original **direct-retrieval path** that
   calls the in-process FAISS store directly.

The planner, executor, synthesizer, and critic nodes are unchanged in
either mode — only the `retrieve` node switches implementation.

