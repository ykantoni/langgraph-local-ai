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

Local document search is exposed via [mcp-local-rag](https://github.com/shinpr/mcp-local-rag)
through the Model Context Protocol. The launcher in `local_agent.mcp_server` runs
`mcp-local-rag` (stdio) or wraps it with [supergateway](https://github.com/supercorp-ai/supergateway)
for streamable HTTP on port **8765** by default.

**Prerequisites:** Node.js 22+ with `npx` on PATH.

### Run the MCP server

Stdio mode (one short-lived subprocess per tool call):

```powershell
python -m local_agent.mcp_server
```

Streamable-HTTP mode (long-lived process; LanceDB + embeddings stay warm):

```powershell
$env:MCP_TRANSPORT="streamable-http"
# optional, defaults shown:
# $env:MCP_HTTP_PORT="8765"
# $env:MCP_HTTP_PATH="/mcp"
# $env:MCP_AUTO_SYNC="1"          # run mcp-local-rag sync before serving
python -m local_agent.mcp_server
```

# cmd
```cmd
set MCP_TRANSPORT=streamable-http
rem optional, defaults shown:
rem set MCP_HTTP_PORT=8765
rem set MCP_HTTP_PATH=/mcp
rem set MCP_AUTO_SYNC=1
python -m local_agent.mcp_server
```

Key tools from mcp-local-rag include `query_documents`, `sync_start`, `list_files`,
and `ingest_file`. Document roots and LanceDB paths follow the same `DOCS_DIR` /
`FAISS_INDEX_DIR` settings as the agent (`BASE_DIR`, `DB_PATH`, `CACHE_DIR` env
vars are set automatically).

Override LanceDB location:

```powershell
$env:MCP_RAG_DB_PATH="./faiss_index/mcp_local_rag"
```

### Make the agent use MCP

**A. Stdio (agent spawns the MCP server)** — simplest, but every tool call spawns
a fresh subprocess:

```powershell
$env:MCP_ENABLED="1"
$env:MCP_SERVERS_JSON='{"local-docs": {"command": "python", "args": ["-m", "local_agent.mcp_server"], "transport": "stdio"}}'
```

**B. Streamable HTTP (out-of-band, persistent MCP server)** — start the MCP server
once in its own terminal, then point the agent at it:

```powershell
# Terminal 1: the MCP server
& .\.venv\Scripts\Activate.ps1
$env:MCP_TRANSPORT="streamable-http"
python -m local_agent.mcp_server   # binds 127.0.0.1:8765/mcp

# Terminal 2: the agent
& .\.venv\Scripts\Activate.ps1
$env:MCP_ENABLED="1"
$env:MCP_SERVERS_JSON='{"local-docs": {"transport": "streamable_http", "url": "http://127.0.0.1:8765/mcp"}}'
$env:MCP_TOOL_SEARCH_NAME="query_documents"
python -m local_agent.server
```

# cmd
```
set MCP_ENABLED=1
set "MCP_SERVERS_JSON={"local-docs":{"transport":"streamable_http","url":"http://127.0.0.1:8765/mcp"}}"
set MCP_TOOL_SEARCH_NAME=query_documents
python -m local_agent.server
```

Note the MCP server binds **port 8765** by default — different from the agent's
FastAPI port (8000). Override with `MCP_HTTP_PORT` on the MCP server and update
the URL in `MCP_SERVERS_JSON` to match.

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

