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
python .\agent.py
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

