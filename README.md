# Local docs LangChain agent

This project runs a CLI agent that answers questions using local `.txt` files under `./docs`.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Set your OpenAI key:

```powershell
$env:OPENAI_API_KEY="your_key_here"
```

Or set it programmatically for this script by providing `OPENAI_API_KEY_VALUE`
(the script copies it to `OPENAI_API_KEY` at startup):

```powershell
$env:OPENAI_API_KEY_VALUE="your_key_here"
```

## Run

```powershell
python .\agent.py
```

## Notes

- Put one or more `.txt` files in a `docs` folder at the project root.
- If `docs` is missing or empty, the script now exits with a clear error message.

