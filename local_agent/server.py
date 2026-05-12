"""HTTP API for the local chat agent (for browser or other UI clients)."""

import os
from pathlib import Path
from contextlib import asynccontextmanager
import json
import queue
import re
import threading
from queue import Empty
import uuid
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.responses import StreamingResponse
from pydantic import BaseModel, Field

from local_agent.runtime import load_chat_agent


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="Latest user message")
    session_id: str | None = Field(default=None, description="Server-side session id")


_agent = None
_sessions_lock = threading.Lock()
_sessions: dict[str, list[ChatMessage]] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _agent
    _agent, _ = load_chat_agent()
    yield
    _agent = None


def _build_input(message: str, history: list[ChatMessage] | None) -> str:
    if not history:
        return message
    parts: list[str] = []
    for m in history:
        label = "User" if m.role == "user" else "Assistant"
        parts.append(f"{label}: {m.content}")
    parts.append(f"User: {message}")
    return "\n".join(parts)

def _new_session_id() -> str:
    return uuid.uuid4().hex

def _get_session_history(session_id: str) -> list[ChatMessage]:
    with _sessions_lock:
        return list(_sessions.get(session_id, []))

def _append_session_message(session_id: str, msg: ChatMessage) -> None:
    with _sessions_lock:
        _sessions.setdefault(session_id, []).append(msg)

def _reset_session(session_id: str) -> None:
    with _sessions_lock:
        _sessions[session_id] = []


def create_app() -> FastAPI:
    app = FastAPI(title="Local agent chat API", lifespan=lifespan)

    origins_raw = os.environ.get("CHAT_CORS_ORIGINS", "*")
    origins = [o.strip() for o in origins_raw.split(",") if o.strip()]
    if not origins:
        origins = ["*"]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# curl 127.0.0.1:8000/health
    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/session/new")
    def session_new() -> dict[str, str]:
        session_id = _new_session_id()
        _reset_session(session_id)
        return {"session_id": session_id}

    @app.post("/chat/stream")
    async def chat_stream(body: ChatRequest, request: Request):
        """SSE streaming endpoint (ChatGPT-like).

        Emits events:
        - {type:"token", token:"..."}  (batched: often a full sentence or multi-token chunk)
        - {type:"done"}
        - {type:"error", error:"..."}

        Tuning via env (optional):
        - CHAT_SSE_CHUNK_CHARS: max chars before a hard flush (default 400)
        - CHAT_SSE_MIN_SOFT_CHARS: min buffered chars before soft flush on .!? or newline (default 24)
        """
        if _agent is None:
            raise HTTPException(status_code=503, detail="Agent not initialized")

        session_id = body.session_id
        history = _get_session_history(session_id) if session_id else None
        user_input = _build_input(body.message, history)
        if session_id:
            _append_session_message(session_id, ChatMessage(role="user", content=body.message))

        from langchain_core.callbacks import BaseCallbackHandler

        token_q: "queue.Queue[str | None]" = queue.Queue()
        err_q: "queue.Queue[str]" = queue.Queue()
        cancel_event = threading.Event()
        assistant_parts: list[str] = []

        chunk_chars = max(64, int(os.environ.get("CHAT_SSE_CHUNK_CHARS", "400")))
        min_soft = max(8, int(os.environ.get("CHAT_SSE_MIN_SOFT_CHARS", "24")))

        class BatchingTokenHandler(BaseCallbackHandler):
            """Buffer LLM tokens; emit on sentence boundary, newline, or size cap."""

            _SENTENCE_END = re.compile(r"[.!?](?:\s|$)")

            def __init__(self) -> None:
                super().__init__()
                self._buf = ""
                self._lock = threading.Lock()
                self._cancel_raised = False

            def on_llm_new_token(self, token: str, **kwargs):  # type: ignore[override]
                # If the client hit "Stop generating" (fetch aborted), the request disconnects
                # and we set cancel_event. Raising here is the most reliable way to stop the
                # underlying streaming LLM call and unwind the agent quickly.
                if cancel_event.is_set():
                    if not self._cancel_raised:
                        self._cancel_raised = True
                        raise RuntimeError("cancelled")
                    return
                if not token:
                    return
                assistant_parts.append(token)
                outgoing: list[str] = []
                with self._lock:
                    self._buf += token
                    while True:
                        piece = self._pop_chunk()
                        if piece is None:
                            break
                        outgoing.append(piece)
                for piece in outgoing:
                    token_q.put(piece)

            def _pop_chunk(self) -> str | None:
                b = self._buf
                if not b:
                    return None

                if len(b) >= chunk_chars:
                    self._buf = b[chunk_chars:]
                    return b[:chunk_chars]

                # Clear sentence boundaries (common in English) — flush even if buffer is short.
                for sep in (". ", "! ", "? ", ".\n", "!\n", "?\n", ".\t", "!\t", "?\t"):
                    idx = b.find(sep)
                    if idx != -1:
                        cut = idx + len(sep)
                        chunk = b[:cut]
                        self._buf = b[cut:]
                        return chunk

                if len(b) >= min_soft:
                    nl = b.find("\n")
                    if nl != -1:
                        line_len = nl + 1
                        if line_len >= min_soft or nl + 1 == len(b):
                            self._buf = b[line_len:]
                            return b[:line_len]

                    m = self._SENTENCE_END.search(b)
                    if m is not None and m.end() >= min_soft:
                        cut = m.end()
                        chunk = b[:cut]
                        self._buf = b[cut:]
                        return chunk

                return None

            def flush(self) -> None:
                with self._lock:
                    rest = self._buf
                    self._buf = ""
                if rest:
                    token_q.put(rest)

        handler = BatchingTokenHandler()

        def run_agent():
            try:
                if cancel_event.is_set():
                    return
                # AgentExecutor is a Runnable; pass callbacks via config.
                _agent.invoke({"input": user_input}, config={"callbacks": [handler]})
            except Exception as e:
                err_q.put(str(e))
            finally:
                handler.flush()
                token_q.put(None)

        t = threading.Thread(target=run_agent, daemon=True)
        t.start()

        async def sse():
            # initial event helps the client start rendering immediately
            yield f"data: {json.dumps({'type': 'start'})}\n\n"

            while True:
                # If the client disconnected, stop streaming immediately.
                if cancel_event.is_set():
                    return
                try:
                    if await request.is_disconnected():
                        cancel_event.set()
                        # Unblock SSE immediately even if the worker is still unwinding.
                        try:
                            token_q.put_nowait(None)
                        except Exception:
                            pass
                        return
                except Exception:
                    # If disconnect detection fails, continue best-effort.
                    pass
                try:
                    token = token_q.get(timeout=0.25)
                except Empty:
                    if not err_q.empty():
                        err = err_q.get_nowait()
                        yield f"data: {json.dumps({'type': 'error', 'error': err})}\n\n"
                        yield f"data: {json.dumps({'type': 'done'})}\n\n"
                        return
                    continue

                if token is None:
                    if not err_q.empty():
                        err = err_q.get_nowait()
                        yield f"data: {json.dumps({'type': 'error', 'error': err})}\n\n"
                    else:
                        if session_id:
                            assistant_text = "".join(assistant_parts).strip()
                            # Token streaming callbacks are best-effort; if nothing arrived,
                            # don't persist an empty assistant message (Pydantic min_length=1).
                            if assistant_text:
                                _append_session_message(
                                    session_id,
                                    ChatMessage(role="assistant", content=assistant_text),
                                )
                    yield f"data: {json.dumps({'type': 'done'})}\n\n"
                    return

                merged = token
                while True:
                    try:
                        nxt = token_q.get_nowait()
                    except Empty:
                        break
                    if nxt is None:
                        token_q.put(None)
                        break
                    merged += nxt

                yield f"data: {json.dumps({'type': 'token', 'token': merged})}\n\n"

        return StreamingResponse(
            sse(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    # Serve built frontend (Vite) if present.
    # Build it via:  cd frontend && npm run build
    dist_dir = Path(os.environ.get("FRONTEND_DIST_DIR", "./frontend/dist")).resolve()
    assets_dir = dist_dir / "assets"
    if dist_dir.exists() and (dist_dir / "index.html").exists():
        if assets_dir.exists():
            app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

        @app.get("/", include_in_schema=False)
        def ui_index() -> FileResponse:
            return FileResponse(str(dist_dir / "index.html"))

        # SPA fallback for client-side routes (registered last so it won't shadow API routes)
        @app.get("/{full_path:path}", include_in_schema=False)
        def ui_fallback(full_path: str) -> FileResponse:
            candidate = (dist_dir / full_path).resolve()
            if str(candidate).startswith(str(dist_dir)) and candidate.exists() and candidate.is_file():
                return FileResponse(str(candidate))
            return FileResponse(str(dist_dir / "index.html"))

    return app


app = create_app()


def main() -> None:
    import uvicorn

    host = os.environ.get("API_HOST", "127.0.0.1")
    port = int(os.environ.get("API_PORT", "8000"))
    uvicorn.run("local_agent.server:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
