"""HTTP API for the local chat agent (for browser or other UI clients)."""

import os
from pathlib import Path
from contextlib import asynccontextmanager
import json
import queue
import threading
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
        - {type:"token", token:"..."}
        - {type:"done"}
        - {type:"error", error:"..."}
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

        class TokenHandler(BaseCallbackHandler):
            def on_llm_new_token(self, token: str, **kwargs):  # type: ignore[override]
                # Don't raise here: it creates noisy logs. The SSE loop will stop
                # on disconnect/cancel; this just prevents queue growth.
                if cancel_event.is_set():
                    return
                if token:
                    assistant_parts.append(token)
                    token_q.put(token)

        handler = TokenHandler()

        def run_agent():
            try:
                # AgentExecutor is a Runnable; pass callbacks via config.
                _agent.invoke({"input": user_input}, config={"callbacks": [handler]})
            except Exception as e:
                err_q.put(str(e))
            finally:
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
                        return
                except Exception:
                    # If disconnect detection fails, continue best-effort.
                    pass
                try:
                    token = token_q.get(timeout=0.25)
                except queue.Empty:
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

                yield f"data: {json.dumps({'type': 'token', 'token': token})}\n\n"

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
