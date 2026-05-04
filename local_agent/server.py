"""HTTP API for the local chat agent (for browser or other UI clients)."""

import os
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from local_agent.runtime import load_chat_agent


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="Latest user message")
    history: list[ChatMessage] | None = Field(
        default=None,
        description="Prior turns; server turns them into context for the agent",
    )


class ChatResponse(BaseModel):
    response: str


_agent = None


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

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/chat", response_model=ChatResponse)
    def chat(body: ChatRequest) -> ChatResponse:
        if _agent is None:
            raise HTTPException(status_code=503, detail="Agent not initialized")
        user_input = _build_input(body.message, body.history)
        try:
            text = _agent.run(user_input)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e
        return ChatResponse(response=text if isinstance(text, str) else str(text))

    return app


app = create_app()


def main() -> None:
    import uvicorn

    host = os.environ.get("API_HOST", "127.0.0.1")
    port = int(os.environ.get("API_PORT", "8000"))
    uvicorn.run("local_agent.server:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
