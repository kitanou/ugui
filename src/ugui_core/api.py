from __future__ import annotations

import hmac
import json
import time
import uuid

from fastapi import Depends, FastAPI, Header
from fastapi.responses import JSONResponse, StreamingResponse

from .backends import BackendError, OpenAIBackend, OpenAIEmbedder
from .config import Settings
from .engine import Engine
from .models import ChatRequest
from .retrieval import LocalRetriever
from .store import Store


def create_app(settings: Settings | None = None, engine: Engine | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    if engine is None:
        store = Store(settings.data_dir / "ugui.sqlite3")
        embedder = OpenAIEmbedder(settings) if settings.embedding_model else None
        engine = Engine(store, LocalRetriever(store, embedder), OpenAIBackend(settings))
    app = FastAPI(title="UGUI Core", version="0.1.0")

    def auth(authorization: str | None = Header(default=None)):
        if settings.api_key and not hmac.compare_digest(authorization or "", f"Bearer {settings.api_key}"):
            from fastapi import HTTPException

            raise HTTPException(status_code=401, detail="Invalid API key")

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "core": "ugui",
            "retrieval": "semantic" if settings.embedding_model else "lexical",
            "backend_checked": False,
        }

    @app.get("/models", dependencies=[Depends(auth)])
    @app.get("/v1/models", dependencies=[Depends(auth)])
    def models():
        return {
            "object": "list",
            "data": [{"id": "ugui", "object": "model", "created": 0, "owned_by": "ugui"}],
        }

    @app.post("/v1/chat/completions", dependencies=[Depends(auth)])
    def chat(body: ChatRequest):
        def error(message, status, kind):
            return JSONResponse(status_code=status, content={"error": {"message": message, "type": kind}})

        if body.model != "ugui":
            return error("Unknown model; use ugui", 404, "model_not_found")
        if sum(len(m.content) for m in body.messages) > 48000:
            return error("Conversation exceeds 48000 characters", 400, "invalid_request_error")
        try:
            content = engine.respond(body.messages, temperature=body.temperature, max_tokens=body.max_tokens)
        except BackendError:
            return error("Local model unavailable; check backend configuration", 503, "backend_error")
        except ValueError as exc:
            return error(str(exc), 400, "invalid_request_error")
        base = {"id": "chatcmpl-" + uuid.uuid4().hex, "created": int(time.time()), "model": "ugui"}
        finish_reason = getattr(content, "finish_reason", "stop")
        if not body.stream:
            return {
                **base,
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": finish_reason,
                    }
                ],
            }

        def events():
            # Buffered SSE compatibility: model generation finishes before chunks are sent.
            def chunk(delta, finish=None):
                return (
                    "data: "
                    + json.dumps(
                        {
                            **base,
                            "object": "chat.completion.chunk",
                            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
                        },
                        ensure_ascii=False,
                    )
                    + "\n\n"
                )

            yield chunk({"role": "assistant"})
            for start in range(0, len(content), 64):
                yield chunk({"content": content[start : start + 64]})
            yield chunk({}, finish_reason)
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"}
        )

    return app
