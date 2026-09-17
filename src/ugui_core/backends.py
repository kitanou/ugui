from __future__ import annotations

import math
from typing import Protocol

import httpx

from .config import Settings, validate_endpoint


class BackendError(RuntimeError):
    pass


class Completion(str):
    """Text-compatible completion preserving the upstream termination reason."""

    def __new__(cls, content: str, finish_reason: str = "stop"):
        value = super().__new__(cls, content)
        value.finish_reason = finish_reason
        return value


class LLM(Protocol):
    def complete(self, messages: list[dict], *, temperature: float = 0.7, max_tokens: int = 1024) -> str: ...


class Embedder(Protocol):
    model_key: str

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class OpenAIBackend:
    def __init__(self, settings: Settings):
        self.settings = settings

    def post(self, base: str, route: str, payload: dict) -> dict:
        base = validate_endpoint(base, self.settings.allow_remote)
        headers = (
            {"Authorization": f"Bearer {self.settings.llm_api_key}"} if self.settings.llm_api_key else {}
        )
        try:
            # Ignore system proxy settings and reject redirects so loopback data stays local.
            with httpx.Client(
                timeout=self.settings.timeout, trust_env=False, follow_redirects=False
            ) as client:
                response = client.post(base + route, json=payload, headers=headers)
                response.raise_for_status()
                return response.json()
        except (httpx.HTTPError, ValueError) as exc:
            # Provider response bodies can contain private prompts. Do not relay or log them.
            raise BackendError("LLM/embedding backend unavailable or returned an invalid response") from exc

    def complete(self, messages: list[dict], *, temperature=0.7, max_tokens=1024) -> str:
        data = self.post(
            self.settings.llm_url,
            "/chat/completions",
            {
                "model": self.settings.llm_model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": False,
            },
        )
        try:
            content = data["choices"][0]["message"]["content"]
            if not isinstance(content, str) or not content.strip():
                raise ValueError("Empty completion")
            reason = data["choices"][0].get("finish_reason", "stop")
            return Completion(content, reason if reason in {"stop", "length", "content_filter"} else "stop")
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise BackendError("LLM backend returned no text completion") from exc


class OpenAIEmbedder(OpenAIBackend):
    def __init__(self, settings: Settings):
        super().__init__(settings)
        if not settings.embedding_model:
            raise ValueError("Set UGUI_EMBEDDING_MODEL to use semantic search")
        self.model_key = settings.embedding_url.rstrip("/") + "#" + settings.embedding_model

    def embed(self, texts: list[str]) -> list[list[float]]:
        data = self.post(
            self.settings.embedding_url,
            "/embeddings",
            {
                "model": self.settings.embedding_model,
                "input": texts,
            },
        )
        try:
            rows = sorted(data["data"], key=lambda r: r["index"])
            if [r["index"] for r in rows] != list(range(len(texts))):
                raise ValueError("Embedding count mismatch")
            vectors = [[float(v) for v in row["embedding"]] for row in rows]
            if any(not v or not all(math.isfinite(x) for x in v) or not any(v) for v in vectors):
                raise ValueError("Invalid vector")
            if len({len(v) for v in vectors}) > 1:
                raise ValueError("Inconsistent embedding dimensions")
            return vectors
        except (KeyError, TypeError, ValueError) as exc:
            raise BackendError("Embedding backend returned invalid vectors") from exc
