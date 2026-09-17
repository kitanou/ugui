from __future__ import annotations

import ipaddress
import os
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit


def validate_endpoint(url: str, allow_remote: bool) -> str:
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("Endpoint must be an HTTP(S) URL without embedded credentials")
    if parsed.query or parsed.fragment:
        raise ValueError("Endpoint must not contain a query or fragment")
    # Require literal loopback addresses: no DNS rebinding or LAN forwarding by default.
    try:
        local = ipaddress.ip_address(parsed.hostname).is_loopback
    except ValueError:
        local = False
    if not local and not allow_remote:
        raise ValueError("Non-loopback endpoints require UGUI_ALLOW_REMOTE=true; use 127.0.0.1 locally")
    return url.rstrip("/")


@dataclass(frozen=True)
class Settings:
    data_dir: Path = field(default_factory=lambda: Path("data"))
    llm_url: str = "http://127.0.0.1:1234/v1"
    llm_model: str = "local-model"
    embedding_url: str = "http://127.0.0.1:1234/v1"
    embedding_model: str = ""
    allow_remote: bool = False
    api_key: str = field(default="", repr=False)
    llm_api_key: str = field(default="", repr=False)
    timeout: float = 120.0

    def __post_init__(self):
        validate_endpoint(self.llm_url, self.allow_remote)
        validate_endpoint(self.embedding_url, self.allow_remote)

    @classmethod
    def from_env(cls):
        return cls(
            data_dir=Path(os.getenv("UGUI_DATA_DIR", "data")),
            llm_url=os.getenv("UGUI_LLM_URL", "http://127.0.0.1:1234/v1"),
            llm_model=os.getenv("UGUI_LLM_MODEL", "local-model"),
            embedding_url=os.getenv("UGUI_EMBEDDING_URL", "http://127.0.0.1:1234/v1"),
            embedding_model=os.getenv("UGUI_EMBEDDING_MODEL", ""),
            allow_remote=os.getenv("UGUI_ALLOW_REMOTE", "false").lower() == "true",
            api_key=os.getenv("UGUI_API_KEY", ""),
            llm_api_key=os.getenv("UGUI_LLM_API_KEY", ""),
        )
