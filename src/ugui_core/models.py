from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Tweet(BaseModel):
    id: str
    timestamp: str
    text: str
    author_id: str | None = None
    reply_to: str | None = None
    quote_to: str | None = None
    kind: Literal["tweet", "reply", "quote", "retweet"] = "tweet"
    source: str = ""
    source_file: str = ""
    topics: list[str] = Field(default_factory=list)
    excluded_reason: str | None = None


class Claim(BaseModel):
    tweet_id: str
    category: Literal["identity", "preferences", "opinions", "interests", "behavior"]
    subject: str = Field(min_length=1, max_length=120)
    attribute: str = Field(min_length=1, max_length=120)
    value: str = Field(min_length=1, max_length=240)
    quote: str = Field(min_length=1, max_length=1000)


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(max_length=32000)


class ChatRequest(BaseModel):
    model: str = "ugui"
    messages: list[Message] = Field(min_length=1, max_length=100)
    stream: bool = False
    temperature: float = Field(default=0.7, ge=0, le=2)
    max_tokens: int = Field(default=1024, ge=1, le=8192)
