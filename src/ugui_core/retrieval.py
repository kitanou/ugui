from __future__ import annotations

import math
import re
from collections import Counter
from typing import Protocol

from .backends import Embedder
from .models import Tweet
from .store import Store


def terms(text: str) -> Counter:
    text = text.casefold()
    tokens = re.findall(r"[a-z0-9_]+", text)
    for run in re.findall(r"[\u3040-\u30ff\u3400-\u9fff]+", text):
        tokens.extend(run[i : i + 2] for i in range(max(1, len(run) - 1)))
    return Counter(tokens)


def cosine(a, b) -> float:
    if len(a) != len(b):
        raise ValueError("Embedding dimensions changed; rebuild the index")
    denom = math.sqrt(sum(x * x for x in a) * sum(x * x for x in b))
    return sum(x * y for x, y in zip(a, b, strict=True)) / denom if denom else 0.0


class Retriever(Protocol):
    def search(self, query: str, limit: int = 8) -> list[Tweet]: ...


class LocalRetriever:
    """Replaceable exact cosine vector retrieval, with explicit lexical-only fallback."""

    def __init__(self, store: Store, embedder: Embedder | None = None):
        self.store, self.embedder = store, embedder

    def index(self, rebuild=False) -> int:
        if self.embedder is None:
            raise ValueError("Semantic indexing requires UGUI_EMBEDDING_MODEL")
        existing = {} if rebuild else self.store.vectors(self.embedder.model_key)
        pending = [t for t in self.store.tweets() if t.id not in existing]
        for start in range(0, len(pending), 32):
            batch = pending[start : start + 32]
            vectors = self.embedder.embed([t.text for t in batch])
            self.store.save_vectors(
                self.embedder.model_key, [(t.id, v) for t, v in zip(batch, vectors, strict=True)]
            )
        return len(pending)

    def search(self, query: str, limit: int = 8) -> list[Tweet]:
        tweets = self.store.tweets()
        if not tweets or not query.strip():
            return []
        if self.embedder:
            vectors = self.store.vectors(self.embedder.model_key)
            if any(t.id not in vectors for t in tweets):
                raise ValueError("Semantic index is incomplete; run ugui index")
            q = self.embedder.embed([query])[0]
            scores = [(cosine(q, vectors[t.id]), t) for t in tweets]
        else:
            q = terms(query)
            document_terms = [terms(t.text) for t in tweets]
            frequencies = Counter(w for d in document_terms for w in d)
            scores = []
            for t, d in zip(tweets, document_terms, strict=True):
                score = sum(
                    math.log(1 + len(tweets) / (1 + frequencies[w])) * min(n, d[w]) for w, n in q.items()
                ) / math.sqrt(max(1, sum(d.values())))
                scores.append((score, t))
        return [
            t for score, t in sorted(scores, key=lambda x: (x[0], x[1].timestamp), reverse=True) if score > 0
        ][:limit]
