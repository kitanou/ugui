import json

from fastapi.testclient import TestClient

from ugui_core.api import create_app
from ugui_core.backends import BackendError
from ugui_core.config import Settings
from ugui_core.engine import Engine
from ugui_core.persona import build_persona
from ugui_core.retrieval import LocalRetriever


class FakeModel:
    def complete(self, messages, **kwargs):
        self.messages = messages
        return "冷たいそばが好きだな。"


def client(store, key=""):
    model = FakeModel()
    app = create_app(Settings(api_key=key), Engine(store, LocalRetriever(store), model))
    return TestClient(app), model


def test_chat_context_and_stream_contract(store, make_tweet):
    store.add_tweet(make_tweet())
    build_persona(store)
    c, llm = client(store)
    body = {"model": "ugui", "messages": [{"role": "user", "content": "そばは好き？"}]}
    response = c.post("/v1/chat/completions", json=body)
    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "冷たいそばが好きだな。"
    assert "2025-01-01" in llm.messages[0]["content"]
    assert "tweet_evidence" in llm.messages[0]["content"]
    streamed = c.post("/v1/chat/completions", json={**body, "stream": True})
    assert streamed.headers["content-type"].startswith("text/event-stream")
    events = [line[6:] for line in streamed.text.splitlines() if line.startswith("data: ")]
    assert events[-1] == "[DONE]"
    assert json.loads(events[-2])["choices"][0]["finish_reason"] == "stop"
    assert (
        "".join(json.loads(e)["choices"][0]["delta"].get("content", "") for e in events[:-1])
        == response.json()["choices"][0]["message"]["content"]
    )


def test_auth_models_and_validation(store):
    c, _ = client(store, "secret")
    assert c.get("/health").status_code == 200
    assert c.get("/models").status_code == 401
    assert c.get("/v1/models", headers={"Authorization": "Bearer secret"}).json()["data"][0]["id"] == "ugui"
    headers = {"Authorization": "Bearer secret"}
    assert c.post("/v1/chat/completions", headers=headers, json={"messages": []}).status_code == 422
    assert (
        c.post(
            "/v1/chat/completions",
            headers=headers,
            json={"messages": [{"role": "assistant", "content": "hello"}]},
        ).status_code
        == 400
    )
    assert (
        c.post(
            "/v1/chat/completions",
            headers=headers,
            json={"model": "other", "messages": [{"role": "user", "content": "hello"}]},
        ).status_code
        == 404
    )


def test_backend_failure_is_safe(store):
    class Broken:
        def complete(self, *args, **kwargs):
            raise BackendError("private source text")

    c = TestClient(create_app(Settings(), Engine(store, LocalRetriever(store), Broken())))
    r = c.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 503
    assert "private" not in r.text


def test_outdated_persona_rejected(store, make_tweet):
    store.add_tweet(make_tweet())
    build_persona(store)
    store.add_tweet(make_tweet("2", "今は温かいそばが好き", day=2))
    c, _ = client(store)
    r = c.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "そばは？"}]})
    assert r.status_code == 400
    assert "outdated" in r.json()["error"]["message"]


def test_length_finish_reason_passes_through(store):
    from ugui_core.backends import Completion

    class Truncated:
        def complete(self, *args, **kwargs):
            return Completion("途中", "length")

    c = TestClient(create_app(Settings(), Engine(store, LocalRetriever(store), Truncated())))
    body = {"messages": [{"role": "user", "content": "hi"}]}
    assert c.post("/v1/chat/completions", json=body).json()["choices"][0]["finish_reason"] == "length"
    assert '"finish_reason": "length"' in c.post("/v1/chat/completions", json={**body, "stream": True}).text
