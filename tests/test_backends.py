import httpx
import pytest

from ugui_core.backends import BackendError, OpenAIBackend, OpenAIEmbedder
from ugui_core.config import Settings, validate_endpoint


@pytest.mark.parametrize(
    "url",
    [
        "https://example.com/v1",
        "http://localhost/v1",
        "http://192.168.1.2/v1",
        "file:///tmp/x",
        "http://user:pass@127.0.0.1/v1",
        "http://127.0.0.1/v1?key=x",
    ],
)
def test_default_blocks_non_loopback_or_unsafe_endpoints(url):
    with pytest.raises(ValueError):
        validate_endpoint(url, False)


def test_remote_explicit_opt_in():
    assert validate_endpoint("https://example.com/v1", True) == "https://example.com/v1"
    assert validate_endpoint("http://[::1]:1234/v1", False)
    assert "secret" not in repr(Settings(llm_api_key="secret", api_key="secret"))


def test_transport_does_not_follow_redirects_or_use_proxy(monkeypatch):
    actual_client = httpx.Client
    calls = []

    def factory(**kwargs):
        assert kwargs["trust_env"] is False
        assert kwargs["follow_redirects"] is False

        def handler(request):
            calls.append(request)
            return httpx.Response(302, headers={"location": "https://example.com"})

        return actual_client(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(httpx, "Client", factory)
    with pytest.raises(BackendError):
        OpenAIBackend(Settings()).complete([{"role": "user", "content": "private"}])
    assert len(calls) == 1


def test_embedding_response_validated(monkeypatch):
    backend = OpenAIEmbedder(Settings(embedding_model="test"))
    monkeypatch.setattr(backend, "post", lambda *args: {"data": [{"index": 0, "embedding": [float("nan")]}]})
    with pytest.raises(BackendError):
        backend.embed(["x"])


def test_completion_preserves_truncation(monkeypatch):
    backend = OpenAIBackend(Settings())
    monkeypatch.setattr(
        backend,
        "post",
        lambda *args: {"choices": [{"message": {"content": "unfinished"}, "finish_reason": "length"}]},
    )
    answer = backend.complete([])
    assert answer == "unfinished"
    assert answer.finish_reason == "length"
