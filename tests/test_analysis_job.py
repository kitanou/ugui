import json

import pytest

from ugui_core.analysis_job import analyze, batches, checkpoint_counts
from ugui_core.backends import BackendError


class Model:
    run_key = "test-extraction-v1"

    def __init__(self, broken=False):
        self.calls = 0
        self.broken = broken

    def complete(self, messages, **kwargs):
        self.calls += 1
        if self.broken:
            raise BackendError("unavailable")
        data = json.loads(messages[-1]["content"])
        return json.dumps(
            {
                "claims": [
                    {
                        "tweet_id": t["id"],
                        "category": "preferences",
                        "subject": "soba",
                        "attribute": "temperature",
                        "value": "cold",
                        "quote": t["text"],
                    }
                    for t in data
                ]
            }
        )


def test_resume_skips_successful_tweets_and_publishes_complete_profile(store, make_tweet):
    for i in range(1, 5):
        store.add_tweet(make_tweet(str(i), f"冷たいそば {i}", day=i))
    model = Model()
    result = analyze(store, model, batch_size=2, retry_delay=0)
    assert result["analysis"]["completed"] == 4
    assert result["patterns"][0]["status"] == "supported"
    assert model.calls == 2
    assert store.snapshot("analysis_status")["state"] == "completed"
    analyze(store, model, retry_delay=0)
    assert model.calls == 2
    store.add_tweet(make_tweet("5", "冷たいそば 5", day=5))
    analyze(store, model, retry_delay=0)
    assert model.calls == 3
    assert store.snapshot()["analysis"]["completed"] == 5


def test_failures_not_counted_complete_and_old_persona_preserved(store, make_tweet):
    store.add_tweet(make_tweet())
    store.save_snapshot("persona", {"original": True})
    with pytest.raises(BackendError):
        analyze(store, Model(broken=True), retry_delay=0)
    assert checkpoint_counts(store, Model.run_key) == (0, 1)
    assert store.snapshot() == {"original": True}
    assert store.snapshot("analysis_status")["state"] == "needs_attention"
    analyze(store, Model(), retry_delay=0)
    assert checkpoint_counts(store, Model.run_key) == (1, 0)


def test_checkpoints_survive_interrupt(store, make_tweet):
    for i in range(1, 5):
        store.add_tweet(make_tweet(str(i), f"そば {i}", day=i))

    class Interrupt(Model):
        def complete(self, messages, **kwargs):
            if self.calls == 1:
                raise KeyboardInterrupt()
            return super().complete(messages, **kwargs)

    with pytest.raises(KeyboardInterrupt):
        analyze(store, Interrupt(), batch_size=2, retry_delay=0)
    assert checkpoint_counts(store, Model.run_key) == (2, 0)
    model = Model()
    analyze(store, model, batch_size=2, retry_delay=0)
    assert model.calls == 1


def test_batch_character_limit(make_tweet):
    tweets = [make_tweet(str(i), "a" * 1500) for i in range(3)]
    assert [len(b) for b in batches(tweets)] == [1, 1, 1]


def test_analysis_backend_uses_bounded_output_and_qwen_non_thinking(monkeypatch):
    from ugui_core.analysis_job import AnalysisBackend
    from ugui_core.config import Settings

    backend = AnalysisBackend(Settings(llm_model="qwen/qwen3-8b"))
    payloads = []

    def post(base, route, payload):
        payloads.append(payload)
        return {"choices": [{"message": {"content": '{"claims": []}'}, "finish_reason": "stop"}]}

    monkeypatch.setattr(backend, "post", post)
    assert backend.complete([{"role": "user", "content": "[]"}]) == '{"claims": []}'
    assert "response_format" not in payloads[0]
    assert payloads[0]["max_tokens"] == 2048
    assert payloads[0]["messages"][-1]["content"].endswith("/no_think")


def test_cooperative_stop_preserves_checkpoints(store, make_tweet):
    for i in range(1, 5):
        store.add_tweet(make_tweet(str(i), f"そば {i}", day=i))

    def progress(status):
        if status["completed"] == 2:
            (store.path.parent / "analysis.stop").touch()

    assert analyze(store, Model(), batch_size=2, progress=progress, retry_delay=0) is None
    assert store.snapshot("analysis_status")["state"] == "stopped"
    assert checkpoint_counts(store, Model.run_key) == (2, 0)
