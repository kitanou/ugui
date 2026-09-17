import json

from ugui_core.evaluation import evaluate


class EvaluationModel:
    def __init__(self):
        self.seen = []

    def complete(self, messages, **kwargs):
        self.seen.append(messages)
        if "extract conservative" in messages[0]["content"]:
            return '{"claims": []}'
        return "そばが好きだな。"


def test_temporal_holdout_has_no_future_or_persona_leakage(store, make_tweet):
    store.add_tweet(make_tweet("1", "そばが好き", day=1))
    store.add_tweet(make_tweet("2", "秘密のホールドアウト回答", day=2))
    store.add_tweet(make_tweet("3", "未来の発言", day=3))
    store.save_snapshot("persona", {"secret": "既存の全データ由来Persona"})
    model = EvaluationModel()
    result = evaluate(store, model, [{"tweet_id": "2", "situation": "昼食の話をして"}])
    prompts = json.dumps(model.seen, ensure_ascii=False)
    assert "秘密のホールドアウト回答" not in prompts
    assert "未来の発言" not in prompts
    assert "既存の全データ由来Persona" not in prompts
    assert result["training_count"] == 1
    assert result["results"][0]["rubric_scores"] is None
    assert store.snapshot() == {"secret": "既存の全データ由来Persona"}
