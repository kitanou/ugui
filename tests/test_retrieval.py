import pytest

from ugui_core.retrieval import LocalRetriever


class Semantic:
    model_key = "fixture"

    def embed(self, texts):
        return [[1.0, 0.0] if "coffee" in t or "珈琲" in t else [0.0, 1.0] for t in texts]


def test_lexical_japanese_and_exclusions(store, make_tweet):
    store.add_tweet(make_tweet("1", "冷たいそばが好き"))
    store.add_tweet(make_tweet("2", "そばの広告", excluded_reason="retweet"))
    store.add_tweet(make_tweet("3", "海へ出かける"))
    assert [t.id for t in LocalRetriever(store).search("そば")] == ["1"]
    assert LocalRetriever(store).search("unrelated") == []


def test_semantic_cross_language_and_model_namespace(store, make_tweet):
    store.add_tweet(make_tweet("1", "珈琲を飲む"))
    store.add_tweet(make_tweet("2", "海へ出かける"))
    r = LocalRetriever(store, Semantic())
    with pytest.raises(ValueError, match="index"):
        r.search("coffee")
    assert r.index() == 2
    assert r.index() == 0
    assert [t.id for t in r.search("coffee")] == ["1"]
    assert not store.vectors("different-model")
