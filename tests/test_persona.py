import json

import pytest

from ugui_core.models import Claim
from ugui_core.persona import aggregate, build_persona, extract_claims


def claim(tweet, value="cold"):
    return Claim(
        tweet_id=tweet.id,
        category="preferences",
        subject="soba",
        attribute="temperature",
        value=value,
        quote=tweet.text,
    )


def test_support_distinct_days_and_conflict(make_tweet):
    tweets = [make_tweet(str(i), f"冷たいそば {i}", day=i) for i in range(1, 5)]
    single = aggregate([claim(tweets[0])] * 10, tweets)
    assert single[0]["evidence_count"] == 1
    assert single[0]["status"] == "candidate"
    supported = aggregate([claim(t) for t in tweets[:3]], tweets)
    assert supported[0]["status"] == "supported"
    changed = aggregate([claim(t) for t in tweets[:3]] + [claim(tweets[3], "hot")], tweets)
    assert next(p for p in changed if p["value"] == "cold")["status"] == "historical"
    assert next(p for p in changed if p["value"] == "hot")["status"] == "candidate"
    assert len(changed) == 2


def test_one_day_is_not_persistent(make_tweet):
    tweets = [make_tweet(str(i), f"そば {i}") for i in range(5)]
    assert aggregate([claim(t) for t in tweets], tweets)[0]["status"] == "candidate"


def test_extractor_rejects_invented_evidence(make_tweet):
    tweet = make_tweet()
    good = claim(tweet).model_dump()

    class Model:
        def complete(self, messages, **kwargs):
            return json.dumps(
                {
                    "claims": [
                        good,
                        {**good, "tweet_id": "999"},
                        {**good, "quote": "書いていない"},
                        {"broken": True},
                    ]
                }
            )

    assert extract_claims([tweet], Model()) == [claim(tweet)]


def test_rebuild_idempotent_and_failure_preserves_claims(store, make_tweet):
    tweet = make_tweet()
    store.add_tweet(tweet)
    store.replace_claims([claim(tweet)])
    profile = build_persona(store)
    assert build_persona(store) == profile
    assert profile["style"]["sample_count"] == 1

    class Broken:
        def complete(self, *args, **kwargs):
            return "not JSON"

    with pytest.raises(ValueError):
        build_persona(store, Broken())
    assert store.claims() == [claim(tweet)]


def test_import_invalidates_persona_until_extracted(store, make_tweet):
    first = make_tweet()
    store.add_tweet(first)
    store.replace_claims([claim(first)])
    original = build_persona(store)
    store.add_tweet(make_tweet("2", "今は温かいそばが好き", day=2))
    with pytest.raises(ValueError, match="refresh claims"):
        build_persona(store)
    assert store.snapshot() == original


def test_concurrent_import_aborts_atomic_persona_update(store, make_tweet):
    first = make_tweet()
    store.add_tweet(first)
    original = build_persona(store)

    class ImportDuringExtraction:
        def complete(self, messages, **kwargs):
            store.add_tweet(make_tweet("2", "新しい発言", day=2))
            return json.dumps({"claims": [claim(first).model_dump()]})

    with pytest.raises(ValueError, match="changed during"):
        build_persona(store, ImportDuringExtraction())
    assert store.snapshot() == original
    assert store.claims() == []


def test_co_supported_values_do_not_create_false_history(make_tweet):
    tweets = [make_tweet(str(i), f"冷たいざるそばが好き {i}", day=i) for i in range(1, 4)]
    patterns = aggregate([claim(t) for t in tweets] + [claim(tweets[-1], "zaru")], tweets)
    assert next(p for p in patterns if p["value"] == "cold")["status"] == "supported"
    assert all(not p["conflicting_values"] for p in patterns)
