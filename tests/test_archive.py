import json
import zipfile

import pytest

from ugui_core.archive import decode_archive, import_archive, normalize


def row(id, text="冷たいそばが好き", **extras):
    return {
        "tweet": {
            "id_str": str(id),
            "full_text": text,
            "created_at": "Wed Jan 01 12:00:00 +0000 2025",
            **extras,
        }
    }


def test_zip_owner_filter_idempotent_and_immutable(tmp_path, store):
    path = tmp_path / "archive.zip"
    rows = [
        row(1),
        row(2),
        row(3, "RT @friend: hi"),
        row(4, "https://t.co/x"),
        row(5, "他人", user_id_str="someone"),
        row(6, "自動投稿", source="IFTTT"),
        row(7, "お返事 &amp; そば", in_reply_to_status_id_str="99"),
        row(8, "引用コメント", quoted_status_id_str="100"),
        {"tweet": {"id_str": "oops"}},
    ]
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("data/account.js", 'window.YTD.account.part0 = [{"account":{"accountId":"42"}}];')
        z.writestr("data/tweets.js", "window.YTD.tweets.part0 = " + json.dumps(rows) + ";")
        z.writestr("../../unrelated.txt", "not extracted")
    original = path.read_bytes()
    counts = import_archive(path, store)
    assert counts == {"imported": 8, "duplicate": 0, "excluded": 5, "invalid": 1}
    assert [t.id for t in store.tweets()] == ["1", "7", "8"]
    assert store.tweets()[1].reply_to == "99"
    assert store.tweets()[1].text == "お返事 & そば"
    assert store.tweets()[2].quote_to == "100"
    assert import_archive(path, store)["duplicate"] == 8
    assert path.read_bytes() == original
    assert not (tmp_path / "unrelated.txt").exists()


def test_owner_required_and_conflict(tmp_path, store):
    p = tmp_path / "tweets.json"
    p.write_text(json.dumps([row(1)]))
    with pytest.raises(ValueError, match="owner"):
        import_archive(p, store)
    import_archive(p, store, "42")
    with pytest.raises(ValueError, match="another"):
        import_archive(p, store, "43")


def test_js_not_executed_and_timezone():
    with pytest.raises(ValueError):
        decode_archive("window.YTD.tweets.part0 = []; malicious();")
    t = normalize(row(1, created_at="2025-01-01T09:00:00+09:00"), "tweets.js", "42", ())
    assert t.timestamp == "2025-01-01T00:00:00+00:00"
    with pytest.raises(ValueError):
        normalize(row(1, created_at="2025-01-01T09:00:00"), "tweets.js", "42", ())


def test_multipart_directory(tmp_path, store):
    (tmp_path / "tweets.js").write_text(json.dumps([row(1)]))
    (tmp_path / "tweets-part1.js").write_text(json.dumps([row(2, "別の投稿")]))
    assert import_archive(tmp_path, store, "42")["imported"] == 2


def test_numeric_foreign_author_is_excluded():
    tweet = normalize(row(1, user={"id": 99}), "tweets.json", "42", ())
    assert tweet.excluded_reason == "other_author"
