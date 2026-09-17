"""Parse archive data as JSON; JavaScript wrappers are never evaluated."""

from __future__ import annotations

import html
import json
import re
import zipfile
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

from .models import Tweet

WRAPPER = re.compile(r"^\s*window\.YTD\.[\w]+\.part\d+\s*=\s*")
TWEET_FILE = re.compile(r"^(tweets?|tweets?-part\d+)\.(js|json)$", re.I)
MAX_FILE_BYTES = 256 * 1024 * 1024
TOPICS = {
    "food": ("蕎麦", "そば", "コーヒー", "食", "料理", "coffee", "food"),
    "technology": ("python", "llm", "コード", "プログラ", "linux", "mac", "ai"),
    "travel": ("旅行", "電車", "旅", "ホテル", "travel"),
    "music": ("音楽", "歌", "ライブ", "music"),
    "work": ("仕事", "会議", "出勤", "work"),
}


def decode_archive(raw: str):
    raw = WRAPPER.sub("", raw.lstrip("\ufeff"), count=1).strip().removesuffix(";").strip()
    return json.loads(raw)


def archive_files(path: Path):
    """Read only expected members; never extract ZIP paths to disk."""
    if path.is_dir():
        files = sorted(
            p
            for p in path.rglob("*")
            if p.is_file() and (TWEET_FILE.match(p.name) or p.name in {"account.js", "account.json"})
        )
        for file in files:
            if file.stat().st_size > MAX_FILE_BYTES:
                raise ValueError("Archive member exceeds 256 MiB; split it before importing")
            yield str(file.relative_to(path)), file.read_text(encoding="utf-8-sig")
    elif zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            for member in sorted(archive.infolist(), key=lambda m: m.filename):
                name = PurePosixPath(member.filename).name
                if TWEET_FILE.match(name) or name in {"account.js", "account.json"}:
                    if member.file_size > MAX_FILE_BYTES:
                        raise ValueError("Archive member exceeds 256 MiB; split it before importing")
                    yield member.filename, archive.read(member).decode("utf-8-sig")
    else:
        if path.stat().st_size > MAX_FILE_BYTES:
            raise ValueError("Archive file exceeds 256 MiB")
        yield path.name, path.read_text(encoding="utf-8-sig")


def timestamp(value: str) -> str:
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        result = datetime.strptime(value, "%a %b %d %H:%M:%S %z %Y")
    if result.tzinfo is None:
        raise ValueError("Tweet timestamps must include a timezone")
    return result.astimezone(UTC).isoformat()


def normalize(row: dict, source_file: str, owner_id: str, auto_sources: tuple[str, ...]) -> Tweet:
    t = row.get("tweet", row)
    tweet_id = str(t.get("id_str") or t.get("id") or "")
    text = html.unescape(t.get("full_text") or t.get("text") or "").strip()
    if not tweet_id or not tweet_id.isdigit() or not t.get("created_at"):
        raise ValueError("Tweet requires a numeric ID and timestamp")
    user = t.get("user") or {}
    author = str(t.get("user_id_str") or t.get("user_id") or user.get("id_str") or user.get("id") or owner_id)
    reply = t.get("in_reply_to_status_id_str") or t.get("in_reply_to_status_id")
    quote = t.get("quoted_status_id_str") or t.get("quoted_status_id")
    is_rt = text.startswith("RT @") or bool(t.get("retweeted_status"))
    source = str(t.get("source") or "")
    kind = (
        "retweet"
        if is_rt
        else "quote"
        if quote or t.get("is_quote_status")
        else "reply"
        if reply
        else "tweet"
    )
    reason = None
    if author != owner_id:
        reason = "other_author"
    elif is_rt:
        reason = "retweet"
    elif any(x.casefold() in source.casefold() for x in auto_sources):
        reason = "automated_source"
    elif not re.search(r"[\w\u3040-\u30ff\u4e00-\u9fff]", re.sub(r"https?://\S+|@\w+", "", text)):
        reason = "noise"
    topics = [name for name, words in TOPICS.items() if any(w in text.casefold() for w in words)]
    return Tweet(
        id=tweet_id,
        timestamp=timestamp(t["created_at"]),
        text=text,
        author_id=author,
        reply_to=str(reply) if reply else None,
        quote_to=str(quote) if quote else None,
        kind=kind,
        source=source,
        source_file=source_file,
        topics=topics or ["other"],
        excluded_reason=reason,
    )


def import_archive(
    path: Path, store, owner_id: str | None = None, auto_sources: tuple[str, ...] = ("IFTTT", "twittbot")
) -> dict:
    # Two passes avoid retaining all raw archive files in memory.
    accounts = set()
    for name, raw in archive_files(path):
        if PurePosixPath(name).name in {"account.js", "account.json"}:
            for row in decode_archive(raw):
                account = row.get("account", row)
                if account.get("accountId"):
                    accounts.add(str(account["accountId"]))
    if owner_id is None:
        if len(accounts) != 1:
            raise ValueError("Specify --owner-id when the archive has no unique account.js owner")
        owner_id = accounts.pop()
    elif accounts and accounts != {owner_id}:
        raise ValueError("--owner-id does not match archive account.js")
    store.bind_owner(owner_id)
    counts = Counter(imported=0, duplicate=0, excluded=0, invalid=0)
    found = False
    for name, raw in archive_files(path):
        if PurePosixPath(name).name in {"account.js", "account.json"}:
            continue
        found = True
        rows = decode_archive(raw)
        if not isinstance(rows, list):
            raise ValueError("Expected an array of tweets")
        for row in rows:
            try:
                tweet = normalize(row, name, owner_id, auto_sources)
            except (ValueError, TypeError, KeyError, AttributeError):
                counts["invalid"] += 1
                continue
            added = store.add_tweet(tweet)
            counts["imported" if added else "duplicate"] += 1
            if added and tweet.excluded_reason:
                counts["excluded"] += 1
    if not found:
        raise ValueError("No tweet files found")
    return dict(counts)
