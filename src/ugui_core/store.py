from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path

from .models import Claim, Tweet


class Store:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with self.connect() as db:
            db.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS tweets(
                    id TEXT PRIMARY KEY, timestamp TEXT NOT NULL, content_hash TEXT NOT NULL,
                    excluded_reason TEXT, payload TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS tweets_hash ON tweets(content_hash);
                CREATE TABLE IF NOT EXISTS claims(
                    key TEXT PRIMARY KEY, tweet_id TEXT NOT NULL REFERENCES tweets(id), payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS vectors(
                    tweet_id TEXT NOT NULL REFERENCES tweets(id), model TEXT NOT NULL, payload TEXT NOT NULL,
                    PRIMARY KEY(tweet_id, model));
                CREATE TABLE IF NOT EXISTS snapshots(name TEXT PRIMARY KEY, payload TEXT NOT NULL);
            """)
        path.chmod(0o600)

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.execute("PRAGMA foreign_keys=ON")
        try:
            with db:
                yield db
        finally:
            db.close()

    def bind_owner(self, owner_id: str):
        with self.connect() as db:
            db.execute("INSERT OR IGNORE INTO metadata VALUES ('owner_id', ?)", (owner_id,))
            current = db.execute("SELECT value FROM metadata WHERE key='owner_id'").fetchone()[0]
            if current != owner_id:
                raise ValueError("This data directory belongs to another archive owner")

    def add_tweet(self, tweet: Tweet) -> bool:
        digest = hashlib.sha256(tweet.text.encode()).hexdigest()
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if db.execute("SELECT 1 FROM tweets WHERE id=?", (tweet.id,)).fetchone():
                return False
            if (
                not tweet.excluded_reason
                and db.execute(
                    "SELECT 1 FROM tweets WHERE content_hash=? AND excluded_reason IS NULL", (digest,)
                ).fetchone()
            ):
                tweet.excluded_reason = "duplicate_text"
            db.execute(
                "INSERT INTO tweets VALUES (?,?,?,?,?)",
                (tweet.id, tweet.timestamp, digest, tweet.excluded_reason, tweet.model_dump_json()),
            )
        return True

    def tweets(self, include_excluded=False) -> list[Tweet]:
        with self.connect() as db:
            where = "" if include_excluded else "WHERE excluded_reason IS NULL"
            return [
                Tweet.model_validate_json(r[0])
                for r in db.execute(f"SELECT payload FROM tweets {where} ORDER BY timestamp, id")
            ]

    def replace_claims(self, claims: list[Claim]):
        with self.connect() as db:
            self._write_claims(db, claims)

    @staticmethod
    def _write_claims(db, claims):
        db.execute("DELETE FROM claims")
        for claim in claims:
            raw = claim.model_dump_json()
            key = hashlib.sha256(raw.encode()).hexdigest()
            db.execute("INSERT OR IGNORE INTO claims VALUES (?,?,?)", (key, claim.tweet_id, raw))

    def evidence_revision(self) -> int:
        # Evidence is append-only; eligible count therefore identifies its revision.
        with self.connect() as db:
            return db.execute("SELECT COUNT(*) FROM tweets WHERE excluded_reason IS NULL").fetchone()[0]

    def save_persona(self, profile: dict, claims: list[Claim] | None):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            count = db.execute("SELECT COUNT(*) FROM tweets WHERE excluded_reason IS NULL").fetchone()[0]
            if count != profile["evidence_count"]:
                raise ValueError("Archive changed during extraction; rerun ugui persona --extract")
            if claims is not None:
                self._write_claims(db, claims)
            db.execute(
                "INSERT OR REPLACE INTO snapshots VALUES ('persona', ?)",
                (json.dumps(profile, ensure_ascii=False),),
            )

    def claims(self) -> list[Claim]:
        with self.connect() as db:
            return [
                Claim.model_validate_json(r[0]) for r in db.execute("SELECT payload FROM claims ORDER BY key")
            ]

    def save_snapshot(self, name: str, payload: dict):
        with self.connect() as db:
            db.execute(
                "INSERT OR REPLACE INTO snapshots VALUES (?,?)",
                (name, json.dumps(payload, ensure_ascii=False)),
            )

    def snapshot(self, name="persona") -> dict:
        with self.connect() as db:
            row = db.execute("SELECT payload FROM snapshots WHERE name=?", (name,)).fetchone()
        return json.loads(row[0]) if row else {}

    def save_vectors(self, model: str, entries: list[tuple[str, list[float]]]):
        with self.connect() as db:
            db.executemany(
                "INSERT OR REPLACE INTO vectors VALUES (?,?,?)",
                [(tid, model, json.dumps(vector)) for tid, vector in entries],
            )

    def vectors(self, model: str) -> dict[str, list[float]]:
        with self.connect() as db:
            return {
                r[0]: json.loads(r[1])
                for r in db.execute("SELECT tweet_id,payload FROM vectors WHERE model=?", (model,))
            }
