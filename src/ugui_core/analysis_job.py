"""Resumable local archive analysis with per-tweet checkpoints."""

from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from datetime import UTC, datetime

from .backends import BackendError, Completion, OpenAIBackend
from .models import Claim
from .persona import EXTRACTION_PROMPT, aggregate, extract_claims, style_profile


class AnalysisBackend(OpenAIBackend):
    """Structured extraction; Qwen's explicit non-thinking mode avoids wasting the JSON budget."""

    def complete(self, messages, *, temperature=0, max_tokens=4096):
        messages = [dict(m) for m in messages]
        if "qwen3" in self.settings.llm_model.lower():
            messages[-1]["content"] += "\n/no_think"
        data = self.post(
            self.settings.llm_url,
            "/chat/completions",
            {
                "model": self.settings.llm_model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": False,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "tweet_claims",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "properties": {
                                "claims": {
                                    "type": "array",
                                    "items": {**Claim.model_json_schema(), "additionalProperties": False},
                                }
                            },
                            "required": ["claims"],
                            "additionalProperties": False,
                        },
                    },
                },
            },
        )
        try:
            choice = data["choices"][0]
            if choice.get("finish_reason") == "length":
                raise ValueError("Extraction exceeded output limit")
            content = choice["message"]["content"]
            if not isinstance(content, str) or not content.strip():
                raise ValueError("Empty extraction")
            return Completion(content)
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise BackendError("Extraction response incomplete; retry with a smaller batch") from exc

    @property
    def run_key(self):
        value = f"analysis-v2|{self.settings.llm_url}|{self.settings.llm_model}|{EXTRACTION_PROMPT}"
        return hashlib.sha256(value.encode()).hexdigest()


def initialize(store):
    with store.connect() as db:
        db.execute("""CREATE TABLE IF NOT EXISTS analysis_results (
            run_key TEXT NOT NULL, tweet_id TEXT NOT NULL REFERENCES tweets(id),
            claims TEXT, error TEXT, updated_at TEXT NOT NULL, PRIMARY KEY(run_key,tweet_id))""")


def save_result(store, run_key, tweet_id, claims=None, error=None):
    with store.connect() as db:
        db.execute(
            "INSERT OR REPLACE INTO analysis_results VALUES (?,?,?,?,?)",
            (
                run_key,
                tweet_id,
                json.dumps([c.model_dump() for c in claims], ensure_ascii=False)
                if claims is not None
                else None,
                error,
                datetime.now(UTC).isoformat(),
            ),
        )


def checkpoint_counts(store, run_key):
    with store.connect() as db:
        row = db.execute(
            "SELECT COUNT(*), SUM(error IS NULL) FROM analysis_results WHERE run_key=?", (run_key,)
        ).fetchone()
    return int(row[1] or 0), int(row[0] - (row[1] or 0))


def batches(tweets, size=8, chars=2800):
    group, length = [], 0
    for tweet in tweets:
        if group and (len(group) >= size or length + len(tweet.text) > chars):
            yield group
            group, length = [], 0
        group.append(tweet)
        length += len(tweet.text)
    if group:
        yield group


def analyze(store, llm, run_key=None, *, batch_size=8, progress=None, retry_delay=5):
    """A successful checkpoint means valid JSON was evaluated, including an empty claim list.

    Failures are recorded and retried on the next invocation, never counted as analyzed.
    Only a completely processed corpus replaces the production persona.
    """
    initialize(store)
    run_key = run_key or llm.run_key
    tweets = store.tweets()
    with store.connect() as db:
        done_ids = {
            row[0]
            for row in db.execute(
                "SELECT tweet_id FROM analysis_results WHERE run_key=? AND error IS NULL", (run_key,)
            )
        }
    # Recent-first gives useful checkpoints early, but every eligible tweet is processed.
    pending = [t for t in reversed(tweets) if t.id not in done_ids]
    started, initial_done = time.monotonic(), len(done_ids)
    status = {}

    def publish(state="running"):
        done, failed = checkpoint_counts(store, run_key)
        elapsed = time.monotonic() - started
        delta = done - initial_done
        status.update(
            state=state,
            run_key=run_key,
            total=len(tweets),
            completed=done,
            failed=failed,
            remaining=len(tweets) - done,
            elapsed_seconds=round(elapsed),
            tweets_per_minute=round(60 * delta / elapsed, 2) if elapsed else 0,
            eta_seconds=round((len(tweets) - done) * elapsed / delta) if delta > 0 else None,
            updated_at=datetime.now(UTC).isoformat(),
        )
        store.save_snapshot("analysis_status", status)
        if progress:
            progress(dict(status))

    def process(group):
        try:
            if len(group) == 1 and len(group[0].text) > 2800:
                tweet = group[0]
                extracted = []
                for start in range(0, len(tweet.text), 2400):
                    segment = tweet.model_copy(update={"text": tweet.text[start : start + 2800]})
                    extracted.extend(extract_claims([segment], llm))
            else:
                extracted = extract_claims(group, llm)
            by_id = {t.id: [] for t in group}
            for c in extracted:
                by_id[c.tweet_id].append(c)
            for t in group:
                save_result(store, run_key, t.id, claims=by_id[t.id])
        except (BackendError, ValueError, TypeError):
            if len(group) > 1:
                midpoint = len(group) // 2
                process(group[:midpoint])
                process(group[midpoint:])
            else:
                # One retry handles transient failures without losing earlier checkpoints.
                time.sleep(retry_delay)
                try:
                    extracted = extract_claims(group, llm)
                    save_result(store, run_key, group[0].id, claims=extracted)
                except (BackendError, ValueError, TypeError):
                    save_result(store, run_key, group[0].id, error="backend_or_invalid_json")

    publish()
    consecutive_failed = 0
    for group in batches(pending, batch_size):
        before, _ = checkpoint_counts(store, run_key)
        process(group)
        after, _ = checkpoint_counts(store, run_key)
        consecutive_failed = consecutive_failed + len(group) if after == before else 0
        publish()
        if consecutive_failed >= 16:
            publish("needs_attention")
            raise BackendError(
                "Repeated extraction failures; checkpoints saved. Check the local model and resume."
            )
    done, failed = checkpoint_counts(store, run_key)
    if failed or done != len(tweets):
        publish("needs_attention")
        raise BackendError("Some tweets failed analysis; rerun to retry only unfinished records")
    claims = []
    with store.connect() as db:
        for row in db.execute(
            "SELECT claims FROM analysis_results WHERE run_key=? AND error IS NULL", (run_key,)
        ):
            claims.extend(Claim.model_validate(c) for c in json.loads(row[0]))
    profile = {
        "schema_version": 1,
        "evidence_count": len(tweets),
        "style": style_profile(tweets),
        "patterns": aggregate(claims, tweets),
        "confidence_note": "Heuristic support score, not a calibrated probability or permanent identity.",
        "as_of": max((t.timestamp for t in tweets), default=None),
        "analysis": {
            "scope": "all_eligible_tweets",
            "completed": done,
            "run_key": run_key,
            "claim_categories": dict(Counter(c.category for c in claims)),
        },
    }
    store.save_persona(profile, claims)
    publish("completed")
    return profile
