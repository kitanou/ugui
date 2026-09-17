from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

from .backends import BackendError, OpenAIBackend, OpenAIEmbedder
from .config import Settings
from .evaluation import evaluate
from .persona import build_persona
from .retrieval import LocalRetriever
from .store import Store


def write_private(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    path.chmod(0o600)


def main():
    parser = argparse.ArgumentParser(prog="ugui")
    parser.add_argument("--data-dir", type=Path, help="Override UGUI_DATA_DIR")
    commands = parser.add_subparsers(dest="command", required=True)
    imp = commands.add_parser("import", help="Import an X ZIP, extracted directory, or tweets.js/json")
    imp.add_argument("archive", type=Path)
    imp.add_argument("--owner-id")
    imp.add_argument("--auto-source", action="append", default=["IFTTT", "twittbot"])
    commands.add_parser("stats")
    analysis = commands.add_parser("analyze", help="Analyze all eligible tweets with resumable checkpoints")
    analysis.add_argument("--batch-size", type=int, default=8)
    commands.add_parser("analysis-status", help="Show durable full-analysis progress")
    idx = commands.add_parser("index", help="Build embeddings with the configured model")
    idx.add_argument("--rebuild", action="store_true")
    search = commands.add_parser("search")
    search.add_argument("query")
    persona = commands.add_parser("persona")
    persona.add_argument("--extract", action="store_true", help="Extract claims using the configured LLM")
    serve = commands.add_parser("serve")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8011)
    ev = commands.add_parser("evaluate")
    ev.add_argument("cases", type=Path, help="JSON array of tweet_id + situation objects")
    ev.add_argument("--judge", action="store_true", help="Score six axes using the configured LLM")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    try:
        settings = Settings.from_env()
        if args.data_dir:
            from dataclasses import replace

            settings = replace(settings, data_dir=args.data_dir)
        if args.command == "serve":
            if args.host not in {"127.0.0.1", "::1", "localhost"} and not settings.api_key:
                raise ValueError("Set UGUI_API_KEY before binding a non-loopback address")
            import uvicorn

            from .api import create_app

            uvicorn.run(create_app(settings), host=args.host, port=args.port, access_log=False)
            return
        store = Store(settings.data_dir / "ugui.sqlite3")
        embedder = OpenAIEmbedder(settings) if settings.embedding_model else None
        if args.command == "import":
            from .archive import import_archive

            result = import_archive(args.archive, store, args.owner_id, tuple(args.auto_source))
        elif args.command == "stats":
            result = {
                "stored": len(store.tweets(True)),
                "eligible": len(store.tweets()),
                "claims": len(store.claims()),
                "persona_generated": bool(store.snapshot()),
            }
        elif args.command == "index":
            result = {"indexed": LocalRetriever(store, embedder).index(args.rebuild)}
        elif args.command == "analysis-status":
            result = store.snapshot("analysis_status") or {"state": "not_started"}
        elif args.command == "analyze":
            from .analysis_job import AnalysisBackend, analyze

            if not 1 <= args.batch_size <= 12:
                raise ValueError("batch-size must be between 1 and 12")
            profile = analyze(
                store,
                AnalysisBackend(settings),
                batch_size=args.batch_size,
                progress=lambda status: print(json.dumps(status), flush=True),
            )
            target = settings.data_dir / "persona" / "profile.json"
            write_private(target, profile)
            result = {"state": "completed", "profile": str(target), "patterns": len(profile["patterns"])}
        elif args.command == "search":
            result = {
                "mode": "semantic" if embedder else "lexical",
                "results": [t.model_dump() for t in LocalRetriever(store, embedder).search(args.query)],
            }
        elif args.command == "persona":
            profile = build_persona(store, OpenAIBackend(settings) if args.extract else None)
            target = settings.data_dir / "persona" / "profile.json"
            write_private(target, profile)
            result = {
                "profile": str(target),
                "patterns": len(profile["patterns"]),
                "supported": sum(p["status"] == "supported" for p in profile["patterns"]),
            }
        elif args.command == "evaluate":
            llm = OpenAIBackend(settings)
            report = evaluate(
                store, llm, json.loads(args.cases.read_text()), embedder, llm if args.judge else None
            )
            target = settings.data_dir / "evaluation" / "report.json"
            write_private(target, report)
            result = {
                "report": str(target),
                "training_count": report["training_count"],
                "holdout_count": report["holdout_count"],
            }
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except (ValueError, OSError, BackendError) as exc:
        # No provider response text or raw tweet content is printed on failures.
        parser.exit(1, f"UGUI error: {exc}\n")


if __name__ == "__main__":
    main()
