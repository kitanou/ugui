# Architecture / Issue #1 mapping

```text
X ZIP / JS / JSON
  -> archive.py: owner verification, normalize UTC, classify, filter
  -> store.py: immutable-by-ID tweet evidence (SQLite)
       -> retrieval.py: lexical or local embedding / cosine search
       -> persona.py: LLM claims with exact evidence quotes
            -> repeated patterns + time + conservative support score
            -> persona/style + semantic memory snapshot
  -> engine.py: persona + relevant memory + evidence + conversation
  -> backends.py: replaceable LLM
  -> api.py: OpenAI-compatible text chat
```

| Phase | Implementation | Validation |
|---|---|---|
| 1 Core | package, environment config, safe logging, backend/retrieval protocols, health/models/chat | API contracts/auth/config tests |
| 2 Import | ZIP/directory/JS/JSON/multipart, owner binding, normalization, exclusion reasons | fixture import, idempotence, invalid data, unchanged source |
| 3 Evidence | SQLite original text and provenance, embedding index and cosine search | lexical Japanese, semantic fixture, incomplete index |
| 4 Persona | LLM extracts claims, quote verification, conservative aggregation, style statistics | fabricated evidence rejection, singleton/repetition/conflict/rebuild |
| 5 Response | evidence and derived memory + persona + style + supplied conversation | prompt provenance and normal/SSE responses |
| 6 Evaluation | temporal split, isolated training store, six optional judge axes | held-out/future/persona leakage tests |

## Boundaries

`LLM.complete`, `Embedder.embed`, and `Retriever.search` are the adapter boundaries.
The default HTTP adapter targets OpenAI-compatible servers. There are no NAHO imports or shared paths.
Model runtime installation, fine-tuning, LoRA, voice, and autonomous inter-Core conversations are outside this MVP.
The specified GitHub repository is named `ugui`; it hosts the independently named `ugui-core` package without renaming the remote.

`/health` is liveness, not model readiness. `model=ugui` is the public model ID; the actual provider model comes from config.
SSE is currently buffered. Non-text content/tool calls are unsupported. NAHO-specific schema parity needs the NAHO contract.

## Data semantics

Tweets retain source IDs, UTC timestamps, owner ID, reply/quote links and source metadata.
Excluded rows remain auditable but never enter retrieval, extraction or style analysis.
Duplicate text cannot inflate personality evidence, even when reposted under different IDs.
A claim cannot invent an evidence ID or quote; semantic interpretation still requires review.
Subjects/attributes/values are normalized by the model and casefolded, not a universal ontology.
For a given subject/attribute, distinct values from non-overlapping evidence are conservatively treated as possible conflicts. Values co-supported by the same tweet may coexist.
This can understate multi-valued preferences. Evidence remains available for later reprocessing.
The `as_of` timestamp bounds knowledge and does not assert that historical preferences hold today.

The semantic memory is the derived claim/pattern snapshot. Episodic memory is dated tweet evidence.
There is intentionally no durable chat memory in this MVP: callers supply the active conversation.
System instructions request treating retrieved text as data. This reduces prompt injection risk but cannot guarantee model compliance.
The API has no execution tools or credentials available to model output.

## Practical limits

Importer processes one archive member at a time; each member is parsed in memory and capped at 256 MiB.
SQLite uses transactions, WAL, parameterized values and per-operation connections for API worker threads.
Import is resumable by tweet ID; malformed archive files can leave earlier successfully imported files in the DB.
Vector retrieval is O(N*d) per request and loads eligible records in memory. It is aimed at personal MVP use,
not a multi-user service. For larger archives, implement an indexed `Retriever` adapter.
Persona extraction is batched; malformed model JSON aborts without replacing prior claims.
Extraction must be rerun after imports to reflect new evidence. The response engine rejects outdated snapshots; concurrent imports abort persona replacement atomically. No automatic background LLM calls run.
Evaluation excludes all tweets at or after the earliest holdout timestamp. Situations must be independently written.
Judge calls see held-out answers only after generation; their scores are subjective, not proof of identity fidelity.
