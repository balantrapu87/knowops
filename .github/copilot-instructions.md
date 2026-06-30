# Copilot instructions — KnowOps

Agentic RAG for Jira + Confluence Q&A. Its reason to exist: **fix date-aware
retrieval** — semantic-only search let stale docs outrank fresher correct ones.
The fix is recency-weighted hybrid scoring, in code (not a prompt).

## Architecture
`question → Planner → Retriever → Reranker → Answering → answer`
- **Planner** (`knowops/agents/planner.py`) classifies `time_sensitivity` (high/medium/low).
- That label picks a **freshness profile** in `configs/pipeline.yaml`.
- **Retriever** (`retriever.py`) runs vector search, applies a **relevance floor**, then scores
  `hybrid_score = semantic_weight·semantic + freshness_weight·exp(-age_days/decay_days)`.
- **Reranker** (`reranker.py`) picks top-k by `hybrid_score` (code decides ranking) + freshness warning.
- **Answering** (`answering.py`) writes a grounded, cited answer.

## Core invariants — do not break
- **The fix lives in code**: `knowops/freshness.py` (`hybrid_score`) + the relevance floor in
  `RetrieverAgent`. Within a trap group the *newer* doc must outrank the older one.
- **Freshness is config-driven**: tune `configs/pipeline.yaml` (weights, decay, `relevance_floor_ratio`),
  never hard-code. Planner's `time_sensitivity` selects the profile.
- **Recency is soft scoring, not a hard date filter** — many correct docs are 6–14 months old, so
  `date_filter_days` defaults to null. Don't add hard date cut-offs that can exclude current docs.
- **`retrieve_baseline` must stay semantic-only** (no freshness, no floor) — it reproduces the bug for the demo.

## Dual mode
Everything runs **offline** (deterministic, no services) or **live** (Milvus + Ollama bge-m3 +
OpenRouter Claude on the server). Set `KNOWOPS_OFFLINE=1` or pass `offline=True`.
- Offline: lexical feature-hash embeddings (`embedder.embed_offline`), in-memory brute-force search
  (`search.OfflineBackend`), deterministic agent logic. `LLMClient.complete()` is never called offline.
- Live metadata source is switchable via `METADATA_BACKEND` (`milvus` default, or `postgres` to join
  the metadata table — see `search.MilvusPostgresBackend`). Offline ignores it.
- Add an offline path for any new agent/backend. All config flows through `knowops/config.SETTINGS` —
  don't read `os.environ` directly elsewhere.

## Run / test
```bash
.venv/bin/python -m pytest tests/ -q                       # full suite, no services
KNOWOPS_OFFLINE=1 .venv/bin/python scripts/demo.py          # baseline vs fixed (≈2/10 vs 10/10)
KNOWOPS_OFFLINE=1 .venv/bin/python scripts/ask.py "..."     # ask a question
```
Live setup: `docker compose up -d`, `python scripts/setup_collection.py`, `python scripts/ingest.py --source all`.

## Conventions
- Small, interview-explainable functions; comment only the non-obvious. Portfolio project, not enterprise.
- Ground-truth lives in `data/trap_manifest.json` (10 trap groups). `tests/test_retrieval.py` asserts the
  fixed path returns each `correct_document_id` and ranks it above the `outdated_document_ids`.
- Keep the pre-existing embedder/ingest tests green.
