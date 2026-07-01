# KnowOps

**Agentic RAG for enterprise service management (Jira + Confluence Q&A) — and a fix
for a real retrieval bug: semantic-only search kept surfacing stale documents over
fresher, correct ones.**

KnowOps reproduces that failure and fixes it with **recency-weighted hybrid retrieval**,
applied as real code and driven by config — not a prompt hack:

```
hybrid_score = semantic_weight · cosine_similarity
             + freshness_weight · exp(−age_days / decay_days)
```

The whole system runs two ways: **live** (Milvus + Ollama + OpenRouter on a server) or
fully **offline** (deterministic, zero infrastructure) for tests and demos.

---

## The problem

A production RAG system answering service-manager questions over Jira tickets and
Confluence pages retrieved purely by semantic similarity. It ignored the `updated_date` /
`created_date` scalar fields. When a topic had two documents — an **old** one with
now-wrong guidance and a **new** one with the correction — the old document often had
*equal or higher* lexical/semantic overlap with the query and **ranked above** the
current one. Service managers got confidently wrong answers (deprecated configs, outdated
rate limits, wrong escalation paths).

## Diagnosis

The corpus contains 10 **trap groups** (`data/trap_manifest.json`): each is a pair of
documents on one topic where the *correct* answer is the **newer** doc. Running the
baseline (semantic-only) retriever over those topics reproduces the bug — the stale doc
wins most of the time. Root cause: ranking used cosine similarity alone; recency was
never a ranking signal, and a recently-corrected document had no way to overtake an
older, lexically-similar one.

## Architecture

```
question
   │
   ▼
┌──────────┐   time_sensitivity     ┌───────────────────────────┐
│ Planner  │ ─────────────────────► │ configs/pipeline.yaml      │
│ (LLM)    │   high / medium / low  │  freshness profile         │
└────┬─────┘                        └───────────────────────────┘
     │ plan                                      │ weights, decay, floor
     ▼                                           ▼
┌──────────┐   vector search    ┌────────────────────────────────────┐
│Retriever │ ─────────────────► │ relevance floor → hybrid_score      │
│ (LLM+code)│   (Milvus / offline)│ (src/knowops/freshness.py)          │
└────┬─────┘                     └────────────────────────────────────┘
     │ scored candidates
     ▼
┌──────────┐  top-k by hybrid   ┌──────────┐  grounded + cited
│ Reranker │ ─────────────────► │Answering │ ───────────────────►  answer
│(code+LLM)│  + freshness warn  │  (LLM)   │
└──────────┘                    └──────────┘
```

**Agentic pipeline** — Planner → Retriever → Reranker → Answering, each loading its prompt
from `prompts/*.md`. The LLM plans, optimizes the query, explains, and writes prose; the
**ranking decision is code** so correctness never depends on LLM judgement.

**Stack**

| Layer       | Live                                   | Offline (tests/demo)                |
|-------------|----------------------------------------|-------------------------------------|
| Vector DB   | Milvus (HNSW, COSINE)                  | in-memory brute-force cosine        |
| Embeddings  | bge-m3 via Ollama (CPU, 1024-dim)     | deterministic lexical feature-hash  |
| LLM agents  | Claude Sonnet via OpenRouter          | deterministic rule-based fallbacks  |
| Metadata    | PostgreSQL (SQLAlchemy)               | — (reads JSON directly)             |

---

## The fix

Three pieces, all tunable from `configs/pipeline.yaml` without touching code:

1. **Freshness scoring** (`src/knowops/freshness.py`) — an exponential recency bonus blended
   with semantic similarity. Because each trap's correct doc is the *newer* one, a non-zero
   `freshness_weight` lets it overtake a stale doc that's marginally more similar.

2. **Config-driven, Planner-selected profiles** — the Planner classifies each question's
   `time_sensitivity` (high/medium/low); that label selects a freshness profile (weights +
   `decay_days`) which the Retriever and Reranker apply. "Date awareness" is configuration.

3. **Relevance floor** — recency reranking only applies to candidates scoring at least
   `relevance_floor_ratio × top_semantic_score`. This stops a *fresh-but-irrelevant*
   document (e.g. a doc edited yesterday on an unrelated topic) from being promoted over a
   genuinely relevant one. Recency breaks ties **among relevant docs**, it doesn't override
   relevance.

> **Why not a hard date filter?** Several *current* documents in this corpus are 6–14 months
> old. A `WHERE updated_date > cutoff` filter would wrongly drop them. So recency is a **soft
> score**, and `date_filter_days` defaults to `null`.

---

## Results

Full suite runs offline with no services:

```bash
.venv/bin/python -m pytest tests/ -q
# 143 passed
```

Before/after over the 10 trap topics (`scripts/demo.py`):

```
Baseline correct: 2/10  |  Fixed correct: 10/10
Traps where baseline returned a STALE document: 7/10
```

The fix returns the correct (newer) document as #1 for **all 10** traps and ranks it above
every stale doc. The score breakdown shows *why* — the stale doc often has the **higher**
semantic score but loses on the blend:

```
trap-milvus-index-type   Query: Recommended Milvus index type for KnowOps
  BASELINE (semantic-only)  CONF-2001  updated 2024-09-28   ❌ STALE
  FIXED   (hybrid+recency)  CONF-2049  updated 2025-12-20   ✅ CORRECT

  Role     Doc ID     Semantic  Freshness  Hybrid
  correct  CONF-2049     0.438      0.201   0.332   ← wins on recency
  stale    CONF-2001     0.510      0.005   0.282   ← higher semantic, but stale
```

---

## Running it

### Offline (no Docker, no keys — recommended first run)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

KNOWOPS_OFFLINE=1 .venv/bin/python scripts/demo.py                 # baseline vs fixed, side by side
KNOWOPS_OFFLINE=1 .venv/bin/python scripts/ask.py "What is the current API rate limit?" --verbose
.venv/bin/python -m pytest tests/ -q                                # 143 tests (hermetic, no env needed)
```

### Live (Milvus + Ollama + OpenRouter)

```bash
cp .env.example .env          # add OPENROUTER_API_KEY; leave KNOWOPS_OFFLINE=0
docker compose up -d          # Milvus, Postgres, Ollama (first run pulls bge-m3 ~1.5 GB)

python scripts/setup_collection.py    # one-time: create Milvus collection + indexes
python scripts/ingest.py --source all # chunk → embed → upsert (Milvus + Postgres)

python scripts/demo.py                # now runs against the real stack
python scripts/ask.py                 # interactive REPL
```

---

## Configuration

Freshness behaviour lives in [`configs/pipeline.yaml`](configs/pipeline.yaml) as per-intent
profiles; secrets and endpoints live in [`.env`](.env.example).

| Where | Setting | Default | What it does |
|-------|---------|---------|--------------|
| yaml  | `freshness_profiles.{high,medium,low}` | — | `semantic_weight`, `freshness_weight`, `decay_days`, `date_filter_days` per time-sensitivity |
| yaml  | `relevance_floor_ratio` | `0.7` | min semantic (× top) to qualify for recency reranking |
| yaml  | `retriever_top_k` / `reranker_top_k` | `20` / `5` | candidates fetched / kept |
| yaml  | `freshness_warning_days` | `180` | answers flag sources older than this |
| env   | `KNOWOPS_OFFLINE` | `0` | `1` = no external services |
| env   | `OPENROUTER_API_KEY` / `OPENROUTER_MODEL` | — / `anthropic/claude-sonnet-4` | live LLM agents |

---

## Project layout

```
KnowOps/
├── configs/pipeline.yaml        # freshness profiles + pipeline tunables
├── prompts/                     # planner / retriever / reranker / answering system prompts
├── src/
│   └── knowops/                 # unified python package
│       ├── config.py            # .env + yaml → Settings + FreshnessProfile; offline flag
│       ├── freshness.py         # freshness_score / hybrid_score (the fix, in code)
│       ├── embedder.py          # Ollama bge-m3 + deterministic offline embeddings
│       ├── search.py            # Milvus + offline brute-force backends
│       ├── llm.py               # OpenRouter client
│       ├── pipeline.py          # orchestrator + baseline/fixed retrieval
│       └── agents/              # Planner, RetrieverAgent, Reranker, Answering
├── scripts/
│   ├── setup_collection.py      # create Milvus collection (live)
│   ├── ingest.py                # ingest data → Milvus + Postgres (live)
│   ├── generate_dataset.py      # Phase 2: scale dataset generator
│   ├── demo.py                  # before/after comparison
│   └── ask.py                   # interactive Q&A CLI
├── data/                        # 60 Jira + 60 Confluence + 10 trap groups
├── tests/                       # 143 tests (run offline)
└── docker-compose.yml           # Milvus + Postgres + Ollama
```

---

## Trap pairs

Each of the 10 groups in `data/trap_manifest.json` pairs an outdated document with its
current correction on the same topic. Both look equally relevant to a plain search; only
recency tells them apart.

| Doc | Updated | Content |
|-----|---------|---------|
| CONF-2001 ❌ | 2024-09-28 | Use IVF_FLAT index, timeout 60 s |
| CONF-2049 ✅ | 2025-12-20 | Use HNSW index, timeout 10 s |

Query *"What Milvus index should I use?"* → semantic-only may return CONF-2001; with
recency scoring it always returns CONF-2049.
