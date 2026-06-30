# KnowOps

An agentic RAG system that answers natural language questions about Jira tickets and Confluence pages — built to demonstrate and fix a real-world production bug: a Milvus retriever that returned **stale documents** over current ones because it used cosine similarity alone.

---

## The Bug

A production RAG system ranked a 2-year-old Confluence page above a 1-month-old one for the same query because both had similar cosine similarity scores. The older page had outdated configuration values. Users got wrong answers.

**Root cause:** semantic similarity has no concept of time. A stale document and a current one on the same topic look identical to a vector search.

**Fix:** recency-weighted reranking — combine cosine similarity with an exponential decay on `updated_date`:

```
recency  = exp(−λ × days_since_update)
combined = α × semantic_score + (1 − α) × recency
```

Default: `α = 0.7`, `λ = 0.005` (half-life ≈ 139 days). The before/after difference is demonstrated with deliberately constructed "trap pairs" — document sets where a naive retriever always picks the wrong answer.

---

## Architecture

```
User Query
    │
    ▼
Planner Agent          ← Claude Sonnet (OpenRouter)
Decomposes query into sub-tasks; classifies temporal vs. factual retrieval
    │
    ▼
Retriever              ← Milvus (HNSW index) + Ollama bge-m3
Hybrid search: vector similarity + scalar metadata filters
    │
    ▼
Reranker               ← recency-weighted scoring (pure Python)
combined = α·semantic + (1−α)·exp(−λ·days_old)
    │
    ▼
Answering Agent        ← Claude Sonnet (OpenRouter)
Synthesises answer with source citations; refuses when context is empty
    │
    ▼
Answer + Sources
```

**Stack:** Python · Milvus · PostgreSQL · Ollama (bge-m3, CPU) · OpenRouter (Claude Sonnet)  
All services run locally via Docker Compose — no GPU required.

---

## Repository Layout

```
KnowOps/
├── docker-compose.yml          # Milvus + Postgres + Ollama (CPU)
├── requirements.txt
├── .env.example
│
├── data/
│   ├── jira_tickets.json       # 60 hand-crafted Jira tickets
│   ├── confluence_pages.json   # 60 hand-crafted Confluence pages
│   └── trap_manifest.json      # 10 trap pair definitions
│
├── knowops/
│   ├── embedder.py             # Ollama bge-m3 client
│   ├── schema.py               # Milvus collection schema + HNSW index params
│   └── db.py                   # PostgreSQL tables (SQLAlchemy)
│
└── scripts/
    ├── setup_collection.py     # Create Milvus collection + indexes (run once)
    ├── ingest.py               # Chunk → embed → upsert pipeline
    └── generate_dataset.py     # Phase 2: generate 10K Jira + 2K Confluence
```

---

## Quick Start

### 1. Prerequisites

- Docker + Docker Compose
- Python 3.11+
- 16 GB RAM minimum (32 GB recommended — bge-m3 uses ~14 GB peak)
- An [OpenRouter](https://openrouter.ai/) API key

### 2. Start the stack

```bash
git clone https://github.com/your-org/knowops.git
cd knowops

cp .env.example .env
# Edit .env — set OPENROUTER_API_KEY

docker compose up -d
```

Milvus takes ~60 seconds on first start. The `ollama-model-init` container then pulls `bge-m3` (~1.5 GB). Watch progress:

```bash
docker logs -f knowops-ollama-init
```

### 3. Set up the Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 4. Create the Milvus collection and ingest data

```bash
python scripts/setup_collection.py
python scripts/ingest.py --source all
```

### 5. (Phase 2) Generate the large-scale dataset

```bash
python scripts/generate_dataset.py          # → data/jira_tickets_large.json etc.
python scripts/ingest.py --source all --data-dir data
```

---

## Trap Pairs

The dataset includes deliberate adversarial test cases. Each trap group contains:

- An **outdated document** with wrong information (low `updated_date`)
- A **correct document** on the same topic with current information (high `updated_date`)

Both documents have high cosine similarity. Only recency-weighted reranking returns the correct one.

**Example trap group — `trap-milvus-index-type`:**

| Doc | `updated_date` | Says |
|-----|---------------|------|
| CONF-2001 ❌ | 2024-09-28 | Use IVF_FLAT, nlist=32, timeout=60s |
| CONF-2049 ✅ | 2025-12-20 | Use HNSW, M=16, ef_construction=200, timeout=10s |

Query: *"What is the recommended Milvus index type?"*  
- Baseline (semantic only): may return CONF-2001  
- Fixed (recency-weighted): always returns CONF-2049

The full set of 10 trap groups is defined in `data/trap_manifest.json`.

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| Single Milvus collection for Jira + Confluence | `source_type` scalar field handles filtering; avoids cross-collection search complexity |
| HNSW over IVF_FLAT | IVF_FLAT with low `nlist` causes O(n) search — the latency bug this project documents |
| bge-m3 over text-embedding-ada-002 | Runs locally on CPU, no API cost, no data leaving the host |
| Formula-based reranker (not ML) | Interpretable, tunable with two parameters, no training data required |
| `α = 0.7` default | 70% semantic / 30% recency; tunable via `PIPELINE_ALPHA` env var |

---

## Environment Variables

See [.env.example](.env.example) for the full list. Key variables:

| Variable | Default | Description |
|---|---|---|
| `OPENROUTER_API_KEY` | — | Required for LLM agent calls |
| `PIPELINE_ALPHA` | `0.7` | Semantic weight in reranker (0–1) |
| `PIPELINE_LAMBDA` | `0.005` | Recency decay constant |
| `EMBEDDING_BATCH_SIZE` | `8` | Docs per Ollama call (max 8 on 32 GB) |
| `RETRIEVER_TOP_K` | `20` | Milvus candidates before reranking |
