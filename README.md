# KnowOps
Production-grade agentic RAG system for enterprise service management.
Fixing date-aware retrieval in Milvus-based pipelines.

Ask questions about Jira tickets and Confluence pages in plain English. Demonstrates and fixes a real bug: a search system that kept returning outdated docs because it only compared meaning, not age.

**The fix** — score each result by both relevance *and* how recently it was updated:

```
score = 0.7 × similarity + 0.3 × exp(−0.005 × days_since_update)
```

---

## How it works

```
Question → Planner (Claude) → Retriever (Milvus) → Reranker → Answering Agent (Claude) → Answer
```

- **Planner** — breaks down the question  
- **Retriever** — finds the top 20 candidates by vector similarity (Milvus + bge-m3)  
- **Reranker** — re-scores by similarity + recency, keeps top 5  
- **Answering Agent** — writes the answer with source citations  

Stack: Python, Milvus, PostgreSQL, Ollama (CPU), Claude via OpenRouter. No GPU needed.

---

## Requirements

- Docker + Docker Compose
- Python 3.11+
- 16 GB RAM (32 GB recommended — the embedding model peaks at ~14 GB)
- [OpenRouter](https://openrouter.ai/) API key

---

## Setup

```bash
git clone https://github.com/balantrapu87/knowops.git && cd knowops
cp .env.example .env          # add your OPENROUTER_API_KEY
docker compose up -d          # starts Milvus, Postgres, Ollama
                              # first run pulls bge-m3 (~1.5 GB, takes ~60 s)
```

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python scripts/setup_collection.py   # one-time: create Milvus collection
python scripts/ingest.py --source all
```

To use the large-scale dataset instead (10 K Jira + 2 K Confluence):

```bash
python scripts/generate_dataset.py
python scripts/ingest.py --source all --data-dir data
```

---

## Files

```
KnowOps/
├── docker-compose.yml          # Milvus, Postgres, Ollama
├── .env.example
├── requirements.txt
├── data/
│   ├── jira_tickets.json       # 60 Jira tickets (hand-crafted)
│   ├── confluence_pages.json   # 60 Confluence pages (hand-crafted)
│   └── trap_manifest.json      # 10 adversarial test groups
├── knowops/
│   ├── embedder.py             # Ollama HTTP client
│   ├── schema.py               # Milvus collection + index config
│   └── db.py                   # Postgres tables
└── scripts/
    ├── setup_collection.py     # run once before ingesting
    ├── ingest.py               # chunk → embed → store
    └── generate_dataset.py     # generates large-scale test data
```

---

## Trap pairs

The dataset has 10 adversarial groups. Each group has two docs on the same topic — one old with wrong info, one recent with correct info. Both look equally relevant to a plain search.

| Doc | Updated | Content |
|-----|---------|---------|
| CONF-2001 ❌ | 2024-09-28 | Use IVF_FLAT index, timeout=60 s |
| CONF-2049 ✅ | 2025-12-20 | Use HNSW index, timeout=10 s |

Query: *"What Milvus index should I use?"*
- Search by similarity alone → may return CONF-2001
- With recency scoring → always returns CONF-2049

---

## Key settings

| Variable | Default | What it does |
|---|---|---|
| `OPENROUTER_API_KEY` | — | Required |
| `PIPELINE_ALPHA` | `0.7` | How much weight goes to similarity vs. recency |
| `PIPELINE_LAMBDA` | `0.005` | How fast recency decays (~139-day half-life) |
| `RETRIEVER_TOP_K` | `20` | Candidates fetched from Milvus |
| `RERANKER_TOP_K` | `5` | Results kept after reranking |
| `EMBEDDING_BATCH_SIZE` | `8` | Docs per embedding call |

Full list: [.env.example](.env.example)
