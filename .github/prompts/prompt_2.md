# KnowOps — Complete Build Prompt

You are helping build "KnowOps" — a production-grade agentic RAG system that 
fixes date-aware retrieval in Milvus-based enterprise pipelines. 

---

## THE CORE PROBLEM THIS PROJECT SOLVES

A real enterprise RAG system (Jira + Confluence Q&A for service managers) was 
retrieving documents purely by semantic similarity, ignoring `updated_date` and 
`created_date` scalar fields in Milvus. Result: stale, outdated documents 
outranked fresher, correct ones. Service managers were getting wrong answers.

The root cause: retriever was an LLM call generating Milvus queries without 
date-awareness. Reranker had no recency weighting. No hybrid scoring existed.

KnowOps demonstrates the correct fix: hybrid retrieval (semantic + recency), 
proper scalar field usage in Milvus, and agentic pipeline with date-aware prompts.

---

## REPO STRUCTURE (already created)

read the local repo structure 

---

## STACK

- **Vector DB:** Milvus (local Docker)
- **Metadata:** PostgreSQL (local Docker, SQLAlchemy ORM)
- **Embeddings:** bge-m3 via Ollama (local, CPU only)
- **LLM:** OpenRouter → Claude Sonnet (for all agent calls)
- **Language:** Python 3.11+
- **Architecture:** Agentic pipeline — Planner → Retriever → Reranker → Answering

## DOCKER RESOURCE ALLOCATION (UM790 Pro, 32GB RAM, 8 cores)

- Ollama: 6GB RAM, 4 CPUs
- Milvus: 6GB RAM, 3 CPUs
- Postgres: 2GB RAM, 1 CPU
- OS + overhead: ~18GB free

---

## DATA SCHEMA

### jira_tickets.json
```json
{
  "id": "JIRA-1001",
  "type": "bug|feature_request|task",
  "title": "string",
  "description": "string (2-5 sentences)",
  "status": "open|in_progress|resolved|closed",
  "priority": "low|medium|high|critical",
  "assignee": "string",
  "created_date": "YYYY-MM-DDTHH:MM:SSZ",
  "updated_date": "YYYY-MM-DDTHH:MM:SSZ",
  "trap_group": "string or null"
}
```

### confluence_pages.json
```json
{
  "id": "CONF-2001",
  "title": "string",
  "space": "string",
  "content": "string (200-500 words)",
  "created_date": "YYYY-MM-DDTHH:MM:SSZ",
  "updated_date": "YYYY-MM-DDTHH:MM:SSZ",
  "trap_group": "string or null"
}
```

### trap_manifest.json
```json
{
  "trap_group": "string",
  "topic": "string",
  "correct_document_id": "string",
  "outdated_document_ids": ["string"],
  "expected_behavior": "string"
}
```

---

## MILVUS SCHEMA (schema.py)

Collection: `knowops_docs`

Fields:
- `doc_id` (VARCHAR, primary key)
- `source_type` (VARCHAR) — "jira" or "confluence"
- `title` (VARCHAR)
- `content_chunk` (VARCHAR)
- `embedding` (FLOAT_VECTOR, dim=1024) — bge-m3 output dimension
- `created_ts` (INT64) — Unix timestamp
- `updated_ts` (INT64) — Unix timestamp, KEY FIELD for date-ordering fix
- `trap_group` (VARCHAR, nullable)

Index: HNSW on embedding field
Scalar index: on `updated_ts` for fast filtering

---

## FRESHNESS SCORING (freshness.py)

This is the core fix. Must be implemented as actual code, NOT a prompt hack.

```python
import math
from datetime import datetime, timezone

def freshness_score(updated_ts: int, decay_days: int = 180) -> float:
    """
    Exponential decay freshness score.
    Score of 1.0 = updated today.
    Score approaches 0 as document ages beyond decay_days.
    """
    now = datetime.now(timezone.utc).timestamp()
    age_days = (now - updated_ts) / 86400
    return math.exp(-age_days / decay_days)

def hybrid_score(
    semantic_score: float,
    updated_ts: int,
    semantic_weight: float = 0.7,
    freshness_weight: float = 0.3,
    decay_days: int = 180
) -> float:
    """
    Combines semantic similarity with freshness.
    semantic_score: cosine similarity from Milvus (0-1)
    freshness_weight: tunable — increase for time-sensitive queries
    """
    fresh = freshness_score(updated_ts, decay_days)
    return (semantic_weight * semantic_score) + (freshness_weight * fresh)
```

Expose `decay_days` and weights as env vars so they can be tuned per query type.

---

## AGENT PROMPTS (prompts/*.md)

### prompts/planner.md
```markdown
# Planner Agent

You are the Planner in an enterprise knowledge retrieval system.
Your job: decompose the user's question into a structured retrieval plan.

## Input
A natural language question from a service manager about Jira tickets 
or Confluence documentation.

## Output (JSON only, no preamble)
{
  "intent": "bug_lookup|doc_lookup|status_check|general_qa",
  "sub_queries": ["string", ...],
  "time_sensitivity": "high|medium|low",
  "source_preference": "jira|confluence|both",
  "recency_required": true|false
}

## Rules
- If the question asks about current status, recent changes, latest 
  procedures, or anything implying NOW — set recency_required: true 
  and time_sensitivity: high.
- Split complex questions into multiple focused sub_queries.
- Never answer the question yourself. Only plan.

## Examples
Q: "What is the current escalation procedure for P1 incidents?"
→ recency_required: true (procedures change over time)

Q: "Show me all critical bugs assigned to Team Alpha"
→ source_preference: jira, recency_required: false
```

### prompts/retriever.md
```markdown
# Retriever Agent

You are the Retriever in an enterprise knowledge retrieval system.
Your job: generate precise Milvus search parameters from a retrieval plan.

## Input
A structured plan from the Planner agent (JSON).

## Output (JSON only, no preamble)
{
  "search_query": "string (optimized for semantic search)",
  "source_filter": "jira|confluence|both",
  "freshness_boost": true|false,
  "date_filter_days": null|number,
  "limit": number
}

## Rules
- If recency_required is true: set freshness_boost true.
- If time_sensitivity is high: set date_filter_days to 90 or less.
- Optimize search_query for semantic similarity — expand acronyms, 
  add synonyms, be specific.
- Default limit: 20 (reranker will reduce to top 5).
- NEVER filter by date alone — always combine with semantic search.

## Data Freshness Awareness
Documents in this system may have outdated versions. Always prefer 
retrieving more candidates (limit 20) so the Reranker can apply 
freshness scoring to surface the most current correct answer.
```

### prompts/reranker.md
```markdown
# Reranker Agent

You are the Reranker in an enterprise knowledge retrieval system.
Your job: select the best 3-5 documents from retrieved candidates, 
with explicit awareness of document freshness.

## Input
- Original query
- List of retrieved documents with: content, semantic_score, 
  updated_date, freshness_score, hybrid_score

## Output (JSON only, no preamble)
{
  "selected_ids": ["string", ...],
  "reasoning": "string (1-2 sentences explaining selection)",
  "freshness_warning": true|false,
  "warning_message": "string or null"
}

## Rules
- Prefer documents with higher hybrid_score (semantic + freshness combined).
- If two documents cover the same topic, ALWAYS prefer the more recently 
  updated one — even if the older one has slightly higher semantic score.
- Set freshness_warning: true if the best available document is older 
  than 180 days — warn the user that information may be outdated.
- If you detect a trap (old doc ranking above newer doc on same topic), 
  explicitly override in favor of the newer document.
- Never select a document just because it has high semantic score if a 
  clearly newer version of the same information exists.
```

### prompts/answering.md
```markdown
# Answering Agent

You are the Answering Agent in an enterprise knowledge retrieval system.
Your job: generate a precise, grounded answer for service managers.

## Input
- Original question
- Top 3-5 reranked documents with metadata
- freshness_warning flag from Reranker

## Output
A clear, concise answer followed by source citations.

## Rules
- Ground every claim in the provided documents. Do not add information 
  from outside the retrieved context.
- If freshness_warning is true, prepend your answer with:
  ⚠️ Note: The most recent available document on this topic is over 
  6 months old. Please verify this information is still current.
- Cite sources as: [JIRA-1001] or [CONF-2001] inline.
- If documents conflict (old vs new), state the conflict explicitly 
  and cite the newer document as authoritative.
- If no document adequately answers the question, say so clearly — 
  do not hallucinate.
- Keep answers under 200 words unless complexity requires more.
```

---

## BUILD ORDER (strict — do not skip steps)

### Step 1 — Infrastructure
- `docker-compose.yml` with Milvus + Postgres + Ollama, correct resource limits
- Verify all three services healthy before any Python

### Step 2 — Data
- Generate `jira_tickets.json`, `confluence_pages.json`, `trap_manifest.json`
- Dates must span 18-24 months with realistic distribution
- 10 trap pairs minimum, clearly documented in trap_manifest

### Step 3 — Schema + DB
- `schema.py` — Milvus collection with HNSW index + scalar index on updated_ts
- `db.py` — Postgres tables mirroring JSON schema via SQLAlchemy
- `setup_collection.py` — idempotent, run once

### Step 4 — Embedder + Ingest
- `embedder.py` — Ollama bge-m3 client, batch embedding with progress bar
- `ingest.py` — chunk → embed → upsert to Milvus + insert metadata to Postgres

### Step 5 — Freshness
- `freshness.py` — implement hybrid_score exactly as specified above
- `test_freshness.py` — unit tests with known inputs/outputs

### Step 6 — Agents
- Implement each agent loading its prompt from `prompts/*.md`
- Wire: Planner → Retriever → Reranker → Answering
- Each agent is a separate class with a single `run()` method

### Step 7 — Demo Script
- `demo.py` — run same trap_manifest queries against:
  1. Baseline: semantic-only retrieval (no freshness)
  2. Fixed: hybrid retrieval with freshness scoring
- Print side-by-side: which document was returned, was it correct per 
  trap_manifest, score breakdown

### Step 8 — Tests
- `test_retrieval.py` — for each trap in trap_manifest, assert that 
  fixed retrieval returns correct_document_id, not outdated_document_ids
- Must pass before repo is considered complete

### Step 9 — README
Write as a technical case study, not a tutorial:
1. Problem — what was broken and why
2. Diagnosis — how the root cause was identified
3. Architecture — diagram (ASCII ok) + stack decisions
4. The Fix — freshness.py hybrid scoring explained
5. Results — test pass rate, before/after demo output
6. Running locally — docker compose up, ingest, demo

---

## CODING CONSTRAINTS

- Explain each significant code block (what + why) as you write it
- Functions small, well-named, interview-explainable
- No unnecessary abstraction — portfolio project, not enterprise software
- Flag "nice to have" items that can be skipped in 2-week timeline
- All config (weights, decay_days, API keys, model names) via .env
- freshness_weight and semantic_weight must be tunable without code changes

---

## PHASE 2 (after Phase 1 complete)

Scale dataset: `generate_dataset.py` → 10,000 Jira + 2,000 Confluence pages.
Same JSON schema, same trap-pair logic at scale.
Goal: demonstrate Milvus performance at volume in demo.py output.

