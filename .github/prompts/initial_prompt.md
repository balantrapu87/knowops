```
I'm building "KnowOps" — a portfolio project demonstrating a production-grade
agentic RAG system for enterprise service management (Jira + Confluence Q&A).

CONTEXT:
This project reproduces and fixes a real-world bug: a Milvus-based RAG system
was retrieving documents purely by semantic similarity, ignoring
`updated_date`/`created_date` scalar fields — so stale/outdated documents were
ranking above fresher, correct ones. This project demonstrates the correct
architecture to fix that.

GOAL: Build a working RAG system end to end, open-sourced on GitHub, suitable
for technical interview discussion. 2-week timeline. Correctness over scale.

STACK:
- Vector DB: Milvus (local Docker)
- Metadata: Postgres (local, already running)
- Embeddings: bge-m3 via Ollama (local, CPU only — no GPU, 32GB RAM machine)
- LLM: OpenRouter (Claude Sonnet) for Planner / Retriever-query-gen / Reranker / Answer agents
- Language: Python
- Architecture: Agentic pipeline — Planner → Retriever → Reranker → Answering agent

PHASE 1 (do this first):
Generate a SMALL, hand-crafted ground-truth dataset — NOT 10,000 documents yet:
- 50-100 Jira tickets (varied: bugs, feature requests, status, priority,
  created_date, updated_date fields)
- 50-100 Confluence pages (technical docs, runbooks, FAQs)
- CRITICALLY: include deliberate "trap" pairs — e.g. an old Confluence page
  with outdated/wrong info on a topic, and a newer page with corrected info
  on the SAME topic, with a different updated_date. These traps must let me
  verify ground truth manually and prove the date-ordering fix actually works.

OUTPUT FORMAT (required, for direct ingestion):

jira_tickets.json — a JSON array, each item:
{
  "id": "JIRA-1001",
  "type": "bug" | "feature_request" | "task",
  "title": "string",
  "description": "string (2-5 sentences)",
  "status": "open" | "in_progress" | "resolved" | "closed",
  "priority": "low" | "medium" | "high" | "critical",
  "assignee": "string (fake name)",
  "created_date": "YYYY-MM-DDTHH:MM:SSZ",
  "updated_date": "YYYY-MM-DDTHH:MM:SSZ",
  "trap_group": "string or null"   // shared id linking trap pairs/sets, null if not part of a trap
}

confluence_pages.json — a JSON array, each item:
{
  "id": "CONF-2001",
  "title": "string",
  "space": "string (e.g. 'Engineering', 'Service Management', 'Platform')",
  "content": "string (200-500 words, realistic technical doc/runbook/FAQ)",
  "created_date": "YYYY-MM-DDTHH:MM:SSZ",
  "updated_date": "YYYY-MM-DDTHH:MM:SSZ",
  "trap_group": "string or null"
}

trap_manifest.json — a JSON array documenting each trap pair/set explicitly,
for manual verification during testing:
{
  "trap_group": "string",
  "topic": "string (what the trap is about)",
  "correct_document_id": "string (id of the document with the CURRENT correct answer)",
  "outdated_document_ids": ["string", ...],
  "expected_behavior": "string (1-2 sentences: what a correctly date-aware
    retriever should return vs what a naive semantic-only retriever would
    incorrectly return)"
}

Ensure dates span a realistic range (e.g. 18-24 months back to present) so
recency-based reranking has real signal to work with, not just a binary
old/new split.

PHASE 2 (after Phase 1 works end to end):
Write a reusable Python generator script to programmatically scale this up to
10,000 Jira tickets + 2,000 Confluence pages, preserving the same trap-pair
logic at scale, for a final "performs at scale" demo. Same output schema as
above.

BUILD ORDER:
1. Docker Compose for Milvus + ingestion script (vectors + metadata into Postgres)
2. Ollama + bge-m3 embedding pipeline
3. Basic retrieval (semantic search only) — confirm baseline bug exists
   (old docs outrank new ones on trap-group queries)
4. Fix: hybrid retrieval — semantic score + recency-weighted reranking
   (actual scoring logic in code, not a prompt-only hack)
5. Agentic pipeline wiring: Planner (decomposes query) → Retriever (Milvus
   hybrid search) → Reranker (date+relevance scoring) → Answering agent
   (OpenRouter/Claude Sonnet)
6. Before/after demo script: same queries from trap_manifest.json, run against
   broken vs fixed retrieval, show trap pairs resolved correctly
7. Clean README written as a technical case study (problem → diagnosis →
   architecture → fix → results), not a tutorial

CONSTRAINTS:
- Explain each significant code block briefly (what + why) as we go
- Keep functions small and well-named, suitable for explaining in interviews
- No unnecessary abstraction layers — this is a portfolio project, not enterprise software
- Flag clearly when something is "nice to have" and can be skipped given the
  2-week deadline

Start with Phase 1, Step 1: the Docker Compose setup for Milvus, and the
hand-crafted dataset generation script producing the three JSON files above.
```