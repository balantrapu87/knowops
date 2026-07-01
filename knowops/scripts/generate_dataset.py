#!/usr/bin/env python3
"""
Phase 2 — Scale-up dataset generator.

Produces 10,000 Jira tickets + 2,000 Confluence pages in the same JSON schema
as the hand-crafted dataset, with trap-pair logic preserved at scale.

Output files (written to --out-dir, default: data/):
    jira_tickets_large.json       — 10,000 Jira tickets
    confluence_pages_large.json   — 2,000 Confluence pages
    trap_manifest_large.json      — all trap group definitions

Usage:
    pip install faker
    python scripts/generate_dataset.py
    python scripts/generate_dataset.py --jira 10000 --confluence 2000 --seed 42

Trap pair design (see CONF-2025 for methodology):
    - Each trap group has 1 old document (wrong fact, early date) and
      1 new document (correct fact, late date) on the same topic.
    - Both documents use similar vocabulary → high cosine similarity.
    - Only recency-weighted reranking can reliably distinguish them.
    - ~200 Jira trap pairs + ~100 Confluence trap pairs are injected.
"""

import argparse
import json
import os
import random
from datetime import datetime, timedelta, timezone
from typing import Any

from faker import Faker

# ── Reproducibility ──────────────────────────────────────────────────────────
fake = Faker()


# ── Date helpers ─────────────────────────────────────────────────────────────

DATE_START    = datetime(2024, 7, 1, tzinfo=timezone.utc)
DATE_END      = datetime(2026, 6, 30, tzinfo=timezone.utc)
OLD_DATE_END  = datetime(2025, 1, 1, tzinfo=timezone.utc)   # trap "old" docs before this
NEW_DATE_START = datetime(2025, 7, 1, tzinfo=timezone.utc)  # trap "new" docs after this


def rand_date(start: datetime = DATE_START, end: datetime = DATE_END) -> str:
    delta = int((end - start).total_seconds())
    return (start + timedelta(seconds=random.randint(0, delta))).strftime("%Y-%m-%dT%H:%M:%SZ")


def old_date() -> str:
    return rand_date(DATE_START, OLD_DATE_END)


def new_date() -> str:
    return rand_date(NEW_DATE_START, DATE_END)


def mid_date() -> str:
    return rand_date(
        datetime(2024, 10, 1, tzinfo=timezone.utc),
        datetime(2025, 6, 30, tzinfo=timezone.utc),
    )


# ── Lookup pools ─────────────────────────────────────────────────────────────

ASSIGNEES = [
    "Alice Chen", "Bob Kumar", "Carlos Ruiz", "Diana Park", "Ethan Moss",
    "Fatima Al-Hassan", "George Lim", "Hannah Novak", "Ivan Petrov", "Julia Santos",
    "Kevin Wright", "Laura Becker", "Marcus Holt", "Nina Ferreira", "Oscar Tanaka",
    "Priya Nair", "Quinn O'Brien", "Rachel Kim", "Sebastian Müller", "Tanya Ivanova",
    "Felix Chen", "Sarah Lin",
]

COMPONENTS = [
    "Milvus", "PostgreSQL", "Ollama", "Redis", "Kafka", "the API gateway",
    "the ingestion pipeline", "the reranker", "the Planner agent", "the Answering Agent",
    "the embedding service", "the CI/CD pipeline", "Docker Compose", "Kubernetes",
    "the monitoring stack", "the authentication service", "the rate limiter",
    "the connection pool", "the vector index", "the query cache",
]

STATUSES = ["open", "in_progress", "resolved", "closed"]
PRIORITIES = ["low", "medium", "high", "critical"]
JIRA_TYPES = ["bug", "feature_request", "task"]
CONF_SPACES = ["Engineering", "Platform", "Service Management", "DevOps", "Security"]

STATUS_WEIGHTS   = [0.15, 0.20, 0.40, 0.25]
PRIORITY_WEIGHTS = [0.20, 0.35, 0.30, 0.15]
TYPE_WEIGHTS     = [0.40, 0.30, 0.30]  # bug / feature / task


# ── Trap pair templates ───────────────────────────────────────────────────────
# Each entry defines a topic where an old document contains wrong guidance
# and a new document contains the correct guidance.
# {old_val} / {new_val} are substituted at generation time.

JIRA_TRAP_TEMPLATES: list[dict] = [
    {
        "topic_tpl":   "correct {component} batch size for embedding",
        "component":   "Ollama bge-m3",
        "old_val": "4",   "new_val": "8",
        "old_title_tpl": "{component} OOM crash — batch size reduced to {old_val} as workaround",
        "old_desc_tpl":  (
            "The {component} service crashes with OOM when batch size exceeds {old_val}. "
            "Temporary workaround applied: reduce batch size to {old_val} in ingestion config. "
            "This halves throughput but prevents crashes on the 32 GB host."
        ),
        "new_title_tpl": "{component} OOM root cause fixed — batch size {new_val} is correct",
        "new_desc_tpl":  (
            "Previous workaround of batch_size={old_val} is no longer needed. "
            "Root cause was missing OLLAMA_NUM_PARALLEL=1 env var causing multiple model "
            "instances to load simultaneously. Setting OLLAMA_NUM_PARALLEL=1 keeps peak RAM "
            "at 18 GB; batch_size={new_val} is stable and correct. Revert all configs to {new_val}."
        ),
    },
    {
        "topic_tpl":   "{component} client-side gRPC timeout value",
        "component":   "Milvus",
        "old_val": "60", "new_val": "10",
        "old_title_tpl": "{component} search timeouts — gRPC timeout increased to {old_val}s workaround",
        "old_desc_tpl":  (
            "Vector search with top_k > 10 times out after 30s under moderate load. "
            "Workaround: increase client-side gRPC timeout to {old_val} seconds. "
            "Root cause not yet identified — suspected IVF_FLAT index misconfiguration."
        ),
        "new_title_tpl": "{component} timeout root cause fixed — revert timeout to {new_val}s",
        "new_desc_tpl":  (
            "Root cause confirmed: IVF_FLAT index used nlist=32, too low for the collection size. "
            "Migrated to HNSW index (M=16, ef_construction=200). p95 latency dropped from 27s to 1.1s. "
            "The {old_val}s timeout workaround must be reverted to {new_val}s. "
            "Batch_size restriction from previous ticket is also no longer needed."
        ),
    },
    {
        "topic_tpl":   "{component} connection pool size setting",
        "component":   "PostgreSQL",
        "old_val": "5",  "new_val": "10",
        "old_title_tpl": "{component} pool exhaustion — pool_size={old_val} insufficient under load",
        "old_desc_tpl":  (
            "Running ingestion with > 5 parallel workers exhausts the SQLAlchemy connection pool. "
            "Workers block on pool_timeout. Current pool_size={old_val}, max_overflow=0. "
            "Reduce workers to 4 as a temporary workaround."
        ),
        "new_title_tpl": "{component} pool size updated to {new_val} — pool exhaustion resolved",
        "new_desc_tpl":  (
            "Increased pool_size from {old_val} to {new_val} and set max_overflow=5, resolving the "
            "connection exhaustion issue. Total max connections = 15, within safe limits for the "
            "shared {component} instance. The 4-worker restriction from the previous ticket is lifted."
        ),
    },
    {
        "topic_tpl":   "API rate limit per authenticated client per minute",
        "component":   "API gateway",
        "old_val": "100", "new_val": "500",
        "old_title_tpl": "{component} rate limit blocks legitimate high-volume clients",
        "old_desc_tpl":  (
            "Authenticated clients hitting the {old_val} req/min limit are receiving 429 errors "
            "for legitimate batch query workflows. Redis cluster is well below capacity. "
            "Limit was set conservatively during initial launch."
        ),
        "new_title_tpl": "{component} rate limit increased to {new_val} req/min after infra upgrade",
        "new_desc_tpl":  (
            "Following Redis cluster upgrade (single-node to 3-node), the rate limit for "
            "authenticated clients has been raised from {old_val} to {new_val} requests per minute. "
            "The old limit of {old_val} is no longer in effect — update any client-side throttling "
            "logic that was based on that value."
        ),
    },
    {
        "topic_tpl":   "JWT access token expiry duration",
        "component":   "authentication service",
        "old_val": "1h",  "new_val": "24h",
        "old_title_tpl": "{component} 1-hour token expiry causes CI pipeline re-auth failures",
        "old_desc_tpl":  (
            "CI pipelines that run longer than {old_val} expire their access tokens mid-run, "
            "causing authenticated API calls to fail with 401. "
            "Workaround: generate a fresh token at the start of each job stage."
        ),
        "new_title_tpl": "{component} token expiry extended to {new_val} with session binding",
        "new_desc_tpl":  (
            "Access token expiry has been increased from {old_val} to {new_val} following a "
            "security review. Session binding (IP + user-agent) compensates for the longer lifetime. "
            "Refresh tokens extended from 7d to 30d. The {old_val} workaround in CI pipelines "
            "should be removed — tokens now outlive any standard pipeline run."
        ),
    },
    {
        "topic_tpl":   "embedding vector dimension for Milvus collection",
        "component":   "embedding pipeline",
        "old_val": "1536", "new_val": "1024",
        "old_title_tpl": "{component} using text-embedding-ada-002 with dim={old_val}",
        "old_desc_tpl":  (
            "Milvus collection created with dim={old_val} for OpenAI text-embedding-ada-002 model. "
            "All ingestion and retrieval code must use dim={old_val}. "
            "OpenAI API key required in OPENAI_API_KEY env var."
        ),
        "new_title_tpl": "{component} migrated to bge-m3 with dim={new_val} — ada-002 deprecated",
        "new_desc_tpl":  (
            "Embedding model migrated from text-embedding-ada-002 (dim={old_val}, cloud API) to "
            "bge-m3 via Ollama (dim={new_val}, local CPU). Milvus collection must be recreated with "
            "dim={new_val}. Any configuration referencing dim={old_val} or text-embedding-ada-002 "
            "is outdated and will cause dimension mismatch errors. Run setup_collection.py --force."
        ),
    },
    {
        "topic_tpl":   "Kafka topic naming convention for document ingestion events",
        "component":   "Kafka",
        "old_val": "documents.ingest", "new_val": "platform.documents.ingest",
        "old_title_tpl": "{component} consumer group still using legacy topic {old_val}",
        "old_desc_tpl":  (
            "The ingestion worker consumer group subscribes to {old_val}. "
            "This topic was created during the initial platform setup. "
            "Producer configuration should target {old_val} for document embedding queue events."
        ),
        "new_title_tpl": "{component} topic renamed to {new_val} — legacy name {old_val} deleted",
        "new_desc_tpl":  (
            "As part of the Platform Naming Convention Migration, all topics were renamed to use "
            "the 'platform.' prefix. The old topic {old_val} no longer exists in the cluster. "
            "Update all consumer and producer configs to use {new_val}. Using the old name will "
            "cause 'Unknown topic' errors. Run the migration script to update consumer group offsets."
        ),
    },
    {
        "topic_tpl":   "on-call escalation tier 2 contact for production incidents",
        "component":   "incident management",
        "old_val": "engineering lead", "new_val": "SRE on-call",
        "old_title_tpl": "{component} escalation path updated — Tier 2 is now {new_val}",
        "old_desc_tpl":  (
            "For P1 incidents not resolved within 30 minutes, escalate to the {old_val} "
            "(the current sprint engineering lead). The {old_val} has production access and "
            "can approve emergency changes. Page via PagerDuty policy KnowOps-Secondary."
        ),
        "new_title_tpl": "{component}: Tier 2 is now {new_val} — do NOT page {old_val}",
        "new_desc_tpl":  (
            "Tier 2 escalation now routes to the {new_val} (PagerDuty: SRE-Primary), not the "
            "{old_val}. This change reflects the introduction of a dedicated SRE function in Q1 2026. "
            "Escalation threshold also reduced from 30 minutes to 20 minutes. "
            "Paging the {old_val} for Tier 2 incidents is incorrect — they are no longer in the rotation."
        ),
    },
    {
        "topic_tpl":   "reranker recency decay lambda parameter value",
        "component":   "reranker",
        "old_val": "0.01", "new_val": "0.005",
        "old_title_tpl": "{component} lambda={old_val} over-penalises relevant older documents",
        "old_desc_tpl":  (
            "With lambda={old_val}, documents older than 6 months receive near-zero combined scores "
            "even when their semantic similarity is high. The decay is too aggressive for a corpus "
            "spanning 24 months. Increasing lambda to make decay more gradual is recommended."
        ),
        "new_title_tpl": "{component} lambda tuned to {new_val} — half-life now 139 days",
        "new_desc_tpl":  (
            "Lambda updated from {old_val} to {new_val} after calibration on the full 24-month corpus. "
            "At lambda={new_val}, the recency half-life is ln(2)/0.005 ≈ 139 days, which balances "
            "freshness preference against relevant older documents. "
            "Any configuration still using lambda={old_val} should be updated to {new_val}."
        ),
    },
    {
        "topic_tpl":   "retriever top_k candidate count passed to reranker",
        "component":   "retriever",
        "old_val": "5", "new_val": "20",
        "old_title_tpl": "{component} top_k={old_val} gives reranker too few candidates",
        "old_desc_tpl":  (
            "The Milvus search is configured to return top_k={old_val} results. "
            "For trap-pair queries, the correct document sometimes falls outside the top {old_val} "
            "Milvus results, so the reranker never sees it. Increasing top_k is required."
        ),
        "new_title_tpl": "{component} top_k increased to {new_val} — trap accuracy improved",
        "new_desc_tpl":  (
            "Milvus search top_k increased from {old_val} to {new_val}. The reranker now receives "
            "20 candidates before applying recency-weighted scoring. "
            "Trap accuracy on the manifest increased from 60%% to 100%% after this change. "
            "The old value of top_k={old_val} is deprecated — do not use it in new configurations."
        ),
    },
]

CONF_TRAP_TEMPLATES: list[dict] = [
    {
        "topic_tpl":   "Milvus index type and configuration for {collection} collection",
        "collection":  "knowops_documents",
        "old_val": "IVF_FLAT (nlist=32)", "new_val": "HNSW (M=16, ef_construction=200)",
        "old_title_tpl": "Milvus Index Configuration Guide",
        "old_content_tpl": (
            "## Recommended Index Type\n\nUse IVF_FLAT with nlist=32 for the {collection} collection. "
            "This provides a good balance between build time and search speed for small collections.\n\n"
            "```python\nindex_params = {{'index_type': 'IVF_FLAT', 'metric_type': 'COSINE', "
            "'params': {{'nlist': 32}}}}\n```\n\n"
            "Set client-side gRPC timeout to **60 seconds** to avoid premature timeouts on "
            "queries with top_k > 10. This is a known characteristic of IVF_FLAT at this nlist setting.\n\n"
            "Search parameters: nprobe=8 (25%% of nlist)."
        ),
        "new_title_tpl": "Milvus Index Configuration Guide (Current — HNSW)",
        "new_content_tpl": (
            "**Note**: This page supersedes the earlier version. IVF_FLAT with nlist=32 caused "
            "severe latency regressions. The correct index is HNSW as described below.\n\n"
            "## Recommended Index Type\n\nUse **HNSW** with M=16, ef_construction=200.\n\n"
            "```python\nindex_params = {{'index_type': 'HNSW', 'metric_type': 'COSINE', "
            "'params': {{'M': 16, 'ef_construction': 200}}}}\n```\n\n"
            "Set client-side gRPC timeout to **10 seconds** (not 60s). "
            "With HNSW, p95 search latency is 1.1s. The 60s timeout from the old guide is incorrect "
            "and must be reverted. Search params: ef=64."
        ),
    },
    {
        "topic_tpl":   "PostgreSQL connection pool size for {service} service",
        "service":     "ingestion",
        "old_val": "pool_size=5, max_overflow=0", "new_val": "pool_size=10, max_overflow=5",
        "old_title_tpl": "PostgreSQL Connection Pool Configuration",
        "old_content_tpl": (
            "## Current Configuration\n\n"
            "```python\nengine = create_engine(DATABASE_URL, pool_size=5, max_overflow=0, "
            "pool_timeout=30)\n```\n\n"
            "**pool_size=5**: Five persistent connections. Sufficient for the {service} service "
            "with up to 5 workers.\n**max_overflow=0**: No connections beyond pool_size permitted.\n\n"
            "Do not increase pool_size above **20** for the shared PostgreSQL instance. "
            "Current guidance: keep total connections at or below 20."
        ),
        "new_content_tpl": (
            "**Note**: This page supersedes the earlier version. pool_size=5 caused connection "
            "exhaustion under production ingestion load.\n\n"
            "## Current Configuration\n\n"
            "```python\nengine = create_engine(DATABASE_URL, pool_size=10, max_overflow=5, "
            "pool_timeout=30, pool_recycle=3600)\n```\n\n"
            "**pool_size=10**: Updated after load testing — 5 was insufficient for 8-worker runs.\n"
            "**max_overflow=5**: Allows 15 total connections. Still within the ≤20 total guideline.\n\n"
            "The old pool_size=5 / max_overflow=0 configuration caused JIRA-1003. Do not revert."
        ),
    },
    {
        "topic_tpl":   "API rate limit for authenticated clients of the {service} API",
        "service":     "KnowOps query",
        "old_val": "100 req/min", "new_val": "500 req/min",
        "old_title_tpl": "API Rate Limiting Policy",
        "old_content_tpl": (
            "## Rate Limits\n\n"
            "Authenticated clients: **100 requests per minute**.\n"
            "Unauthenticated clients: 20 requests per minute.\n"
            "Premium service accounts: 100 requests per minute.\n\n"
            "All tiers share a hard cap of 100 req/min. Requests above this receive HTTP 429 "
            "with a Retry-After header. The limit is enforced using a Redis sliding window counter."
        ),
        "new_content_tpl": (
            "**Note**: Rate limits were increased after a Redis cluster upgrade in Q3 2025. "
            "The 100 req/min limit from the earlier version of this page is no longer correct.\n\n"
            "## Current Rate Limits\n\n"
            "Authenticated clients: **500 requests per minute**.\n"
            "Premium service accounts: 1,000 requests per minute.\n\n"
            "The limit increase was enabled by upgrading from a single-node to a 3-node Redis cluster. "
            "Any client-side throttling code based on 100 req/min should be updated to 500 req/min."
        ),
    },
    {
        "topic_tpl":   "production {env} deployment rollback procedure",
        "env":         "production",
        "old_val": "incomplete — no pre-rollback checklist", "new_val": "complete v2 procedure",
        "old_title_tpl": "Production Deployment Rollback Runbook",
        "old_content_tpl": (
            "## Rollback Steps\n\n"
            "1. SSH to the {env} host\n"
            "2. Navigate to /opt/knowops\n"
            "3. Identify previous image tag from deploy.log\n"
            "4. Update IMAGE_TAG in .env\n"
            "5. Run: docker compose pull && docker compose up -d\n"
            "6. Verify: docker compose ps\n"
            "7. Post confirmation in #deployments\n\n"
            "Run smoke tests: python tests/smoke_test.py --env {env}"
        ),
        "new_content_tpl": (
            "**Note**: This page supersedes the incomplete earlier version. The previous runbook "
            "was missing a pre-rollback health snapshot, database migration safety check, and "
            "PagerDuty notification steps.\n\n"
            "## Pre-Rollback Checklist\n\n"
            "- [ ] Declare P1 incident in #incidents if not already open\n"
            "- [ ] Record state: docker compose ps > /tmp/pre_rollback_state.txt\n"
            "- [ ] Check for DB migrations in the failed deployment — if present, contact DBA first\n\n"
            "## Rollback Steps\n\n"
            "1. SSH to {env} host\n"
            "2. cd /opt/knowops\n"
            "3. export ROLLBACK_TAG=$(git tag --sort=-creatordate | sed -n '2p')\n"
            "4. IMAGE_TAG=$ROLLBACK_TAG docker compose up -d\n"
            "5. Verify health, run smoke tests, update PagerDuty incident"
        ),
    },
    {
        "topic_tpl":   "JWT access token expiry and refresh token lifetime policy",
        "service":     "authentication",
        "old_val": "access=1h, refresh=7d", "new_val": "access=24h, refresh=30d",
        "old_title_tpl": "Authentication Token Policy",
        "old_content_tpl": (
            "## Token Lifetimes\n\n"
            "| Token Type | Expiry |\n|------------|--------|\n"
            "| Access Token | **1 hour** |\n"
            "| Refresh Token | 7 days |\n\n"
            "Access tokens expire after **1 hour**. Clients must call POST /auth/refresh before "
            "expiry. Storing tokens in localStorage is forbidden. Use HttpOnly cookies for refresh tokens."
        ),
        "new_content_tpl": (
            "**Note**: Token lifetimes were updated in Q4 2025 after a security review. "
            "The 1-hour access token lifetime from the earlier version of this page is incorrect.\n\n"
            "## Current Token Lifetimes\n\n"
            "| Token Type | Expiry |\n|------------|--------|\n"
            "| Access Token | **24 hours** |\n"
            "| Refresh Token | 30 days |\n\n"
            "Access tokens now expire after **24 hours** (not 1 hour). Session binding compensates "
            "for the longer lifetime. Refresh tokens valid for 30 days (previously 7). "
            "CI pipelines no longer need per-stage token refresh workarounds."
        ),
    },
    {
        "topic_tpl":   "Kafka topic names for {domain} domain events",
        "domain":      "document processing",
        "old_val": "events.raw / documents.ingest", "new_val": "platform.events.ingested / platform.documents.ingest",
        "old_title_tpl": "Kafka Topic Registry",
        "old_content_tpl": (
            "## Topic Inventory\n\n"
            "| Topic | Purpose | Partitions |\n|-------|---------|------------|\n"
            "| events.raw | Raw inbound events | 6 |\n"
            "| events.processed | Enriched events | 6 |\n"
            "| documents.ingest | Documents queued for embedding | 3 |\n"
            "| documents.embedded | Vectors ready for Milvus | 3 |\n\n"
            "Follow the naming pattern {domain}.{stage} for new topics. "
            "Contact Platform before creating topics."
        ),
        "new_content_tpl": (
            "**Note**: ALL topic names changed in December 2025 as part of the Platform Naming "
            "Convention Migration. The legacy names from the earlier version of this page no longer "
            "exist in the cluster. Consumers using old names will see 'Unknown topic' errors.\n\n"
            "## Current Topic Inventory\n\n"
            "| Topic | Purpose | Partitions |\n|-------|---------|------------|\n"
            "| platform.events.ingested | Raw inbound events | 12 |\n"
            "| platform.events.processed | Enriched events | 12 |\n"
            "| platform.documents.ingest | Embedding queue | 6 |\n"
            "| platform.documents.embedded | Vectors ready | 6 |\n"
            "| platform.documents.ingest.dlq | Dead letter | 2 |\n\n"
            "New naming pattern: platform.{domain}.{stage}"
        ),
    },
    {
        "topic_tpl":   "embedding model configuration and vector dimension for {system}",
        "system":      "KnowOps",
        "old_val": "text-embedding-ada-002, dim=1536", "new_val": "bge-m3 via Ollama, dim=1024",
        "old_title_tpl": "Embedding Model Configuration",
        "old_content_tpl": (
            "## Current Model\n\nKnowOps uses **text-embedding-ada-002** via the OpenAI API. "
            "Output dimension: **1536**.\n\n"
            "```python\nresponse = openai.Embedding.create(input=text, model='text-embedding-ada-002')\n"
            "vector = response['data'][0]['embedding']  # 1536-dimensional\n```\n\n"
            "Milvus collection must be configured with dim=1536. "
            "OPENAI_API_KEY is required. Pricing: $0.0001 per 1K tokens."
        ),
        "new_content_tpl": (
            "**Note**: The embedding model was migrated to bge-m3 in March 2026. "
            "text-embedding-ada-002 and dim=1536 from the earlier version are outdated — "
            "using them will cause dimension mismatch errors in Milvus.\n\n"
            "## Current Model\n\nKnowOps uses **bge-m3** via Ollama (local CPU). "
            "Output dimension: **1024**.\n\n"
            "```python\nresponse = httpx.post('http://ollama:11434/api/embeddings', "
            "json={'model': 'bge-m3', 'prompt': text})\nvector = response.json()['embedding']  "
            "# 1024-dimensional\n```\n\n"
            "Milvus dim must be 1024. No OpenAI key required — model runs locally."
        ),
    },
    {
        "topic_tpl":   "on-call escalation path and tier 2 contact for {service} incidents",
        "service":     "production",
        "old_val": "Tier 2 = engineering lead, threshold = 30 min", "new_val": "Tier 2 = SRE on-call, threshold = 20 min",
        "old_title_tpl": "On-Call Escalation Policy",
        "old_content_tpl": (
            "## Escalation Tiers\n\n"
            "**Tier 1 — On-Call Engineer**: Respond within 15 min. Weekly rotation.\n\n"
            "**Tier 2 — Engineering Lead**: Escalate if not resolved within **30 minutes**. "
            "The engineering lead for the current sprint is Tier 2. PagerDuty: KnowOps-Secondary.\n\n"
            "**Tier 3 — Engineering Manager**: Escalate after **60 minutes** or for P0 incidents."
        ),
        "new_content_tpl": (
            "**Note**: Escalation path updated in Q1 2026. Tier 2 is now the SRE on-call, "
            "not the engineering lead. Thresholds also reduced. Using the old path routes incidents "
            "to the wrong person.\n\n"
            "## Current Escalation Tiers\n\n"
            "**Tier 1 — On-Call Engineer**: Respond within 15 min.\n\n"
            "**Tier 2 — SRE On-Call**: Escalate if not resolved within **20 minutes** (was 30). "
            "PagerDuty: SRE-Primary. Do NOT page the engineering lead for Tier 2.\n\n"
            "**Tier 3 — Engineering Manager + VP Engineering**: Escalate after **45 minutes** (was 60)."
        ),
    },
    {
        "topic_tpl":   "reranker alpha and lambda parameter values for {corpus} corpus",
        "corpus":      "24-month",
        "old_val": "alpha=0.9, lambda=0.01", "new_val": "alpha=0.7, lambda=0.005",
        "old_title_tpl": "Reranker Tuning Guide",
        "old_content_tpl": (
            "## Recommended Parameters\n\n"
            "After initial tuning on a 6-month test corpus:\n\n"
            "```\nalpha = 0.9   # high semantic weight\nlambda = 0.01  # moderate recency decay\n```\n\n"
            "With these values, the reranker strongly favours semantic similarity with a mild "
            "recency preference. Suitable when document dates are tightly clustered."
        ),
        "new_content_tpl": (
            "**Note**: Parameters were re-tuned on the full {corpus} corpus. "
            "The alpha=0.9 / lambda=0.01 values from the earlier guide are miscalibrated for "
            "this dataset and cause the reranker to miss some trap pair cases.\n\n"
            "## Current Recommended Parameters\n\n"
            "```\nalpha = 0.7    # balanced semantic + recency\nlambda = 0.005  # slower decay: half-life ~139 days\n```\n\n"
            "With alpha=0.7 and lambda=0.005, the reranker correctly resolves all 10 hand-crafted "
            "trap pairs. The old alpha=0.9 setting over-weighted semantic similarity and caused "
            "stale documents to outrank correct ones on 3 of the 10 trap queries."
        ),
    },
    {
        "topic_tpl":   "ingestion chunk size in tokens for {model} embedding model",
        "model":       "bge-m3",
        "old_val": "256 tokens", "new_val": "512 tokens",
        "old_title_tpl": "Document Chunking Strategy Guide",
        "old_content_tpl": (
            "## Chunk Size\n\nDocuments are split into chunks of **{old_val}** with 50-token overlap. "
            "This conservative setting ensures no chunk exceeds the {model} input limit.\n\n"
            "```python\nCHUNK_SIZE = 256  # tokens\nCHUNK_OVERLAP = 50\n```\n\n"
            "Smaller chunks improve retrieval precision for short factual questions."
        ),
        "new_content_tpl": (
            "**Note**: Chunk size was increased after evaluation showed {old_val} chunks lacked "
            "sufficient context for multi-sentence answers.\n\n"
            "## Current Chunk Size\n\nUse **{new_val}** with 64-token overlap for {model}.\n\n"
            "```python\nCHUNK_SIZE = 512  # tokens (~1800 characters)\nCHUNK_OVERLAP = 64\n```\n\n"
            "{model} supports up to 8192 tokens — {new_val} is well within the limit and provides "
            "better answer quality than {old_val}. Do not use the old value of {old_val}."
        ),
    },
]


# ── Generic content pools for non-trap documents ─────────────────────────────

BUG_TITLE_TEMPLATES = [
    "{component} {issue} under high load",
    "{component} returns incorrect results when {condition}",
    "{component} connection timeout after {N} minutes idle",
    "{component} crashes with OOM when processing batches of {N}",
    "{component} health check fails intermittently on cold start",
    "{component} gRPC connection drops silently after {N} minutes",
    "{component} scalar filter ignored in hybrid search",
    "{component} version upgrade breaks {feature}",
    "{component} data loss on unclean shutdown",
    "{component} permission error on volume mount",
]

ISSUES = ["OOM crash", "connection timeout", "data corruption", "incorrect results",
          "performance degradation", "authentication failure", "race condition", "deadlock"]

CONDITIONS = ["batch size exceeds {N}", "running {N} concurrent workers",
              "after {N} minutes idle", "processing special characters",
              "after a version upgrade", "under sustained load"]

FEATURES = ["caching", "streaming responses", "retry logic", "rate limiting",
            "deduplication", "incremental ingestion", "confidence scores",
            "date range filtering", "trace logging", "health endpoints",
            "Markdown output", "A/B testing harness"]

CONF_TOPICS = [
    ("Architecture Overview", "Engineering"),
    ("API Reference", "Engineering"),
    ("Developer Guide", "Engineering"),
    ("Debugging Guide", "Engineering"),
    ("Performance Tuning Guide", "Engineering"),
    ("Security Hardening Checklist", "Security"),
    ("Incident Response Runbook", "Service Management"),
    ("Service Restart Procedure", "Service Management"),
    ("Backup and Recovery Guide", "Service Management"),
    ("SLA Definitions", "Service Management"),
    ("Docker Compose Setup Guide", "Platform"),
    ("Environment Variable Reference", "Platform"),
    ("Monitoring and Alerting Guide", "Platform"),
    ("CI/CD Pipeline Configuration", "Platform"),
    ("Infrastructure Cost Estimation", "Platform"),
    ("On-Call Schedule", "Service Management"),
    ("Change Management Checklist", "Service Management"),
    ("Post-Incident Review Template", "Service Management"),
    ("Data Retention Policy", "Service Management"),
    ("Access Control Matrix", "Security"),
]


def fake_bug_title() -> str:
    tpl = random.choice(BUG_TITLE_TEMPLATES)
    return tpl.format(
        component=random.choice(COMPONENTS),
        issue=random.choice(ISSUES),
        condition=random.choice(CONDITIONS).format(N=random.choice([4, 8, 10, 16, 32])),
        feature=random.choice(FEATURES),
        N=random.choice([5, 10, 15, 30, 60]),
    )


def fake_bug_desc(title: str) -> str:
    symptom = fake.sentence(nb_words=12)
    workaround = fake.sentence(nb_words=10)
    impact = fake.sentence(nb_words=8)
    return (
        f"{symptom} Observed in the {random.choice(COMPONENTS)} component under production conditions. "
        f"Temporary workaround applied: {workaround.lower()} "
        f"{impact} Root cause investigation is ongoing."
    )


def fake_feature_title() -> str:
    feature = random.choice(FEATURES)
    component = random.choice(COMPONENTS)
    return f"Add {feature} support to {component}"


def fake_feature_desc() -> str:
    use_case = fake.sentence(nb_words=10)
    benefit = fake.sentence(nb_words=8)
    impl_note = fake.sentence(nb_words=12)
    return (
        f"{use_case} {benefit} "
        f"Implementation approach: {impl_note.lower()} "
        f"This feature is requested by {fake.company()} and aligns with the Q{random.randint(1,4)} roadmap."
    )


def fake_task_title() -> str:
    verbs = ["Set up", "Configure", "Migrate", "Document", "Implement", "Validate", "Optimise", "Automate"]
    return f"{random.choice(verbs)} {random.choice(COMPONENTS)} for {fake.bs()}"


def fake_task_desc() -> str:
    objective = fake.sentence(nb_words=10)
    approach = fake.sentence(nb_words=12)
    done_criteria = fake.sentence(nb_words=8)
    return (
        f"{objective} {approach} "
        f"Done when: {done_criteria.lower()} "
        f"Estimated effort: {random.choice(['0.5', '1', '2', '3', '5'])} days."
    )


def fake_conf_content(title: str, space: str) -> str:
    """Generate 200–400 word realistic technical documentation."""
    intro = f"This document covers {title.lower()} for the KnowOps platform."
    sections = []
    for _ in range(random.randint(2, 4)):
        heading = fake.bs().title()
        body = " ".join(fake.sentences(nb=random.randint(3, 6)))
        sections.append(f"## {heading}\n\n{body}")
    footer = (
        f"For questions, contact the {space} team in the #knowops-{space.lower().replace(' ', '-')} "
        f"Slack channel. Last reviewed by {random.choice(ASSIGNEES)}."
    )
    return intro + "\n\n" + "\n\n".join(sections) + "\n\n" + footer


# ── Ticket and page ID generators ─────────────────────────────────────────────

def jira_id(n: int) -> str:
    return f"JIRA-{10001 + n}"


def conf_id(n: int) -> str:
    return f"CONF-{20001 + n}"


def trap_group_id(prefix: str, n: int) -> str:
    return f"{prefix}-{n:04d}"


# ── Main generators ───────────────────────────────────────────────────────────

def generate_jira_tickets(total: int, n_trap_pairs: int) -> tuple[list[dict], list[dict]]:
    """Generate `total` Jira tickets with `n_trap_pairs` trap pairs injected.

    Returns (tickets, trap_manifest_entries).
    """
    tickets: list[dict] = []
    manifest: list[dict] = []

    # ── Inject trap pairs first ──────────────────────────────────────────────
    for i in range(n_trap_pairs):
        tpl = JIRA_TRAP_TEMPLATES[i % len(JIRA_TRAP_TEMPLATES)]
        group_id = trap_group_id("jira-trap", i + 1)

        comp     = tpl.get("component", random.choice(COMPONENTS))
        old_val  = tpl["old_val"]
        new_val  = tpl["new_val"]
        assignee_old = random.choice(ASSIGNEES)
        assignee_new = random.choice(ASSIGNEES)

        old_cd = old_date()
        new_cd = new_date()

        old_ticket = {
            "id":           jira_id(len(tickets)),
            "type":         "bug",
            "title":        tpl["old_title_tpl"].format(component=comp, old_val=old_val, new_val=new_val),
            "description":  tpl["old_desc_tpl"].format(component=comp, old_val=old_val, new_val=new_val),
            "status":       "resolved",
            "priority":     "high",
            "assignee":     assignee_old,
            "created_date": old_cd,
            "updated_date": old_cd,
            "trap_group":   group_id,
        }
        tickets.append(old_ticket)

        new_ticket = {
            "id":           jira_id(len(tickets)),
            "type":         "bug",
            "title":        tpl["new_title_tpl"].format(component=comp, old_val=old_val, new_val=new_val),
            "description":  tpl["new_desc_tpl"].format(component=comp, old_val=old_val, new_val=new_val),
            "status":       "closed",
            "priority":     "critical",
            "assignee":     assignee_new,
            "created_date": new_cd,
            "updated_date": new_date(),  # slightly later than created
            "trap_group":   group_id,
        }
        tickets.append(new_ticket)

        topic = tpl["topic_tpl"].format(
            component=comp,
            service=tpl.get("service", comp),
            collection=tpl.get("collection", ""),
            env=tpl.get("env", "production"),
        )
        manifest.append({
            "trap_group":           group_id,
            "topic":                topic,
            "correct_document_id":  new_ticket["id"],
            "outdated_document_ids": [old_ticket["id"]],
            "expected_behavior": (
                f"A recency-aware retriever should return {new_ticket['id']} "
                f"(updated {new_ticket['updated_date'][:10]}) which contains the correct "
                f"guidance: {new_val}. A semantic-only retriever may return {old_ticket['id']} "
                f"(updated {old_ticket['updated_date'][:10]}) which contains the outdated "
                f"guidance: {old_val} — still plausible but now incorrect."
            ),
        })

    # ── Fill remaining with regular tickets ──────────────────────────────────
    while len(tickets) < total:
        ticket_type = random.choices(JIRA_TYPES, weights=TYPE_WEIGHTS)[0]

        if ticket_type == "bug":
            title = fake_bug_title()
            desc  = fake_bug_desc(title)
        elif ticket_type == "feature_request":
            title = fake_feature_title()
            desc  = fake_feature_desc()
        else:
            title = fake_task_title()
            desc  = fake_task_desc()

        cd = mid_date()
        ud = rand_date(
            start=datetime.fromisoformat(cd.replace("Z", "+00:00")),
            end=DATE_END,
        )
        tickets.append({
            "id":           jira_id(len(tickets)),
            "type":         ticket_type,
            "title":        title[:200],
            "description":  desc,
            "status":       random.choices(STATUSES, weights=STATUS_WEIGHTS)[0],
            "priority":     random.choices(PRIORITIES, weights=PRIORITY_WEIGHTS)[0],
            "assignee":     random.choice(ASSIGNEES),
            "created_date": cd,
            "updated_date": ud,
            "trap_group":   None,
        })

    random.shuffle(tickets)
    return tickets, manifest


def generate_confluence_pages(total: int, n_trap_pairs: int) -> tuple[list[dict], list[dict]]:
    """Generate `total` Confluence pages with `n_trap_pairs` trap pairs injected.

    Returns (pages, trap_manifest_entries).
    """
    pages: list[dict] = []
    manifest: list[dict] = []

    # ── Inject trap pairs ────────────────────────────────────────────────────
    for i in range(n_trap_pairs):
        tpl      = CONF_TRAP_TEMPLATES[i % len(CONF_TRAP_TEMPLATES)]
        group_id = trap_group_id("conf-trap", i + 1)
        space    = random.choice(CONF_SPACES)

        old_val  = tpl["old_val"]
        new_val  = tpl["new_val"]
        sub_vars = {k: tpl.get(k, "") for k in ("collection", "service", "domain", "model", "corpus", "env", "system")}

        old_cd = old_date()
        new_cd = new_date()

        old_page = {
            "id":           conf_id(len(pages)),
            "title":        tpl["old_title_tpl"],
            "space":        space,
            "content":      tpl["old_content_tpl"].format(**sub_vars, old_val=old_val, new_val=new_val),
            "created_date": old_cd,
            "updated_date": old_cd,
            "trap_group":   group_id,
        }
        pages.append(old_page)

        new_title = tpl.get("new_title_tpl", tpl["old_title_tpl"] + " (Current — Updated)")
        new_page = {
            "id":           conf_id(len(pages)),
            "title":        new_title,
            "space":        space,
            "content":      tpl["new_content_tpl"].format(**sub_vars, old_val=old_val, new_val=new_val),
            "created_date": new_cd,
            "updated_date": new_date(),
            "trap_group":   group_id,
        }
        pages.append(new_page)

        topic = tpl["topic_tpl"].format(**sub_vars)
        manifest.append({
            "trap_group":            group_id,
            "topic":                 topic,
            "correct_document_id":   new_page["id"],
            "outdated_document_ids": [old_page["id"]],
            "expected_behavior": (
                f"A recency-aware retriever should return {new_page['id']} "
                f"(updated {new_page['updated_date'][:10]}) which contains the current correct "
                f"information: {new_val}. A semantic-only retriever may return {old_page['id']} "
                f"(updated {old_page['updated_date'][:10]}) which contains outdated "
                f"information: {old_val} — semantically similar but factually incorrect."
            ),
        })

    # ── Fill remaining with regular pages ────────────────────────────────────
    topic_cycle = CONF_TOPICS * ((total // len(CONF_TOPICS)) + 2)
    while len(pages) < total:
        topic_label, space = topic_cycle[len(pages) % len(topic_cycle)]
        component = random.choice(COMPONENTS)
        title = f"{topic_label}: {component}"
        content = fake_conf_content(title, space)

        cd = mid_date()
        ud = rand_date(
            start=datetime.fromisoformat(cd.replace("Z", "+00:00")),
            end=DATE_END,
        )
        pages.append({
            "id":           conf_id(len(pages)),
            "title":        title[:512],
            "space":        space,
            "content":      content,
            "created_date": cd,
            "updated_date": ud,
            "trap_group":   None,
        })

    random.shuffle(pages)
    return pages, manifest


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 2 — generate large-scale Jira + Confluence dataset"
    )
    parser.add_argument("--jira",        type=int, default=10_000, help="Number of Jira tickets to generate")
    parser.add_argument("--confluence",  type=int, default=2_000,  help="Number of Confluence pages to generate")
    parser.add_argument("--jira-traps",  type=int, default=200,    help="Number of Jira trap pairs to inject")
    parser.add_argument("--conf-traps",  type=int, default=100,    help="Number of Confluence trap pairs to inject")
    parser.add_argument("--seed",        type=int, default=42,     help="Random seed for reproducibility")
    parser.add_argument(
        "--out-dir",
        default=os.path.join(os.path.dirname(__file__), "..", "data"),
        help="Output directory (default: data/)",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    fake.seed_instance(args.seed)

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    print(f"Generating {args.jira:,} Jira tickets ({args.jira_traps} trap pairs)...")
    jira_tickets, jira_manifest = generate_jira_tickets(args.jira, args.jira_traps)

    print(f"Generating {args.confluence:,} Confluence pages ({args.conf_traps} trap pairs)...")
    conf_pages, conf_manifest = generate_confluence_pages(args.confluence, args.conf_traps)

    combined_manifest = jira_manifest + conf_manifest

    # ── Write output files ───────────────────────────────────────────────────
    def write_json(path: str, data: Any) -> None:
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Wrote {len(data):,} records → {path}")

    write_json(os.path.join(out_dir, "jira_tickets_large.json"),      jira_tickets)
    write_json(os.path.join(out_dir, "confluence_pages_large.json"),   conf_pages)
    write_json(os.path.join(out_dir, "trap_manifest_large.json"),      combined_manifest)

    print(f"\nDone — {len(combined_manifest)} total trap groups "
          f"({len(jira_manifest)} Jira, {len(conf_manifest)} Confluence)")
    print("Ingest the large dataset with:")
    print(f"  python scripts/ingest.py --source jira       --data-dir {out_dir}")
    print(f"  python scripts/ingest.py --source confluence --data-dir {out_dir}")


if __name__ == "__main__":
    main()
