#!/usr/bin/env python3
"""
Build Order Steps 1 & 2 — Ingestion pipeline.

Reads jira_tickets.json and/or confluence_pages.json, chunks each document,
embeds chunks via Ollama bge-m3, and upserts vectors + metadata into Milvus.
Document metadata and a content-hash deduplication key are written to Postgres.

Usage:
    python scripts/ingest.py --source all        # ingest both datasets
    python scripts/ingest.py --source jira       # Jira tickets only
    python scripts/ingest.py --source confluence # Confluence pages only
    python scripts/ingest.py --source all --dry-run   # validate without writing
    python scripts/ingest.py --source all --data-dir ./data/large  # custom input dir
"""

import argparse
import hashlib
import json
import logging
import os
import sys
import uuid
from datetime import datetime, timezone
from typing import Iterator

from tqdm import tqdm
from dotenv import load_dotenv
from pymilvus import connections, Collection

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from knowops.embedder import embed_batch, check_ollama_health
from knowops.schema import COLLECTION_NAME, OUTPUT_FIELDS, SEARCH_PARAMS
from knowops.db import (
    create_tables, get_session, DocumentMetadata, IngestionRun,
    check_postgres_health,
)

load_dotenv()
logging.basicConfig(
    level=os.getenv("PIPELINE_LOG_LEVEL", "INFO"),
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
log = logging.getLogger("ingest")

# ── Tunables (override via env) ──────────────────────────────────────────────
BATCH_SIZE     = int(os.getenv("EMBEDDING_BATCH_SIZE", "8"))
CHUNK_SIZE     = 1800   # characters (~450 tokens); keeps well inside bge-m3's 8192-token limit
CHUNK_OVERLAP  = 200    # characters of overlap between adjacent chunks


# ── Document chunking ────────────────────────────────────────────────────────

def chunk_text(text: str) -> list[str]:
    """Split text into overlapping chunks using a character-level sliding window.

    Why character-level instead of token-level:
    Avoids a tiktoken dependency for the ingestion step. At an average of
    4 chars/token, CHUNK_SIZE=1800 maps to ~450 tokens — well within bge-m3's
    8192-token input limit and producing chunks of reasonable semantic density.
    """
    if len(text) <= CHUNK_SIZE:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


# ── Date helpers ─────────────────────────────────────────────────────────────

def iso_to_unix(iso_str: str) -> int:
    """Convert an ISO 8601 date string ('2024-09-10T09:15:00Z') to Unix timestamp."""
    return int(datetime.fromisoformat(iso_str.replace("Z", "+00:00")).timestamp())


def content_hash(text: str) -> str:
    """SHA-256 hex digest of document content — used for change detection."""
    return hashlib.sha256(text.encode()).hexdigest()


# ── Milvus entity builder ────────────────────────────────────────────────────

def build_jira_entities(record: dict, chunks: list[str], vectors: list[list[float]]) -> list[dict]:
    """Map a Jira ticket record + its chunk vectors to a list of Milvus entity dicts."""
    return [
        {
            "doc_id":       record["id"],
            "chunk_index":  i,
            "source_type":  "jira",
            "title":        record["title"][:512],
            "content":      chunk[:4096],
            "space":        "",
            "priority":     record.get("priority", ""),
            "jira_type":    record.get("type", ""),
            "created_date": iso_to_unix(record["created_date"]),
            "updated_date": iso_to_unix(record["updated_date"]),
            "trap_group":   record.get("trap_group") or "",
            "embedding":    vector,
        }
        for i, (chunk, vector) in enumerate(zip(chunks, vectors))
    ]


def build_confluence_entities(record: dict, chunks: list[str], vectors: list[list[float]]) -> list[dict]:
    """Map a Confluence page record + its chunk vectors to a list of Milvus entity dicts."""
    return [
        {
            "doc_id":       record["id"],
            "chunk_index":  i,
            "source_type":  "confluence",
            "title":        record["title"][:512],
            "content":      chunk[:4096],
            "space":        record.get("space", "")[:128],
            "priority":     "",
            "jira_type":    "",
            "created_date": iso_to_unix(record["created_date"]),
            "updated_date": iso_to_unix(record["updated_date"]),
            "trap_group":   record.get("trap_group") or "",
            "embedding":    vector,
        }
        for i, (chunk, vector) in enumerate(zip(chunks, vectors))
    ]


# ── Validation ───────────────────────────────────────────────────────────────

REQUIRED_JIRA_FIELDS = {"id", "title", "description", "created_date", "updated_date"}
REQUIRED_CONF_FIELDS = {"id", "title", "content", "created_date", "updated_date"}


def validate_record(record: dict, source: str) -> str | None:
    """Return an error string if required fields are missing, else None."""
    required = REQUIRED_JIRA_FIELDS if source == "jira" else REQUIRED_CONF_FIELDS
    missing = required - set(record.keys())
    return f"missing fields: {missing}" if missing else None


# ── Batched ingestion ────────────────────────────────────────────────────────

def ingest_records(
    records: list[dict],
    source_type: str,
    collection: Collection,
    run_id: str,
    dry_run: bool,
) -> tuple[int, int, int]:
    """Ingest a list of records into Milvus and Postgres.

    Returns (processed, skipped, errors) counts.
    """
    processed = skipped = errors = 0
    session = get_session()

    content_field = "description" if source_type == "jira" else "content"

    for record in tqdm(records, desc=f"Ingesting {source_type}", unit="doc"):
        # ── 1. Validate required fields ──────────────────────────────────────
        error = validate_record(record, source_type)
        if error:
            log.warning("Skipping %s — %s", record.get("id", "?"), error)
            errors += 1
            continue

        doc_id = record["id"]
        body   = record[content_field]
        chash  = content_hash(body)

        # ── 2. Deduplication check ───────────────────────────────────────────
        # Skip embedding if the document hasn't changed since last ingestion.
        existing = session.get(DocumentMetadata, doc_id)
        if existing and existing.content_hash == chash and not dry_run:
            skipped += 1
            continue

        # ── 3. Chunk the document text ───────────────────────────────────────
        chunks = chunk_text(body)

        if dry_run:
            log.debug("[dry-run] %s → %d chunk(s)", doc_id, len(chunks))
            processed += 1
            continue

        # ── 4. Embed in batches ──────────────────────────────────────────────
        try:
            vectors: list[list[float]] = []
            for i in range(0, len(chunks), BATCH_SIZE):
                batch = chunks[i : i + BATCH_SIZE]
                vectors.extend(embed_batch(batch))
        except Exception as exc:
            log.error("Embedding failed for %s: %s", doc_id, exc)
            errors += 1
            continue

        # ── 5. Upsert into Milvus ─────────────────────────────────────────────
        try:
            if source_type == "jira":
                entities = build_jira_entities(record, chunks, vectors)
            else:
                entities = build_confluence_entities(record, chunks, vectors)

            # Transpose list-of-dicts → dict-of-lists (Milvus insert format)
            column_data = {key: [e[key] for e in entities] for key in entities[0]}
            collection.insert(column_data)
        except Exception as exc:
            log.error("Milvus insert failed for %s: %s", doc_id, exc)
            errors += 1
            continue

        # ── 6. Record metadata in Postgres ───────────────────────────────────
        try:
            meta = DocumentMetadata(
                id=doc_id,
                source_type=source_type,
                title=record["title"],
                space=record.get("space"),
                created_date=datetime.fromisoformat(record["created_date"].replace("Z", "+00:00")),
                updated_date=datetime.fromisoformat(record["updated_date"].replace("Z", "+00:00")),
                content_hash=chash,
                ingested_at=datetime.now(timezone.utc),
                chunk_count=len(chunks),
            )
            session.merge(meta)  # upsert: inserts or updates on primary key
            session.commit()
        except Exception as exc:
            log.error("Postgres write failed for %s: %s", doc_id, exc)
            session.rollback()

        processed += 1

    collection.flush()  # persist buffered inserts to disk
    session.close()
    return processed, skipped, errors


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest Jira/Confluence data into KnowOps")
    parser.add_argument(
        "--source",
        choices=["jira", "confluence", "all"],
        default="all",
        help="Which dataset to ingest (default: all)",
    )
    parser.add_argument(
        "--data-dir",
        default=os.path.join(os.path.dirname(__file__), "..", "data"),
        help="Directory containing jira_tickets.json and confluence_pages.json",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate records and count chunks without writing to Milvus or Postgres",
    )
    parser.add_argument("--milvus-host", default=os.getenv("MILVUS_HOST", "localhost"))
    parser.add_argument("--milvus-port", default=os.getenv("MILVUS_PORT", "19530"))
    args = parser.parse_args()

    data_dir = os.path.abspath(args.data_dir)

    # ── Preflight checks ─────────────────────────────────────────────────────
    if not args.dry_run:
        if not check_ollama_health():
            sys.exit("ERROR: Ollama is not reachable or bge-m3 is not loaded. "
                     "Run: docker compose up -d && wait for ollama-model-init to finish.")
        if not check_postgres_health():
            sys.exit("ERROR: Postgres is not reachable. Check DATABASE_URL in .env.")

    # ── Connect to Milvus ────────────────────────────────────────────────────
    if not args.dry_run:
        connections.connect(alias="default", host=args.milvus_host, port=args.milvus_port)
        collection = Collection(COLLECTION_NAME)
        collection.load()
    else:
        collection = None  # type: ignore[assignment]

    # ── Create Postgres tables (idempotent) ──────────────────────────────────
    if not args.dry_run:
        create_tables()

    # ── Start ingestion run record ───────────────────────────────────────────
    run_id = str(uuid.uuid4())
    run_start = datetime.now(timezone.utc)
    log.info("Ingestion run %s started (source=%s, dry_run=%s)", run_id, args.source, args.dry_run)

    total_processed = total_skipped = total_errors = 0

    # ── Ingest Jira tickets ──────────────────────────────────────────────────
    if args.source in ("jira", "all"):
        path = os.path.join(data_dir, "jira_tickets.json")
        with open(path) as f:
            records = json.load(f)
        log.info("Loaded %d Jira tickets from %s", len(records), path)
        p, s, e = ingest_records(records, "jira", collection, run_id, args.dry_run)
        total_processed += p; total_skipped += s; total_errors += e

    # ── Ingest Confluence pages ───────────────────────────────────────────────
    if args.source in ("confluence", "all"):
        path = os.path.join(data_dir, "confluence_pages.json")
        with open(path) as f:
            records = json.load(f)
        log.info("Loaded %d Confluence pages from %s", len(records), path)
        p, s, e = ingest_records(records, "confluence", collection, run_id, args.dry_run)
        total_processed += p; total_skipped += s; total_errors += e

    # ── Save run summary to Postgres ─────────────────────────────────────────
    if not args.dry_run:
        session = get_session()
        run = IngestionRun(
            run_id=run_id,
            started_at=run_start,
            completed_at=datetime.now(timezone.utc),
            documents_processed=total_processed,
            documents_skipped=total_skipped,
            errors_count=total_errors,
        )
        session.add(run)
        session.commit()
        session.close()

    log.info(
        "Ingestion complete — processed=%d  skipped=%d  errors=%d",
        total_processed, total_skipped, total_errors,
    )
    if total_errors > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
