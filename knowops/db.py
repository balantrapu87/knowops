"""
PostgreSQL setup and session management.

Tables:
  document_metadata  — one row per ingested document (deduplication key)
  ingestion_runs     — audit trail for each ingest.py execution
"""

import os
from datetime import datetime, timezone, timedelta
from sqlalchemy import (
    create_engine, Column, String, Integer, DateTime, Text, text, select
)
from sqlalchemy.orm import declarative_base, Session

_DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://knowops:knowops_dev@localhost:5432/knowops",
)

_engine = create_engine(_DATABASE_URL, pool_pre_ping=True, pool_size=10, max_overflow=5)
Base = declarative_base()


class DocumentMetadata(Base):
    """One row per source document. Authoritative metadata store.

    When METADATA_BACKEND=postgres, the retrieval path reads source_type,
    title, dates, and trap_group from here (joined to Milvus vector hits by
    doc_id) instead of from Milvus scalar fields.
    """
    __tablename__ = "document_metadata"

    id           = Column(String(64),  primary_key=True)
    source_type  = Column(String(32),  nullable=False)
    title        = Column(Text,        nullable=False)
    space        = Column(String(128), nullable=True)
    priority     = Column(String(16),  nullable=True)   # Jira only
    jira_type    = Column(String(32),  nullable=True)   # Jira only
    trap_group   = Column(String(64),  nullable=True)   # links trap pairs/sets
    created_date = Column(DateTime(timezone=True), nullable=False)
    updated_date = Column(DateTime(timezone=True), nullable=False)
    content_hash = Column(String(64),  nullable=False)   # SHA-256 hex
    ingested_at  = Column(DateTime(timezone=True), nullable=False,
                          default=lambda: datetime.now(timezone.utc))
    chunk_count  = Column(Integer, nullable=False, default=0)


class IngestionRun(Base):
    """Audit row created at the start of each ingest.py execution."""
    __tablename__ = "ingestion_runs"

    run_id              = Column(String(36),  primary_key=True)   # UUID string
    started_at          = Column(DateTime(timezone=True), nullable=False)
    completed_at        = Column(DateTime(timezone=True), nullable=True)
    documents_processed = Column(Integer, nullable=False, default=0)
    documents_skipped   = Column(Integer, nullable=False, default=0)
    errors_count        = Column(Integer, nullable=False, default=0)
    notes               = Column(Text, nullable=True)


def create_tables() -> None:
    """Create all tables if they do not already exist."""
    Base.metadata.create_all(_engine)


def get_session() -> Session:
    """Return a new SQLAlchemy session. Caller is responsible for closing it."""
    return Session(_engine)


def check_postgres_health() -> bool:
    """Return True if Postgres is reachable."""
    try:
        with _engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


# ── Metadata reads (used when METADATA_BACKEND=postgres) ─────────────────────

def _row_to_meta(row: "DocumentMetadata") -> dict:
    """Map a DocumentMetadata row to the dict shape the search layer expects."""
    return {
        "source_type": row.source_type,
        "title": row.title,
        "created_ts": int(row.created_date.timestamp()),
        "updated_ts": int(row.updated_date.timestamp()),
        "trap_group": row.trap_group or "",
        "priority": row.priority or "",
        "jira_type": row.jira_type or "",
    }


def fetch_metadata(doc_ids: list[str]) -> dict[str, dict]:
    """Return {doc_id: metadata dict} for the given doc_ids (one query)."""
    if not doc_ids:
        return {}
    session = get_session()
    try:
        rows = session.scalars(
            select(DocumentMetadata).where(DocumentMetadata.id.in_(doc_ids))
        ).all()
        return {row.id: _row_to_meta(row) for row in rows}
    finally:
        session.close()


def recent_docs_by_priority(
    priority: str = "high", days: int = 30, source_type: str | None = "jira"
) -> list[dict]:
    """Complex filter that is trivial in SQL, awkward in a vector store:
    'high-priority bugs updated in the last N days'."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    session = get_session()
    try:
        stmt = select(DocumentMetadata).where(
            DocumentMetadata.priority == priority,
            DocumentMetadata.updated_date >= cutoff,
        )
        if source_type:
            stmt = stmt.where(DocumentMetadata.source_type == source_type)
        stmt = stmt.order_by(DocumentMetadata.updated_date.desc())
        return [{"id": r.id, **_row_to_meta(r)} for r in session.scalars(stmt).all()]
    finally:
        session.close()


def docs_in_trap_group(trap_group: str) -> list[dict]:
    """All documents sharing a trap_group (the 'related documents' query)."""
    session = get_session()
    try:
        stmt = (
            select(DocumentMetadata)
            .where(DocumentMetadata.trap_group == trap_group)
            .order_by(DocumentMetadata.updated_date.desc())
        )
        return [{"id": r.id, **_row_to_meta(r)} for r in session.scalars(stmt).all()]
    finally:
        session.close()
