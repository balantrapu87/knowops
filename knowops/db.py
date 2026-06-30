"""
PostgreSQL setup and session management.

Tables:
  document_metadata  — one row per ingested document (deduplication key)
  ingestion_runs     — audit trail for each ingest.py execution
"""

import os
from datetime import datetime, timezone
from sqlalchemy import (
    create_engine, Column, String, Integer, DateTime, Text, text
)
from sqlalchemy.orm import declarative_base, Session

_DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://knowops:knowops_dev@localhost:5432/knowops",
)

_engine = create_engine(_DATABASE_URL, pool_pre_ping=True, pool_size=10, max_overflow=5)
Base = declarative_base()


class DocumentMetadata(Base):
    """One row per source document. Used for deduplication on re-ingestion."""
    __tablename__ = "document_metadata"

    id           = Column(String(64),  primary_key=True)
    source_type  = Column(String(32),  nullable=False)
    title        = Column(Text,        nullable=False)
    space        = Column(String(128), nullable=True)
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
