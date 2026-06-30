"""
Vector retrieval backends.

Two interchangeable implementations expose the same ``semantic_search`` contract:

  * MilvusBackend  — production path: embeds the query with bge-m3 and runs an
    HNSW search against the ``knowops_documents`` collection.
  * OfflineBackend — zero-infrastructure path: loads the JSON corpus, embeds it
    with deterministic lexical vectors, and brute-forces cosine similarity.

Both return ``Candidate`` objects collapsed to one row per document (best chunk),
with ``semantic_score`` filled in. Freshness/hybrid scores are layered on later
by the Retriever agent using the configured FreshnessProfile.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from knowops.config import SETTINGS, Settings
from knowops.embedder import embed, embed_offline

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_DATA_DIR = _PROJECT_ROOT / "data"


@dataclass
class Candidate:
    """A single retrieved document with its scores."""
    doc_id: str
    source_type: str
    title: str
    content: str
    created_ts: int
    updated_ts: int
    trap_group: str = ""
    chunk_index: int = 0
    semantic_score: float = 0.0
    freshness_score: float = 0.0
    hybrid_score: float = 0.0

    @property
    def updated_date(self) -> str:
        return datetime.fromtimestamp(self.updated_ts, timezone.utc).strftime("%Y-%m-%d")


# ── Helpers ──────────────────────────────────────────────────────────────────

def _iso_to_unix(iso_str: str) -> int:
    return int(datetime.fromisoformat(iso_str.replace("Z", "+00:00")).timestamp())


def _now_ts(now: Optional[float]) -> float:
    return now if now is not None else datetime.now(timezone.utc).timestamp()


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    return dot  # both operands are L2-normalised, so dot == cosine


def _collapse_by_doc(candidates: list[Candidate]) -> list[Candidate]:
    """Keep the highest-scoring chunk per doc_id (live search returns chunks)."""
    best: dict[str, Candidate] = {}
    for c in candidates:
        cur = best.get(c.doc_id)
        if cur is None or c.semantic_score > cur.semantic_score:
            best[c.doc_id] = c
    return list(best.values())


def _passes_source(source_type: str, source_filter: str) -> bool:
    return source_filter in ("both", None, "") or source_type == source_filter


# ── Offline backend ──────────────────────────────────────────────────────────

class OfflineBackend:
    """In-memory brute-force search over the JSON corpus (no services needed)."""

    def __init__(self, data_dir: str | Path = _DEFAULT_DATA_DIR):
        self.data_dir = Path(data_dir)
        self._docs: list[dict] = []
        self._vectors: dict[str, list[float]] = {}
        self._load()

    def _load(self) -> None:
        for fname, source in (
            ("jira_tickets.json", "jira"),
            ("confluence_pages.json", "confluence"),
        ):
            path = self.data_dir / fname
            if not path.exists():
                continue
            for rec in json.loads(path.read_text()):
                body = rec.get("description") if source == "jira" else rec.get("content", "")
                doc = {
                    "doc_id": rec["id"],
                    "source_type": source,
                    "title": rec.get("title", ""),
                    "content": body or "",
                    "created_ts": _iso_to_unix(rec["created_date"]),
                    "updated_ts": _iso_to_unix(rec["updated_date"]),
                    "trap_group": rec.get("trap_group") or "",
                }
                self._docs.append(doc)
                # Embed title + body so topical queries match on either.
                self._vectors[doc["doc_id"]] = embed_offline(f"{doc['title']} {doc['content']}")

    def semantic_search(
        self,
        query_text: str,
        top_k: int,
        source_filter: str = "both",
        date_filter_days: Optional[int] = None,
        now: Optional[float] = None,
    ) -> list[Candidate]:
        qvec = embed_offline(query_text)
        cutoff = None
        if date_filter_days is not None:
            cutoff = _now_ts(now) - date_filter_days * 86400

        scored: list[Candidate] = []
        for doc in self._docs:
            if not _passes_source(doc["source_type"], source_filter):
                continue
            if cutoff is not None and doc["updated_ts"] < cutoff:
                continue
            sim = _cosine(qvec, self._vectors[doc["doc_id"]])
            scored.append(Candidate(semantic_score=max(0.0, sim), **doc))

        scored.sort(key=lambda c: c.semantic_score, reverse=True)
        return scored[:top_k]


# ── Milvus backend ───────────────────────────────────────────────────────────

class MilvusBackend:
    """Production search against the Milvus ``knowops_documents`` collection."""

    def __init__(self, settings: Settings = SETTINGS):
        self.settings = settings
        self._collection = None

    def _connect(self):
        if self._collection is not None:
            return self._collection
        from pymilvus import connections, Collection
        from knowops.schema import COLLECTION_NAME

        connections.connect(
            alias="default",
            host=self.settings.milvus_host,
            port=self.settings.milvus_port,
        )
        col = Collection(COLLECTION_NAME)
        col.load()
        self._collection = col
        return col

    @staticmethod
    def _build_expr(source_filter: str, cutoff: Optional[int]) -> Optional[str]:
        clauses = []
        if source_filter in ("jira", "confluence"):
            clauses.append(f'source_type == "{source_filter}"')
        if cutoff is not None:
            clauses.append(f"updated_date >= {int(cutoff)}")
        return " and ".join(clauses) if clauses else None

    def semantic_search(
        self,
        query_text: str,
        top_k: int,
        source_filter: str = "both",
        date_filter_days: Optional[int] = None,
        now: Optional[float] = None,
    ) -> list[Candidate]:
        from knowops.schema import OUTPUT_FIELDS, SEARCH_PARAMS

        col = self._connect()
        qvec = embed(query_text, base_url=self.settings.ollama_base_url, offline=False)

        cutoff = None
        if date_filter_days is not None:
            cutoff = int(_now_ts(now) - date_filter_days * 86400)
        expr = self._build_expr(source_filter, cutoff)

        results = col.search(
            data=[qvec],
            anns_field="embedding",
            param=SEARCH_PARAMS,
            limit=top_k,
            expr=expr,
            output_fields=OUTPUT_FIELDS,
        )

        candidates: list[Candidate] = []
        for hit in results[0]:
            e = hit.entity
            candidates.append(
                Candidate(
                    doc_id=e.get("doc_id"),
                    source_type=e.get("source_type"),
                    title=e.get("title"),
                    content=e.get("content"),
                    created_ts=int(e.get("created_date")),
                    updated_ts=int(e.get("updated_date")),
                    trap_group=e.get("trap_group") or "",
                    chunk_index=int(e.get("chunk_index") or 0),
                    semantic_score=max(0.0, float(hit.distance)),
                )
            )
        return _collapse_by_doc(candidates)[:top_k]


# ── Factory ──────────────────────────────────────────────────────────────────

def get_backend(
    offline: Optional[bool] = None,
    settings: Settings = SETTINGS,
    data_dir: str | Path = _DEFAULT_DATA_DIR,
):
    """Return the offline or Milvus backend based on ``offline`` (defaults to SETTINGS)."""
    use_offline = settings.offline if offline is None else offline
    if use_offline:
        return OfflineBackend(data_dir=data_dir)
    return MilvusBackend(settings=settings)
