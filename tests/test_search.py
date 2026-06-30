import os
from datetime import datetime, timezone

os.environ.setdefault("KNOWOPS_OFFLINE", "1")

from knowops.search import Candidate, OfflineBackend

QUERY = "Milvus index configuration"


def _ts(iso: str) -> int:
    return int(datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp())


def test_semantic_search_returns_candidates():
    results = OfflineBackend().semantic_search(QUERY, top_k=5)

    assert results
    assert all(isinstance(candidate, Candidate) for candidate in results)


def test_semantic_search_results_sorted_by_semantic_score_descending():
    results = OfflineBackend().semantic_search(QUERY, top_k=10)

    assert [c.semantic_score for c in results] == sorted(
        (c.semantic_score for c in results), reverse=True
    )


def test_semantic_search_returns_one_row_per_doc_id():
    results = OfflineBackend().semantic_search(QUERY, top_k=20)
    doc_ids = [c.doc_id for c in results]

    assert len(doc_ids) == len(set(doc_ids))


def test_semantic_search_jira_source_filter_returns_only_jira():
    results = OfflineBackend().semantic_search(QUERY, top_k=10, source_filter="jira")

    assert results
    assert {c.source_type for c in results} == {"jira"}


def test_semantic_search_confluence_source_filter_returns_only_confluence():
    results = OfflineBackend().semantic_search(QUERY, top_k=10, source_filter="confluence")

    assert results
    assert {c.source_type for c in results} == {"confluence"}


def test_semantic_search_date_filter_excludes_older_documents():
    now = float(_ts("2025-12-25T00:00:00Z"))
    cutoff = now - 30 * 86_400

    results = OfflineBackend().semantic_search(
        QUERY,
        top_k=20,
        date_filter_days=30,
        now=now,
    )

    assert results
    assert all(candidate.updated_ts >= cutoff for candidate in results)
