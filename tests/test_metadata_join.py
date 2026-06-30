"""Offline unit tests for the Milvus↔Postgres metadata join (pure function)."""

from knowops.search import Candidate, merge_postgres_metadata


def _candidate(doc_id="JIRA-1", **kw):
    base = dict(
        doc_id=doc_id,
        source_type="milvus_src",
        title="milvus title",
        content="chunk text from milvus",
        created_ts=1_000,
        updated_ts=2_000,
        trap_group="milvus_group",
        semantic_score=0.42,
    )
    base.update(kw)
    return Candidate(**base)


def test_merge_overrides_metadata_from_postgres():
    cand = _candidate()
    meta = {
        "JIRA-1": {
            "source_type": "jira",
            "title": "pg title",
            "created_ts": 111,
            "updated_ts": 999,
            "trap_group": "pg_group",
        }
    }

    (merged,) = merge_postgres_metadata([cand], meta)

    assert merged.source_type == "jira"
    assert merged.title == "pg title"
    assert merged.created_ts == 111
    assert merged.updated_ts == 999          # the field that drives freshness
    assert merged.trap_group == "pg_group"


def test_merge_preserves_milvus_owned_fields():
    cand = _candidate()
    meta = {"JIRA-1": {"updated_ts": 999, "source_type": "jira"}}

    (merged,) = merge_postgres_metadata([cand], meta)

    # content + semantic_score stay with the vector store
    assert merged.content == "chunk text from milvus"
    assert merged.semantic_score == 0.42


def test_merge_keeps_fallback_when_doc_missing_in_postgres():
    cand = _candidate(doc_id="ORPHAN")

    result = merge_postgres_metadata([cand], {"OTHER": {"updated_ts": 1}})

    assert result[0].updated_ts == 2_000     # unchanged Milvus fallback
    assert result[0].source_type == "milvus_src"


def test_merge_empty_metadata_is_noop():
    cands = [_candidate(doc_id="A"), _candidate(doc_id="B")]

    result = merge_postgres_metadata(cands, {})

    assert [c.doc_id for c in result] == ["A", "B"]
    assert all(c.updated_ts == 2_000 for c in result)
