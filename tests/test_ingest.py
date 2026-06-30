"""
Unit tests for scripts/ingest.py.

Coverage:
  - chunk_text()              — sliding-window chunker
  - iso_to_unix()             — ISO 8601 → Unix timestamp
  - content_hash()            — SHA-256 fingerprint
  - validate_record()         — required-field guard
  - build_jira_entities()     — Milvus entity shape for Jira
  - build_confluence_entities() — Milvus entity shape for Confluence
  - ingest_records()          — full pipeline flow (Milvus + Postgres mocked)

Milvus, Postgres, and Ollama are all remote — nothing is called for real.

Run:
    source .venv/bin/activate
    pytest tests/test_ingest.py -v
"""

import hashlib
import sys
import os
import pytest
from unittest.mock import MagicMock, patch, call

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Import the pure helper functions directly (no server deps)
from scripts.ingest import (
    chunk_text,
    iso_to_unix,
    content_hash,
    validate_record,
    build_jira_entities,
    build_confluence_entities,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    REQUIRED_JIRA_FIELDS,
    REQUIRED_CONF_FIELDS,
)

# ── Shared test fixtures ──────────────────────────────────────────────────────

FAKE_VECTOR = [0.1] * 1024   # shape only matters, values don't


SAMPLE_JIRA = {
    "id": "JIRA-1001",
    "type": "bug",
    "title": "Milvus search times out on top-k > 10",
    "description": (
        "Vector search with top_k greater than 10 consistently times out after 30 seconds. "
        "Temporary workaround: increase client timeout to 60 seconds. "
        "Root cause suspected to be IVF_FLAT misconfiguration."
    ),
    "status": "resolved",
    "priority": "high",
    "assignee": "Marcus Holt",
    "created_date": "2024-09-10T09:15:00Z",
    "updated_date": "2024-09-25T14:30:00Z",
    "trap_group": "trap-milvus-search-timeout",
}

SAMPLE_CONFLUENCE = {
    "id": "CONF-2001",
    "title": "Milvus Index Configuration Guide",
    "space": "Engineering",
    "content": (
        "## Recommended Index Type\n\n"
        "Use IVF_FLAT with nlist=32 for the knowops_documents collection. "
        "Set client-side gRPC timeout to 60 seconds to avoid premature timeouts. "
        "Search parameters: nprobe=8 (25% of nlist). "
        "This guide covers setup, tuning, and known limitations of the current index."
    ),
    "created_date": "2024-08-15T10:00:00Z",
    "updated_date": "2024-09-28T14:00:00Z",
    "trap_group": "trap-milvus-index-type",
}


# ── chunk_text() ──────────────────────────────────────────────────────────────

class TestChunkText:

    def test_short_text_returns_single_chunk(self):
        """Text shorter than CHUNK_SIZE is returned as-is in a one-element list."""
        text = "short text"
        result = chunk_text(text)
        assert result == [text]

    def test_text_exactly_chunk_size_is_single_chunk(self):
        text = "x" * CHUNK_SIZE
        assert chunk_text(text) == [text]

    def test_long_text_produces_multiple_chunks(self):
        text = "a" * (CHUNK_SIZE * 2)
        chunks = chunk_text(text)
        assert len(chunks) > 1

    def test_all_chunks_within_max_length(self):
        text = "b" * (CHUNK_SIZE * 3 + 500)
        for chunk in chunk_text(text):
            assert len(chunk) <= CHUNK_SIZE

    def test_overlap_between_adjacent_chunks(self):
        """Each pair of adjacent chunks shares CHUNK_OVERLAP characters at the boundary."""
        text = "c" * (CHUNK_SIZE + CHUNK_OVERLAP + 100)
        chunks = chunk_text(text)
        assert len(chunks) >= 2
        # The tail of chunk 0 equals the head of chunk 1 for CHUNK_OVERLAP chars
        tail = chunks[0][-CHUNK_OVERLAP:]
        head = chunks[1][:CHUNK_OVERLAP]
        assert tail == head

    def test_no_text_lost_across_all_chunks(self):
        """Concatenating de-overlapped chunks reconstructs the original text."""
        text = "d" * (CHUNK_SIZE * 2 + 300)
        chunks = chunk_text(text)
        # Reconstruct: first chunk in full, then each subsequent chunk sans overlap prefix
        reconstructed = chunks[0]
        for chunk in chunks[1:]:
            reconstructed += chunk[CHUNK_OVERLAP:]
        assert reconstructed == text

    def test_empty_string_returns_single_empty_chunk(self):
        assert chunk_text("") == [""]

    def test_unicode_text_does_not_error(self):
        text = "日本語テキスト " * 300   # Unicode, well above CHUNK_SIZE
        chunks = chunk_text(text)
        assert len(chunks) > 1
        assert all(isinstance(c, str) for c in chunks)


# ── iso_to_unix() ─────────────────────────────────────────────────────────────

class TestIsoToUnix:

    def test_known_utc_timestamp(self):
        # 2024-01-01T00:00:00Z == 1704067200
        assert iso_to_unix("2024-01-01T00:00:00Z") == 1704067200

    def test_returns_integer(self):
        result = iso_to_unix("2024-09-10T09:15:00Z")
        assert isinstance(result, int)

    def test_later_date_has_higher_timestamp(self):
        old = iso_to_unix("2024-09-10T09:15:00Z")
        new = iso_to_unix("2025-11-28T16:45:00Z")
        assert new > old

    def test_handles_z_suffix(self):
        """Both 'Z' and '+00:00' forms parse to the same value."""
        assert iso_to_unix("2025-06-01T12:00:00Z") == iso_to_unix("2025-06-01T12:00:00+00:00")

    def test_sample_jira_dates_parse(self):
        """All dates in the sample Jira record parse without error."""
        iso_to_unix(SAMPLE_JIRA["created_date"])
        iso_to_unix(SAMPLE_JIRA["updated_date"])

    def test_sample_confluence_dates_parse(self):
        iso_to_unix(SAMPLE_CONFLUENCE["created_date"])
        iso_to_unix(SAMPLE_CONFLUENCE["updated_date"])


# ── content_hash() ────────────────────────────────────────────────────────────

class TestContentHash:

    def test_returns_64_char_hex_string(self):
        result = content_hash("hello")
        assert len(result) == 64
        assert all(c in "0123456789abcdef" for c in result)

    def test_same_text_same_hash(self):
        assert content_hash("same text") == content_hash("same text")

    def test_different_text_different_hash(self):
        assert content_hash("text A") != content_hash("text B")

    def test_matches_stdlib_sha256(self):
        text = "verify against stdlib"
        expected = hashlib.sha256(text.encode()).hexdigest()
        assert content_hash(text) == expected

    def test_whitespace_sensitive(self):
        """A trailing space produces a different hash — content is compared exactly."""
        assert content_hash("text") != content_hash("text ")

    def test_empty_string_produces_known_hash(self):
        # SHA-256 of "" is a well-known constant
        expected = hashlib.sha256(b"").hexdigest()
        assert content_hash("") == expected


# ── validate_record() ─────────────────────────────────────────────────────────

class TestValidateRecord:

    def test_valid_jira_record_returns_none(self):
        assert validate_record(SAMPLE_JIRA, "jira") is None

    def test_valid_confluence_record_returns_none(self):
        assert validate_record(SAMPLE_CONFLUENCE, "confluence") is None

    def test_missing_jira_field_returns_error_string(self):
        record = {k: v for k, v in SAMPLE_JIRA.items() if k != "description"}
        result = validate_record(record, "jira")
        assert result is not None
        assert "description" in result

    def test_missing_confluence_field_returns_error_string(self):
        record = {k: v for k, v in SAMPLE_CONFLUENCE.items() if k != "content"}
        result = validate_record(record, "confluence")
        assert result is not None
        assert "content" in result

    def test_missing_id_reported(self):
        record = {k: v for k, v in SAMPLE_JIRA.items() if k != "id"}
        result = validate_record(record, "jira")
        assert "id" in result

    def test_missing_created_date_reported(self):
        record = {k: v for k, v in SAMPLE_CONFLUENCE.items() if k != "created_date"}
        result = validate_record(record, "confluence")
        assert "created_date" in result

    def test_empty_dict_reports_all_required_fields(self):
        result = validate_record({}, "jira")
        for field in REQUIRED_JIRA_FIELDS:
            assert field in result

    def test_extra_fields_do_not_cause_errors(self):
        """Optional fields like trap_group, priority, space must not cause validation failures."""
        record = dict(SAMPLE_JIRA, extra_field="bonus")
        assert validate_record(record, "jira") is None

    def test_jira_does_not_require_content_field(self):
        """'content' is a Confluence field — Jira uses 'description'."""
        record = {k: v for k, v in SAMPLE_JIRA.items() if k != "content"}
        assert validate_record(record, "jira") is None

    def test_confluence_does_not_require_description_field(self):
        record = {k: v for k, v in SAMPLE_CONFLUENCE.items() if k != "description"}
        assert validate_record(record, "confluence") is None


# ── build_jira_entities() ─────────────────────────────────────────────────────

class TestBuildJiraEntities:

    def _build(self, record=None, n_chunks=1):
        record = record or SAMPLE_JIRA
        chunks = ["chunk text"] * n_chunks
        vectors = [FAKE_VECTOR] * n_chunks
        return build_jira_entities(record, chunks, vectors)

    def test_returns_one_entity_per_chunk(self):
        assert len(self._build(n_chunks=3)) == 3

    def test_chunk_index_is_sequential(self):
        entities = self._build(n_chunks=3)
        assert [e["chunk_index"] for e in entities] == [0, 1, 2]

    def test_source_type_is_jira(self):
        entity = self._build()[0]
        assert entity["source_type"] == "jira"

    def test_doc_id_matches_record(self):
        entity = self._build()[0]
        assert entity["doc_id"] == SAMPLE_JIRA["id"]

    def test_title_matches_record(self):
        entity = self._build()[0]
        assert entity["title"] == SAMPLE_JIRA["title"]

    def test_priority_mapped_correctly(self):
        entity = self._build()[0]
        assert entity["priority"] == SAMPLE_JIRA["priority"]

    def test_jira_type_mapped_correctly(self):
        entity = self._build()[0]
        assert entity["jira_type"] == SAMPLE_JIRA["type"]

    def test_space_is_empty_string_for_jira(self):
        entity = self._build()[0]
        assert entity["space"] == ""

    def test_trap_group_preserved(self):
        entity = self._build()[0]
        assert entity["trap_group"] == SAMPLE_JIRA["trap_group"]

    def test_trap_group_none_becomes_empty_string(self):
        record = dict(SAMPLE_JIRA, trap_group=None)
        entity = build_jira_entities(record, ["text"], [FAKE_VECTOR])[0]
        assert entity["trap_group"] == ""

    def test_created_date_is_unix_int(self):
        entity = self._build()[0]
        assert isinstance(entity["created_date"], int)
        assert entity["created_date"] == iso_to_unix(SAMPLE_JIRA["created_date"])

    def test_updated_date_is_unix_int(self):
        entity = self._build()[0]
        assert isinstance(entity["updated_date"], int)
        assert entity["updated_date"] == iso_to_unix(SAMPLE_JIRA["updated_date"])

    def test_embedding_vector_stored(self):
        entity = self._build()[0]
        assert entity["embedding"] == FAKE_VECTOR

    def test_content_capped_at_4096_chars(self):
        long_chunk = "x" * 5000
        entities = build_jira_entities(SAMPLE_JIRA, [long_chunk], [FAKE_VECTOR])
        assert len(entities[0]["content"]) == 4096

    def test_title_capped_at_512_chars(self):
        record = dict(SAMPLE_JIRA, title="t" * 600)
        entity = build_jira_entities(record, ["chunk"], [FAKE_VECTOR])[0]
        assert len(entity["title"]) == 512

    def test_all_required_milvus_fields_present(self):
        expected_keys = {
            "doc_id", "chunk_index", "source_type", "title", "content",
            "space", "priority", "jira_type", "created_date", "updated_date",
            "trap_group", "embedding",
        }
        entity = self._build()[0]
        assert set(entity.keys()) == expected_keys


# ── build_confluence_entities() ───────────────────────────────────────────────

class TestBuildConfluenceEntities:

    def _build(self, record=None, n_chunks=1):
        record = record or SAMPLE_CONFLUENCE
        chunks = ["chunk text"] * n_chunks
        vectors = [FAKE_VECTOR] * n_chunks
        return build_confluence_entities(record, chunks, vectors)

    def test_returns_one_entity_per_chunk(self):
        assert len(self._build(n_chunks=2)) == 2

    def test_source_type_is_confluence(self):
        assert self._build()[0]["source_type"] == "confluence"

    def test_doc_id_matches_record(self):
        assert self._build()[0]["doc_id"] == SAMPLE_CONFLUENCE["id"]

    def test_space_mapped_correctly(self):
        assert self._build()[0]["space"] == SAMPLE_CONFLUENCE["space"]

    def test_priority_is_empty_string_for_confluence(self):
        assert self._build()[0]["priority"] == ""

    def test_jira_type_is_empty_string_for_confluence(self):
        assert self._build()[0]["jira_type"] == ""

    def test_trap_group_preserved(self):
        assert self._build()[0]["trap_group"] == SAMPLE_CONFLUENCE["trap_group"]

    def test_trap_group_none_becomes_empty_string(self):
        record = dict(SAMPLE_CONFLUENCE, trap_group=None)
        entity = build_confluence_entities(record, ["text"], [FAKE_VECTOR])[0]
        assert entity["trap_group"] == ""

    def test_dates_are_unix_ints(self):
        entity = self._build()[0]
        assert entity["created_date"] == iso_to_unix(SAMPLE_CONFLUENCE["created_date"])
        assert entity["updated_date"] == iso_to_unix(SAMPLE_CONFLUENCE["updated_date"])

    def test_space_capped_at_128_chars(self):
        record = dict(SAMPLE_CONFLUENCE, space="S" * 200)
        entity = build_confluence_entities(record, ["chunk"], [FAKE_VECTOR])[0]
        assert len(entity["space"]) == 128

    def test_missing_space_defaults_to_empty_string(self):
        record = {k: v for k, v in SAMPLE_CONFLUENCE.items() if k != "space"}
        entity = build_confluence_entities(record, ["chunk"], [FAKE_VECTOR])[0]
        assert entity["space"] == ""

    def test_all_required_milvus_fields_present(self):
        expected_keys = {
            "doc_id", "chunk_index", "source_type", "title", "content",
            "space", "priority", "jira_type", "created_date", "updated_date",
            "trap_group", "embedding",
        }
        assert set(self._build()[0].keys()) == expected_keys


# ── ingest_records() — pipeline flow (all I/O mocked) ─────────────────────────

def _make_mock_collection():
    col = MagicMock()
    col.insert.return_value = MagicMock()
    col.flush.return_value = None
    return col


def _make_mock_session(existing_doc=None):
    session = MagicMock()
    session.get.return_value = existing_doc   # None → new doc; object → existing
    session.merge.return_value = None
    session.commit.return_value = None
    session.close.return_value = None
    return session


class TestIngestRecords:
    """Tests for ingest_records() with Milvus, Postgres, and Ollama fully mocked."""

    def _run(self, records, source_type="jira", dry_run=False,
             existing_doc=None, embed_side_effect=None):
        """Helper: run ingest_records() with all external calls patched."""
        from scripts.ingest import ingest_records

        mock_collection = _make_mock_collection()
        mock_session = _make_mock_session(existing_doc)

        embed_return = [FAKE_VECTOR] * 8  # default: batch returns up to 8 vectors

        with patch("scripts.ingest.get_session", return_value=mock_session), \
             patch("scripts.ingest.embed_batch",
                   side_effect=embed_side_effect or (lambda batch, **kw: [FAKE_VECTOR] * len(batch))):

            processed, skipped, errors = ingest_records(
                records, source_type, mock_collection, "run-001", dry_run
            )

        return processed, skipped, errors, mock_collection, mock_session

    # ── Happy path ────────────────────────────────────────────────────────────

    def test_new_jira_record_is_processed(self):
        p, s, e, col, _ = self._run([SAMPLE_JIRA], "jira")
        assert p == 1
        assert s == 0
        assert e == 0

    def test_new_confluence_record_is_processed(self):
        p, s, e, col, _ = self._run([SAMPLE_CONFLUENCE], "confluence")
        assert p == 1
        assert s == 0
        assert e == 0

    def test_milvus_insert_called_once_per_record(self):
        _, _, _, col, _ = self._run([SAMPLE_JIRA], "jira")
        col.insert.assert_called_once()

    def test_milvus_flush_called_after_batch(self):
        _, _, _, col, _ = self._run([SAMPLE_JIRA], "jira")
        col.flush.assert_called_once()

    def test_postgres_merge_called_for_new_doc(self):
        _, _, _, _, session = self._run([SAMPLE_JIRA], "jira")
        session.merge.assert_called_once()

    def test_postgres_commit_called(self):
        _, _, _, _, session = self._run([SAMPLE_JIRA], "jira")
        session.commit.assert_called_once()

    def test_multiple_records_all_processed(self):
        jira2 = dict(SAMPLE_JIRA, id="JIRA-1002",
                     description="Second bug report about a different component.")
        p, s, e, col, _ = self._run([SAMPLE_JIRA, jira2], "jira")
        assert p == 2
        assert col.insert.call_count == 2

    # ── Deduplication ─────────────────────────────────────────────────────────

    def test_unchanged_document_is_skipped(self):
        """When content_hash matches an existing Postgres record, skip re-embedding."""
        from scripts.ingest import content_hash as ch
        existing = MagicMock()
        existing.content_hash = ch(SAMPLE_JIRA["description"])
        p, s, e, col, _ = self._run([SAMPLE_JIRA], "jira", existing_doc=existing)
        assert s == 1
        assert p == 0
        col.insert.assert_not_called()

    def test_changed_document_is_re_ingested(self):
        """When content_hash differs from the stored value, the doc is re-embedded."""
        existing = MagicMock()
        existing.content_hash = "old_hash_that_does_not_match"
        p, s, e, col, _ = self._run([SAMPLE_JIRA], "jira", existing_doc=existing)
        assert p == 1
        col.insert.assert_called_once()

    # ── Dry run ───────────────────────────────────────────────────────────────

    def test_dry_run_counts_processed_without_writing(self):
        p, s, e, col, session = self._run([SAMPLE_JIRA], "jira", dry_run=True)
        assert p == 1
        col.insert.assert_not_called()
        session.merge.assert_not_called()

    def test_dry_run_does_not_skip_unchanged_docs(self):
        """dry_run always counts every valid record — deduplication is bypassed."""
        existing = MagicMock()
        existing.content_hash = "anything"
        p, s, e, col, _ = self._run([SAMPLE_JIRA], "jira",
                                     dry_run=True, existing_doc=existing)
        assert p == 1
        assert s == 0

    # ── Validation errors ─────────────────────────────────────────────────────

    def test_invalid_record_counted_as_error(self):
        bad_record = {"id": "JIRA-BAD"}   # missing required fields
        p, s, e, col, _ = self._run([bad_record], "jira")
        assert e == 1
        assert p == 0
        col.insert.assert_not_called()

    def test_valid_and_invalid_mixed(self):
        bad = {"id": "JIRA-BAD"}
        p, s, e, col, _ = self._run([SAMPLE_JIRA, bad], "jira")
        assert p == 1
        assert e == 1

    # ── Embedding failure ─────────────────────────────────────────────────────

    def test_embedding_failure_counted_as_error(self):
        def bad_embed(batch, **kw):
            raise RuntimeError("Ollama unavailable")

        p, s, e, col, _ = self._run([SAMPLE_JIRA], "jira",
                                     embed_side_effect=bad_embed)
        assert e == 1
        assert p == 0
        col.insert.assert_not_called()

    def test_one_embedding_failure_does_not_abort_others(self):
        """An error on one record should not stop ingestion of subsequent records."""
        jira2 = dict(SAMPLE_JIRA, id="JIRA-1002",
                     description="Second ticket with no embedding issues.")
        call_count = {"n": 0}

        def flaky_embed(batch, **kw):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise RuntimeError("first call fails")
            return [FAKE_VECTOR] * len(batch)

        p, s, e, col, _ = self._run([SAMPLE_JIRA, jira2], "jira",
                                     embed_side_effect=flaky_embed)
        assert e == 1
        assert p == 1

    # ── Milvus insert failure ─────────────────────────────────────────────────

    def test_milvus_insert_failure_counted_as_error(self):
        from scripts.ingest import ingest_records
        col = _make_mock_collection()
        col.insert.side_effect = Exception("Milvus write error")
        session = _make_mock_session()

        with patch("scripts.ingest.get_session", return_value=session), \
             patch("scripts.ingest.embed_batch",
                   side_effect=lambda b, **kw: [FAKE_VECTOR] * len(b)):
            p, s, e = ingest_records([SAMPLE_JIRA], "jira", col, "run-1", False)

        assert e == 1
        assert p == 0

    # ── Empty input ───────────────────────────────────────────────────────────

    def test_empty_record_list_returns_zeros(self):
        p, s, e, col, _ = self._run([], "jira")
        assert (p, s, e) == (0, 0, 0)
        col.insert.assert_not_called()

    # ── Milvus insert payload format ──────────────────────────────────────────

    def test_milvus_receives_column_oriented_dict(self):
        """Milvus insert() expects {field: [val, val, ...]} not [{field: val}, ...]"""
        _, _, _, col, _ = self._run([SAMPLE_JIRA], "jira")
        args, _ = col.insert.call_args
        column_data = args[0]
        assert isinstance(column_data, dict)
        assert "embedding" in column_data
        assert isinstance(column_data["embedding"], list)
        assert isinstance(column_data["doc_id"], list)

    def test_confluence_source_type_in_milvus_payload(self):
        _, _, _, col, _ = self._run([SAMPLE_CONFLUENCE], "confluence")
        args, _ = col.insert.call_args
        assert col.insert.call_args[0][0]["source_type"] == ["confluence"]
