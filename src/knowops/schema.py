"""
Milvus collection schema and index configuration.

Schema design decisions:
  - Single collection for both Jira and Confluence; source_type field
    acts as a filter discriminator instead of separate collections.
  - Dates stored as INT64 Unix timestamps for efficient range filtering
    and arithmetic in the recency-weighted reranker.
  - Empty string (not null) is used for optional fields (space, priority,
    jira_type, trap_group) to maximise compatibility across Milvus versions.
"""

from pymilvus import CollectionSchema, FieldSchema, DataType, MilvusClient

COLLECTION_NAME = "knowops_documents"
VECTOR_DIM = 1024  # bge-m3 output dimension

# ── Index configuration ──────────────────────────────────────────────────────
# HNSW is the production-grade choice: sub-linear query time, near-perfect
# recall, and no query-time nprobe tuning needed (unlike IVF_FLAT).
# M=16, ef_construction=200 are the recommended defaults for most workloads.
INDEX_PARAMS = {
    "index_type": "HNSW",
    "metric_type": "COSINE",
    "params": {"M": 16, "efConstruction": 200},  # camelCase required by Milvus
}

# ef ≥ top_k is required; ef=64 provides high recall for top_k up to 20.
SEARCH_PARAMS = {"metric_type": "COSINE", "params": {"ef": 64}}


def build_schema() -> CollectionSchema:
    """Return the Milvus CollectionSchema for knowops_documents."""
    fields = [
        FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="chunk_index", dtype=DataType.INT32),
        FieldSchema(
            name="source_type", dtype=DataType.VARCHAR, max_length=32
        ),  # "jira" | "confluence"
        FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=4096),
        FieldSchema(
            name="space", dtype=DataType.VARCHAR, max_length=128
        ),  # "" for Jira docs
        FieldSchema(
            name="priority", dtype=DataType.VARCHAR, max_length=16
        ),  # "" for Confluence
        FieldSchema(
            name="jira_type", dtype=DataType.VARCHAR, max_length=32
        ),  # "" for Confluence
        FieldSchema(name="created_date", dtype=DataType.INT64),  # Unix timestamp (UTC)
        FieldSchema(name="updated_date", dtype=DataType.INT64),  # Unix timestamp (UTC)
        FieldSchema(
            name="trap_group", dtype=DataType.VARCHAR, max_length=64
        ),  # "" if not a trap doc
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=VECTOR_DIM),
    ]
    return CollectionSchema(
        fields=fields,
        description="KnowOps: Jira + Confluence document embeddings",
        enable_dynamic_field=False,
    )


# Field names returned with every search hit (excludes the vector itself).
OUTPUT_FIELDS = [
    "doc_id",
    "chunk_index",
    "source_type",
    "title",
    "content",
    "space",
    "priority",
    "jira_type",
    "created_date",
    "updated_date",
    "trap_group",
]


def build_milvus_schema():
    """Return a MilvusClient-compatible schema for knowops_documents.

    Uses the MilvusClient.create_schema() builder API (PyMilvus >= 2.4 / 3.x).
    """
    schema = MilvusClient.create_schema(auto_id=True, enable_dynamic_field=False)
    schema.add_field(field_name="pk",           datatype=DataType.INT64,        is_primary=True)
    schema.add_field(field_name="doc_id",        datatype=DataType.VARCHAR,      max_length=64)
    schema.add_field(field_name="chunk_index",   datatype=DataType.INT32)
    schema.add_field(field_name="source_type",   datatype=DataType.VARCHAR,      max_length=32)
    schema.add_field(field_name="title",         datatype=DataType.VARCHAR,      max_length=512)
    schema.add_field(field_name="content",       datatype=DataType.VARCHAR,      max_length=4096)
    schema.add_field(field_name="space",         datatype=DataType.VARCHAR,      max_length=128)
    schema.add_field(field_name="priority",      datatype=DataType.VARCHAR,      max_length=16)
    schema.add_field(field_name="jira_type",     datatype=DataType.VARCHAR,      max_length=32)
    schema.add_field(field_name="created_date",  datatype=DataType.INT64)
    schema.add_field(field_name="updated_date",  datatype=DataType.INT64)
    schema.add_field(field_name="trap_group",    datatype=DataType.VARCHAR,      max_length=64)
    schema.add_field(field_name="embedding",     datatype=DataType.FLOAT_VECTOR, dim=VECTOR_DIM)
    return schema
