#!/usr/bin/env python3
"""
Build Order Step 1 — Create the Milvus collection with schema and HNSW index.

Run once before the first ingestion:
    python scripts/setup_collection.py

To wipe and recreate (e.g. after a schema change):
    python scripts/setup_collection.py --force
"""

import argparse
import os
import sys

from pymilvus import connections, utility, Collection

# Allow `import knowops` from the project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from knowops.schema import (
    COLLECTION_NAME, INDEX_PARAMS, VECTOR_DIM, build_schema,
)


def connect_milvus(host: str, port: str) -> None:
    """Open a connection to the Milvus gRPC endpoint."""
    connections.connect(alias="default", host=host, port=port)
    print(f"Connected to Milvus at {host}:{port}")


def drop_collection_if_exists(name: str) -> None:
    """Drop the named collection when --force is used."""
    if utility.has_collection(name):
        utility.drop_collection(name)
        print(f"Dropped existing collection '{name}'")


def create_collection(force: bool) -> Collection:
    """Create the knowops_documents collection and build the HNSW index."""
    if utility.has_collection(COLLECTION_NAME):
        if not force:
            print(
                f"Collection '{COLLECTION_NAME}' already exists. "
                "Use --force to drop and recreate."
            )
            return Collection(COLLECTION_NAME)
        drop_collection_if_exists(COLLECTION_NAME)

    schema = build_schema()
    collection = Collection(name=COLLECTION_NAME, schema=schema)
    print(f"Created collection '{COLLECTION_NAME}' (dim={VECTOR_DIM})")

    # Build HNSW index on the embedding field.
    # Why HNSW instead of IVF_FLAT:
    #   IVF_FLAT with low nlist causes O(n) search on large collections,
    #   producing the timeout bug this project demonstrates and fixes.
    #   HNSW provides sub-linear query time with no per-query tuning.
    collection.create_index(field_name="embedding", index_params=INDEX_PARAMS)
    print(
        f"Built HNSW index — M={INDEX_PARAMS['params']['M']}, "
        f"ef_construction={INDEX_PARAMS['params']['ef_construction']}"
    )

    # Build scalar indexes for fast metadata filtering
    for field in ("doc_id", "source_type", "updated_date"):
        collection.create_index(field_name=field, index_name=f"idx_{field}")
    print("Built scalar indexes on doc_id, source_type, updated_date")

    # Load the collection into memory so it is immediately queryable
    collection.load()
    print(f"Collection '{COLLECTION_NAME}' loaded — ready for ingestion")

    return collection


def main() -> None:
    parser = argparse.ArgumentParser(description="Create Milvus collection for KnowOps")
    parser.add_argument("--host",  default=os.getenv("MILVUS_HOST", "localhost"))
    parser.add_argument("--port",  default=os.getenv("MILVUS_PORT", "19530"))
    parser.add_argument(
        "--force",
        action="store_true",
        help="Drop and recreate the collection if it already exists (data loss!)",
    )
    args = parser.parse_args()

    connect_milvus(args.host, args.port)
    create_collection(force=args.force)
    print("Setup complete.")


if __name__ == "__main__":
    main()
