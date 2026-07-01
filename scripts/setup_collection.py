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

# Allow `import knowops` from the project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pymilvus import MilvusClient
from knowops.schema import (
    COLLECTION_NAME, INDEX_PARAMS, VECTOR_DIM, build_milvus_schema,
)


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

    client = MilvusClient(host=args.host, port=args.port)
    print(f"Connected to Milvus at {args.host}:{args.port}")

    if client.has_collection(COLLECTION_NAME):
        if not args.force:
            print(
                f"Collection '{COLLECTION_NAME}' already exists. "
                "Use --force to drop and recreate."
            )
            client.load_collection(COLLECTION_NAME)
            print(f"Collection '{COLLECTION_NAME}' loaded — ready for query")
            return
        
        client.drop_collection(COLLECTION_NAME)
        print(f"Dropped existing collection '{COLLECTION_NAME}'")

    schema = build_milvus_schema()
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="embedding",
        index_type=INDEX_PARAMS["index_type"],
        metric_type=INDEX_PARAMS["metric_type"],
        params=INDEX_PARAMS["params"],
    )
    # Add scalar indexes for fast filtering
    index_params.add_index(field_name="doc_id")
    index_params.add_index(field_name="source_type")
    index_params.add_index(field_name="updated_date")

    client.create_collection(
        collection_name=COLLECTION_NAME,
        schema=schema,
        index_params=index_params,
    )
    print(f"Created collection '{COLLECTION_NAME}' (dim={VECTOR_DIM})")
    
    client.load_collection(COLLECTION_NAME)
    print(f"Collection '{COLLECTION_NAME}' loaded — ready for ingestion")
    print("Setup complete.")


if __name__ == "__main__":
    main()
