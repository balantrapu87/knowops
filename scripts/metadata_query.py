#!/usr/bin/env python3
"""
Showcase the relational-metadata queries that are trivial in Postgres but
awkward in a pure vector store. Live-only (requires Postgres + ingested data).

Examples:
    python scripts/metadata_query.py --recent-bugs --days 30
    python scripts/metadata_query.py --trap-group trap-milvus-index-type

This is the motivation for METADATA_BACKEND=postgres: the vector store handles
similarity; the relational store answers "which high-priority bugs changed in
the last N days?" or "what else belongs to this topic group?" in one SQL query.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv

from knowops.db import (
    check_postgres_health, recent_docs_by_priority, docs_in_trap_group,
)

load_dotenv()


def _print_rows(rows: list[dict]) -> None:
    if not rows:
        print("  (no matching documents)")
        return
    for r in rows:
        from datetime import datetime, timezone
        updated = datetime.fromtimestamp(r["updated_ts"], timezone.utc).strftime("%Y-%m-%d")
        extra = f" priority={r['priority']}" if r.get("priority") else ""
        print(f"  {r['id']:10} updated={updated} [{r['source_type']}]{extra}  {r['title']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="KnowOps Postgres metadata queries")
    parser.add_argument("--recent-bugs", action="store_true",
                        help="List high-priority bugs updated in the last --days")
    parser.add_argument("--priority", default="high")
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--trap-group", default=None,
                        help="List all documents sharing a trap_group")
    args = parser.parse_args()

    if not check_postgres_health():
        sys.exit("ERROR: Postgres is not reachable. Start the stack and run ingest first.")

    if args.recent_bugs:
        print(f"High-priority bugs updated in the last {args.days} days:")
        _print_rows(recent_docs_by_priority(priority=args.priority, days=args.days))

    if args.trap_group:
        print(f"\nDocuments in trap group '{args.trap_group}':")
        _print_rows(docs_in_trap_group(args.trap_group))

    if not args.recent_bugs and not args.trap_group:
        parser.print_help()


if __name__ == "__main__":
    main()
