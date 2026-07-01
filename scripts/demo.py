#!/usr/bin/env python3
"""Show KnowOps semantic-only retrieval before and recency-aware retrieval after."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Allow `import knowops` when the script is executed from the scripts directory.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

from knowops.config import SETTINGS
from knowops.pipeline import Pipeline
from knowops.search import Candidate

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = PROJECT_ROOT / "data" / "trap_manifest.json"


def env_truthy(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def choose_offline(args: argparse.Namespace) -> bool:
    """Default to offline when explicitly requested or no LLM key is configured."""
    if args.offline:
        return True
    if args.live:
        return False
    return env_truthy("KNOWOPS_OFFLINE") or not os.getenv("OPENROUTER_API_KEY", "").strip()


def load_traps(limit: int | None) -> list[dict]:
    traps = json.loads(MANIFEST_PATH.read_text())
    return traps[:limit] if limit else traps


def verdict(candidate: Candidate | None, trap: dict) -> tuple[str, bool, bool]:
    if candidate is None:
        return "❌ OTHER", False, False
    if candidate.doc_id == trap["correct_document_id"]:
        return "✅ CORRECT", True, False
    if candidate.doc_id in set(trap.get("outdated_document_ids", [])):
        return "❌ STALE(outdated)", False, True
    return "❌ OTHER", False, False


def candidate_by_id(candidates: list[Candidate], doc_id: str) -> Candidate | None:
    return next((c for c in candidates if c.doc_id == doc_id), None)


def print_result_block(trap: dict, baseline_top: Candidate | None, fixed_top: Candidate | None, debug: dict) -> tuple[bool, bool, bool]:
    base_label, base_ok, base_stale = verdict(baseline_top, trap)
    fixed_label, fixed_ok, _ = verdict(fixed_top, trap)

    print(f"\n{trap['trap_group']}")
    print(f"Query: {trap['topic']}")
    print("-" * 96)
    print(f"{'Path':<28} {'Doc ID':<12} {'Updated':<12} Verdict")
    print(f"{'BASELINE (semantic-only)':<28} {doc_id(baseline_top):<12} {updated(baseline_top):<12} {base_label}")
    print(f"{'FIXED (hybrid+recency)':<28} {doc_id(fixed_top):<12} {updated(fixed_top):<12} {fixed_label}")

    candidates = debug.get("candidates", [])
    correct = candidate_by_id(candidates, trap["correct_document_id"])
    stale = next(
        (candidate_by_id(candidates, doc_id) for doc_id in trap.get("outdated_document_ids", []) if candidate_by_id(candidates, doc_id)),
        None,
    )
    print("Score breakdown from fixed candidate list:")
    print(f"{'Role':<10} {'Doc ID':<12} {'Semantic':>9} {'Freshness':>10} {'Hybrid':>9}")
    print_score("correct", correct)
    print_score("stale", stale)
    return base_ok, fixed_ok, base_stale


def doc_id(candidate: Candidate | None) -> str:
    return candidate.doc_id if candidate else "<none>"


def updated(candidate: Candidate | None) -> str:
    return candidate.updated_date if candidate else "-"


def print_score(role: str, candidate: Candidate | None) -> None:
    if candidate is None:
        print(f"{role:<10} {'<not ranked>':<12} {'-':>9} {'-':>10} {'-':>9}")
        return
    print(
        f"{role:<10} {candidate.doc_id:<12} "
        f"{candidate.semantic_score:>9.3f} {candidate.freshness_score:>10.3f} {candidate.hybrid_score:>9.3f}"
    )


def print_full_examples(pipeline: Pipeline, traps: list[dict]) -> None:
    print("\nFULL PIPELINE EXAMPLES")
    print("=" * 96)
    for trap in traps[:2]:
        result = pipeline.run(trap["topic"])
        plan = result.plan
        print(f"\nQuestion: {result.question}")
        print(
            "Planner: "
            f"intent={plan.get('intent')} | time_sensitivity={plan.get('time_sensitivity')} | "
            f"source={plan.get('source_preference')} | recency_required={plan.get('recency_required')}"
        )
        print(f"Freshness warning: {result.freshness_warning}")
        print("Answer:")
        print(result.answer)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Demonstrate KnowOps recency-aware retrieval.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--offline", action="store_true", help="Use deterministic offline mode.")
    mode.add_argument("--live", action="store_true", help="Use live services.")
    parser.add_argument("--limit", type=int, help="Limit the number of manifest traps to run.")
    parser.add_argument("--full", action="store_true", help="Also run full agentic answers for two traps.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    offline = choose_offline(args)
    pipeline = Pipeline(settings=SETTINGS, offline=offline)
    traps = load_traps(args.limit)

    print(f"KnowOps retrieval demo — mode={pipeline.mode} — traps={len(traps)}")
    print("=" * 96)

    baseline_correct = fixed_correct = baseline_stale = 0
    for trap in traps:
        baseline = pipeline.retrieve_baseline(trap["topic"])
        fixed_sel, _plan, debug = pipeline.retrieve_fixed(trap["topic"])
        base_ok, fixed_ok, base_stale = print_result_block(
            trap,
            baseline[0] if baseline else None,
            fixed_sel[0] if fixed_sel else None,
            debug,
        )
        baseline_correct += int(base_ok)
        fixed_correct += int(fixed_ok)
        baseline_stale += int(base_stale)

    total = len(traps)
    print("\nSUMMARY")
    print("=" * 96)
    print(f"Baseline correct: {baseline_correct}/{total}  |  Fixed correct: {fixed_correct}/{total}")
    print(f"Traps where baseline returned a STALE document: {baseline_stale}/{total}")

    if args.full:
        print_full_examples(pipeline, traps)


if __name__ == "__main__":
    main()
