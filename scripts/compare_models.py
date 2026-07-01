#!/usr/bin/env python3
"""Compare two OpenRouter models (e.g., Qwen vs Gemma) live on the KnowOps retrieval traps."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Allow importing from src/
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


def load_traps(limit: int | None) -> list[dict]:
    traps = json.loads(MANIFEST_PATH.read_text())
    return traps[:limit] if limit else traps


def get_verdict(candidate: Candidate | None, trap: dict) -> str:
    if candidate is None:
        return "OTHER"
    if candidate.doc_id == trap["correct_document_id"]:
        return "CORRECT"
    if candidate.doc_id in set(trap.get("outdated_document_ids", [])):
        return "STALE"
    return "OTHER"


def run_model_on_traps(model_name: str, traps: list[dict]) -> list[dict]:
    print(f"\nEvaluating model: {model_name}...")
    pipeline = Pipeline(settings=SETTINGS, offline=False)
    # Override client model
    pipeline.llm.model = model_name

    results = []
    for i, trap in enumerate(traps, 1):
        print(f"[{i}/{len(traps)}] Querying: '{trap['topic'][:50]}...'")
        try:
            # Retrieve fixed path (runs the planner, retriever, and reranker agents)
            selected, plan, debug = pipeline.retrieve_fixed(trap["topic"])
            top_candidate = selected[0] if selected else None
            verdict = get_verdict(top_candidate, trap)

            results.append({
                "trap_group": trap["trap_group"],
                "plan": plan,
                "top_doc": top_candidate.doc_id if top_candidate else "<none>",
                "verdict": verdict,
                "updated": top_candidate.updated_date if top_candidate else "-",
                "semantic_score": top_candidate.semantic_score if top_candidate else 0.0,
                "hybrid_score": top_candidate.hybrid_score if top_candidate else 0.0,
                "error": None
            })
        except Exception as e:
            print(f"  Error querying {model_name}: {e}")
            results.append({
                "trap_group": trap["trap_group"],
                "plan": {},
                "top_doc": "<error>",
                "verdict": "ERROR",
                "updated": "-",
                "semantic_score": 0.0,
                "hybrid_score": 0.0,
                "error": str(e)
            })
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two OpenRouter LLM models on the retrieval traps.")
    parser.add_argument("--model-a", default="qwen/qwen-2.5-7b-instruct", help="First model to evaluate (Model A)")
    parser.add_argument("--model-b", default="google/gemma-2-9b-it", help="Second model to evaluate (Model B)")
    parser.add_argument("--limit", type=int, help="Limit number of traps to evaluate.")
    args = parser.parse_args()

    # Ensure API Key is present
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        print("Error: OPENROUTER_API_KEY is not set in environment or .env file.")
        sys.exit(1)

    traps = load_traps(args.limit)
    print(f"Starting comparison: Model A ({args.model_a}) vs Model B ({args.model_b})")
    print(f"Total Traps to test: {len(traps)}")
    print("=" * 100)

    # Disable verbose logs to keep CLI clean during evaluation
    logging.getLogger("knowops.llm").setLevel(logging.WARNING)
    logging.getLogger("knowops.planner").setLevel(logging.WARNING)
    logging.getLogger("knowops.retriever").setLevel(logging.WARNING)
    logging.getLogger("knowops.reranker").setLevel(logging.WARNING)

    results_a = run_model_on_traps(args.model_a, traps)
    results_b = run_model_on_traps(args.model_b, traps)

    # Re-enable info logging for printing comparison output
    logging.getLogger("knowops.llm").setLevel(logging.INFO)

    print("\n" + "=" * 100)
    print(f"COMPARISON TABLE: {args.model_a} (A) vs {args.model_b} (B)")
    print("=" * 100)
    print(f"{'Trap Group':<26} | {'Plan A (Sens)':<15} | {'Doc A':<9} | {'Verdict A':<9} || {'Plan B (Sens)':<15} | {'Doc B':<9} | {'Verdict B':<9}")
    print("-" * 100)

    correct_a = correct_b = 0
    stale_a = stale_b = 0

    for r_a, r_b in zip(results_a, results_b):
        plan_a_str = f"{r_a['plan'].get('intent', '??')[:5]}({r_a['plan'].get('time_sensitivity', '??')[:4]})"
        plan_b_str = f"{r_b['plan'].get('intent', '??')[:5]}({r_b['plan'].get('time_sensitivity', '??')[:4]})"

        # Highlight correct/incorrect verdicts
        v_a = f"✅ {r_a['verdict']}" if r_a['verdict'] == "CORRECT" else f"❌ {r_a['verdict']}"
        v_b = f"✅ {r_b['verdict']}" if r_b['verdict'] == "CORRECT" else f"❌ {r_b['verdict']}"

        print(f"{r_a['trap_group'][:26]:<26} | {plan_a_str:<15} | {r_a['top_doc']:<9} | {v_a:<9} || {plan_b_str:<15} | {r_b['top_doc']:<9} | {v_b:<9}")

        if r_a["verdict"] == "CORRECT":
            correct_a += 1
        elif r_a["verdict"] == "STALE":
            stale_a += 1

        if r_b["verdict"] == "CORRECT":
            correct_b += 1
        elif r_b["verdict"] == "STALE":
            stale_b += 1

    print("=" * 100)
    print("FINAL SCORES:")
    print("-" * 100)
    print(f"Model A ({args.model_a}):")
    print(f"  - Correct Retrievals: {correct_a}/{len(traps)}")
    print(f"  - Stale/Outdated Retrievals: {stale_a}/{len(traps)}")
    print(f"Model B ({args.model_b}):")
    print(f"  - Correct Retrievals: {correct_b}/{len(traps)}")
    print(f"  - Stale/Outdated Retrievals: {stale_b}/{len(traps)}")
    print("=" * 100)

    winner = "Tie!"
    if correct_a > correct_b:
        winner = args.model_a
    elif correct_b > correct_a:
        winner = args.model_b
    print(f"Winner: {winner}")


if __name__ == "__main__":
    main()
