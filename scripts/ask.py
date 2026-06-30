#!/usr/bin/env python3
"""Ask KnowOps free-form questions from one-shot CLI args or an interactive REPL."""

from __future__ import annotations

import argparse
import os
import sys

# Allow `import knowops` when the script is executed from the scripts directory.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

from knowops.config import SETTINGS
from knowops.pipeline import Pipeline, PipelineResult


def env_truthy(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def choose_offline(args: argparse.Namespace) -> bool:
    """Default to offline when explicitly requested or no LLM key is configured."""
    if args.offline:
        return True
    if args.live:
        return False
    return env_truthy("KNOWOPS_OFFLINE") or not os.getenv("OPENROUTER_API_KEY", "").strip()


def print_verbose(result: PipelineResult) -> None:
    plan = result.plan
    print("\nPlanner:")
    print(
        f"  intent={plan.get('intent')} | time_sensitivity={plan.get('time_sensitivity')} | "
        f"source={plan.get('source_preference')} | recency_required={plan.get('recency_required')}"
    )
    print("Selected documents:")
    for doc in result.selected:
        print(f"  - {doc.doc_id} | updated={doc.updated_date} | hybrid={doc.hybrid_score:.3f}")
    rr = result.reranker_out
    print("Reranker:")
    print(f"  reasoning={rr.get('reasoning')}")
    print(f"  freshness_warning={rr.get('freshness_warning')}")
    if rr.get("warning_message"):
        print(f"  warning_message={rr.get('warning_message')}")


def answer_question(pipeline: Pipeline, question: str, verbose: bool) -> None:
    result = pipeline.run(question)
    if verbose:
        print_verbose(result)
        print("\nAnswer:")
    print(result.answer)


def run_repl(pipeline: Pipeline, verbose: bool) -> None:
    while True:
        try:
            question = input("knowops> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not question or question.lower() in {"exit", "quit"}:
            return
        answer_question(pipeline, question, verbose)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ask the KnowOps agentic RAG pipeline a question.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--offline", action="store_true", help="Use deterministic offline mode.")
    mode.add_argument("--live", action="store_true", help="Use live services.")
    parser.add_argument("--verbose", action="store_true", help="Print plan, selected docs, and reranker details.")
    parser.add_argument("question", nargs="*", help="Question to answer once; omit to start a REPL.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    pipeline = Pipeline(settings=SETTINGS, offline=choose_offline(args))
    print(f"KnowOps ask — mode={pipeline.mode}")

    question = " ".join(args.question).strip()
    if question:
        answer_question(pipeline, question, args.verbose)
    else:
        run_repl(pipeline, args.verbose)


if __name__ == "__main__":
    main()
