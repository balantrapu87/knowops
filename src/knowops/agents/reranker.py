"""Reranker agent — selects the top documents by hybrid score.

Ordering is decided in code (hybrid_score = semantic + recency), so the
"newer correct doc beats older stale doc" guarantee never depends on an LLM.
In live mode the LLM only writes the human-readable reasoning string.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from knowops.agents.base import Agent
from knowops.search import Candidate

log = logging.getLogger("knowops.reranker")


def _now_ts(now: Optional[float]) -> float:
    return now if now is not None else datetime.now(timezone.utc).timestamp()


class Reranker(Agent):
    prompt_filename = "reranker.md"

    def run(
        self, query: str, candidates: list[Candidate], now: Optional[float] = None
    ) -> tuple[list[Candidate], dict]:
        ranked = sorted(candidates, key=lambda c: c.hybrid_score, reverse=True)
        top = ranked[: self.settings.reranker_top_k]

        log.info("[Reranker] selected top-%d docs from %d candidates:",
                 len(top), len(candidates))
        for c in top:
            log.info("   %s | hybrid=%.3f | updated=%s | %s",
                     c.doc_id, c.hybrid_score, c.updated_date, c.title[:60])

        warning, message = self._freshness_warning(top, now)
        if warning:
            log.info("[Reranker] ⚠ freshness warning: %s", message)
        reasoning = self._reasoning(query, top, warning)

        out = {
            "selected_ids": [c.doc_id for c in top],
            "reasoning": reasoning,
            "freshness_warning": warning,
            "warning_message": message,
        }
        return top, out

    def _freshness_warning(
        self, top: list[Candidate], now: Optional[float]
    ) -> tuple[bool, Optional[str]]:
        if not top:
            return False, None
        age_days = (_now_ts(now) - top[0].updated_ts) / 86400.0
        if age_days > self.settings.freshness_warning_days:
            return True, (
                f"The most recent available document is ~{int(age_days)} days old; "
                "the information may be outdated."
            )
        return False, None

    def _reasoning(self, query: str, top: list[Candidate], warning: bool) -> str:
        if not top:
            return "No candidates met the retrieval criteria."
        if not self.offline:
            llm_reason = self._reasoning_llm(query, top)
            if llm_reason:
                return llm_reason
        best = top[0]
        return (
            f"Selected by hybrid score (semantic + recency); top result {best.doc_id} "
            f"(updated {best.updated_date}) is the most current strong match."
        )

    def _reasoning_llm(self, query: str, top: list[Candidate]) -> Optional[str]:
        summary = "\n".join(
            f"- {c.doc_id} (updated {c.updated_date}, semantic={c.semantic_score:.3f}, "
            f"hybrid={c.hybrid_score:.3f}): {c.title}"
            for c in top
        )
        user = f"Original query: {query}\n\nCandidates (pre-ranked by hybrid score):\n{summary}"
        try:
            raw = self.llm.complete(self.load_prompt(), user, json_mode=True)
            return self.parse_json(raw).get("reasoning")
        except Exception:
            return None
