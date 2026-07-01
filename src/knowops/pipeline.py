"""
Pipeline orchestrator: Planner -> Retriever -> Reranker -> Answering.

Also exposes two retrieval helpers used by the demo and tests:
  * retrieve_baseline — semantic-only ranking (reproduces the original bug)
  * retrieve_fixed    — full date-aware hybrid ranking (the fix)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from knowops.agents import Answering, Planner, Reranker, RetrieverAgent
from knowops.config import SETTINGS, Settings
from knowops.llm import LLMClient
from knowops.search import Candidate, get_backend


@dataclass
class PipelineResult:
    question: str
    plan: dict
    search_params: dict
    candidates: list[Candidate]
    selected: list[Candidate]
    reranker_out: dict
    answer: str
    freshness_warning: bool
    mode: str


class Pipeline:
    def __init__(self, settings: Settings = SETTINGS, offline: Optional[bool] = None):
        self.settings = settings
        self.offline = settings.offline if offline is None else offline

        llm_offline = self.offline or settings.llm_offline
        self.llm = LLMClient(settings, offline=llm_offline)
        self.backend = get_backend(offline=self.offline, settings=settings)

        self.planner = Planner(self.llm, settings)
        self.retriever = RetrieverAgent(self.llm, settings, backend=self.backend)
        self.reranker = Reranker(self.llm, settings)
        self.answering = Answering(self.llm, settings)

    @property
    def mode(self) -> str:
        return "offline" if self.offline else "live"

    # ── full agentic pipeline ────────────────────────────────────────────────
    def run(self, question: str, now: Optional[float] = None) -> PipelineResult:
        plan = self.planner.run(question)
        params, candidates = self.retriever.run(plan, now=now)
        selected, rr = self.reranker.run(question, candidates, now=now)
        answer = self.answering.run(question, selected, rr["freshness_warning"], now=now)
        return PipelineResult(
            question=question,
            plan=plan,
            search_params=params,
            candidates=candidates,
            selected=selected,
            reranker_out=rr,
            answer=answer,
            freshness_warning=rr["freshness_warning"],
            mode=self.mode,
        )

    # ── retrieval-only helpers (demo + tests) ────────────────────────────────
    def retrieve_fixed(
        self, question: str, now: Optional[float] = None
    ) -> tuple[list[Candidate], dict, dict]:
        """Full date-aware path; returns (selected, plan, debug)."""
        plan = self.planner.run(question)
        params, candidates = self.retriever.run(plan, now=now)
        selected, rr = self.reranker.run(question, candidates, now=now)
        return selected, plan, {"params": params, "reranker": rr, "candidates": candidates}

    def retrieve_baseline(
        self,
        question: str,
        top_k: Optional[int] = None,
        source_filter: str = "both",
        now: Optional[float] = None,
    ) -> list[Candidate]:
        """Semantic-only ranking — no Planner, no freshness. Reproduces the bug."""
        k = top_k or self.settings.reranker_top_k
        candidates = self.backend.semantic_search(
            query_text=question,
            top_k=self.settings.retriever_top_k,
            source_filter=source_filter,
            date_filter_days=None,
            now=now,
        )
        candidates.sort(key=lambda c: c.semantic_score, reverse=True)
        return candidates[:k]
