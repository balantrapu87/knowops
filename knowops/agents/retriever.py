"""Retriever agent — turns a plan into a vector search and attaches scores.

Date awareness lives here: the Planner's ``time_sensitivity`` selects a
FreshnessProfile (from configs/pipeline.yaml), and this agent applies that
profile's recency weighting (and optional soft date window) to every candidate.
The LLM only optimises the query *text*; the config owns the recency behaviour.
"""

from __future__ import annotations

import logging
from typing import Optional

from knowops.agents.base import Agent
from knowops.config import SETTINGS, Settings
from knowops.freshness import freshness_score, score_with_profile
from knowops.llm import LLMClient
from knowops.search import Candidate, get_backend

log = logging.getLogger("knowops.retriever")


class RetrieverAgent(Agent):
    prompt_filename = "retriever.md"

    def __init__(
        self,
        llm: Optional[LLMClient] = None,
        settings: Settings = SETTINGS,
        backend=None,
        offline: Optional[bool] = None,
    ):
        super().__init__(llm, settings)
        self.backend = backend if backend is not None else get_backend(
            offline=offline, settings=settings
        )

    def run(self, plan: dict, now: Optional[float] = None) -> tuple[dict, list[Candidate]]:
        profile = self.settings.get_profile(plan.get("time_sensitivity"))
        params = self._search_params(plan, profile)

        log.info("[Retriever] profile=%s  semantic_w=%.2f  freshness_w=%.2f  decay=%dd",
                 profile.name, profile.semantic_weight, profile.freshness_weight, profile.decay_days)
        log.info("[Retriever] search_query=%r  source=%s  limit=%d  date_filter=%s",
                 params["search_query"], params["source_filter"],
                 params["limit"], params["date_filter_days"])

        candidates = self.backend.semantic_search(
            query_text=params["search_query"],
            top_k=params["limit"],
            source_filter=params["source_filter"],
            date_filter_days=params["date_filter_days"],
            now=now,
        )
        log.info("[Retriever] vector search returned %d candidates", len(candidates))

        candidates = self._apply_relevance_floor(candidates)
        log.info("[Retriever] after relevance floor: %d candidates remain", len(candidates))

        for c in candidates:
            c.freshness_score = freshness_score(c.updated_ts, profile.decay_days, now=now)
            c.hybrid_score = score_with_profile(c.semantic_score, c.updated_ts, profile, now=now)

        candidates.sort(key=lambda c: c.hybrid_score, reverse=True)

        log.info("[Retriever] top-%d hybrid scores:", min(3, len(candidates)))
        for c in candidates[:3]:
            log.info("   %s | semantic=%.3f  freshness=%.3f  hybrid=%.3f | updated=%s",
                     c.doc_id, c.semantic_score, c.freshness_score, c.hybrid_score, c.updated_date)

        return params, candidates

    def _apply_relevance_floor(self, candidates: list[Candidate]) -> list[Candidate]:
        """Drop weakly-relevant candidates before recency reranking.

        Keeps only documents scoring at least ``relevance_floor_ratio`` of the top
        semantic score, so a fresh-but-off-topic document can never be promoted
        above a genuinely relevant one. The top match always survives.
        """
        ratio = self.settings.relevance_floor_ratio
        if not ratio or not candidates:
            return candidates
        top_semantic = max(c.semantic_score for c in candidates)
        floor = ratio * top_semantic
        gated = [c for c in candidates if c.semantic_score >= floor]
        return gated or candidates

    # ── search-parameter construction ────────────────────────────────────────
    def _search_params(self, plan: dict, profile) -> dict:
        sub_queries = plan.get("sub_queries") or []
        base_query = " ".join(sub_queries) if sub_queries else plan.get("question", "")

        search_query = base_query
        if not self.offline:
            search_query = self._optimise_query(plan, base_query)

        return {
            "search_query": search_query or base_query,
            "source_filter": plan.get("source_preference", "both"),
            "freshness_boost": bool(plan.get("recency_required")),
            # Config (the selected profile) owns the date window, not the LLM.
            "date_filter_days": profile.date_filter_days,
            "limit": self.settings.retriever_top_k,
        }

    def _optimise_query(self, plan: dict, fallback: str) -> str:
        import json

        try:
            raw = self.llm.complete(self.load_prompt(), json.dumps(plan), json_mode=True)
            params = self.parse_json(raw)
            return params.get("search_query") or fallback
        except Exception:
            return fallback
