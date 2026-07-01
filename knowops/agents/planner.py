"""Planner agent — decomposes a question into a structured retrieval plan."""

from __future__ import annotations

import logging
import re

from knowops.agents.base import Agent

log = logging.getLogger("knowops.planner")

_DEFAULTS = {
    "intent": "general_qa",
    "sub_queries": [],
    "time_sensitivity": "medium",
    "source_preference": "both",
    "recency_required": False,
}


class Planner(Agent):
    prompt_filename = "planner.md"

    def run(self, question: str) -> dict:
        log.info("[Planner] question: %r", question)
        if self.offline:
            plan = self._plan_offline(question)
        else:
            raw = self.llm.complete(self.load_prompt(), question, json_mode=True)
            plan = self._normalise(self.parse_json(raw), question)
        log.info("[Planner] plan → intent=%s  time_sensitivity=%s  source=%s  recency=%s",
                 plan.get("intent"), plan.get("time_sensitivity"),
                 plan.get("source_preference"), plan.get("recency_required"))
        return plan

    # ── live-output normalisation ────────────────────────────────────────────
    def _normalise(self, plan: dict, question: str) -> dict:
        out = {**_DEFAULTS, **{k: plan.get(k) for k in _DEFAULTS if plan.get(k) is not None}}
        if not out["sub_queries"]:
            out["sub_queries"] = [question]
        if out["time_sensitivity"] not in self.settings.profiles:
            out["time_sensitivity"] = self.settings.default_time_sensitivity
        return out

    # ── deterministic offline planner ────────────────────────────────────────
    def _plan_offline(self, question: str) -> dict:
        kw = self.settings.planner_keywords
        words = set(re.findall(r"[a-z0-9]+", question.lower()))

        recency = bool(words & kw.recency_words)
        time_sensitivity = "high" if recency else "medium"

        jira_hits = len(words & kw.jira_words)
        conf_hits = len(words & kw.confluence_words)
        if jira_hits > conf_hits:
            source = "jira"
        elif conf_hits > jira_hits:
            source = "confluence"
        else:
            source = "both"

        intent = "general_qa"
        for intent_label, triggers in kw.intent_words.items():
            if words & triggers:
                intent = intent_label
                break

        return {
            "intent": intent,
            "sub_queries": [question],
            "time_sensitivity": time_sensitivity,
            "source_preference": source,
            "recency_required": recency,
        }
