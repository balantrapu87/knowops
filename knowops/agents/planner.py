"""Planner agent — decomposes a question into a structured retrieval plan."""

from __future__ import annotations

import re

from knowops.agents.base import Agent

_RECENCY_WORDS = {
    "current", "currently", "latest", "now", "today", "recent", "recently",
    "updated", "newest", "change", "changed", "procedure", "policy", "escalation",
    "version", "correct", "recommended", "supported", "deprecated", "still",
    "valid", "new", "present",
}
_JIRA_WORDS = {
    "jira", "bug", "bugs", "ticket", "tickets", "incident", "incidents",
    "assignee", "assigned", "priority", "status", "crash", "error", "timeout",
}
_CONFLUENCE_WORDS = {
    "confluence", "doc", "docs", "documentation", "runbook", "guide", "page",
    "pages", "wiki", "procedure", "policy", "configuration", "config", "faq",
}

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
        if self.offline:
            return self._plan_offline(question)
        raw = self.llm.complete(self.load_prompt(), question, json_mode=True)
        return self._normalise(self.parse_json(raw), question)

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
        words = set(re.findall(r"[a-z0-9]+", question.lower()))

        recency = bool(words & _RECENCY_WORDS)
        time_sensitivity = "high" if recency else "medium"

        jira_hits, conf_hits = len(words & _JIRA_WORDS), len(words & _CONFLUENCE_WORDS)
        if jira_hits > conf_hits:
            source = "jira"
        elif conf_hits > jira_hits:
            source = "confluence"
        else:
            source = "both"

        if words & {"bug", "bugs", "crash", "error", "timeout", "incident"}:
            intent = "bug_lookup"
        elif words & {"status", "state", "progress", "assigned"}:
            intent = "status_check"
        elif words & {"doc", "docs", "guide", "procedure", "policy", "configuration", "config", "how"}:
            intent = "doc_lookup"
        else:
            intent = "general_qa"

        return {
            "intent": intent,
            "sub_queries": [question],
            "time_sensitivity": time_sensitivity,
            "source_preference": source,
            "recency_required": recency,
        }
