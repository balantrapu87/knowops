"""Answering agent — writes a grounded, cited answer from the reranked docs."""

from __future__ import annotations

import re
from typing import Optional

from knowops.agents.base import Agent
from knowops.search import Candidate

_FRESHNESS_NOTE = (
    "⚠️ Note: The most recent available document on this topic is over "
    "6 months old. Please verify this information is still current."
)
_NO_ANSWER = (
    "No document in the knowledge base adequately answers this question. "
    "Please refine the query or consult a subject-matter expert."
)


class Answering(Agent):
    prompt_filename = "answering.md"

    def run(
        self,
        question: str,
        docs: list[Candidate],
        freshness_warning: bool = False,
        now: Optional[float] = None,
    ) -> str:
        if not docs:
            return _NO_ANSWER
        if self.offline:
            return self._answer_offline(question, docs, freshness_warning)
        context = self._format_context(docs, freshness_warning)
        return self.llm.complete(self.load_prompt(), f"Question: {question}\n\n{context}")

    def _format_context(self, docs: list[Candidate], freshness_warning: bool) -> str:
        blocks = [f"freshness_warning: {str(freshness_warning).lower()}", "Documents:"]
        for d in docs:
            blocks.append(
                f"[{d.doc_id}] (updated {d.updated_date})\n{d.title}\n{d.content}"
            )
        return "\n\n".join(blocks)

    def _answer_offline(
        self, question: str, docs: list[Candidate], freshness_warning: bool
    ) -> str:
        top = docs[0]
        sentences = re.split(r"(?<=[.!?])\s+", top.content.strip())
        summary = " ".join(sentences[:2]).strip()
        citations = ", ".join(f"[{d.doc_id}]" for d in docs[:3])

        parts = []
        if freshness_warning:
            parts.append(_FRESHNESS_NOTE)
        parts.append(f"{summary} {citations}")
        parts.append(
            f"Most current source: [{top.doc_id}] (updated {top.updated_date})."
        )
        return "\n\n".join(parts)
