"""Base class shared by the four pipeline agents."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

from knowops.config import SETTINGS, Settings
from knowops.llm import LLMClient

_PROMPTS_DIR = Path(__file__).resolve().parent.parent.parent / "prompts"

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


class Agent:
    """Common plumbing: load the markdown system prompt, call the LLM, parse JSON."""

    prompt_filename: str = ""

    def __init__(self, llm: Optional[LLMClient] = None, settings: Settings = SETTINGS):
        self.settings = settings
        self.llm = llm if llm is not None else LLMClient(settings)

    @property
    def offline(self) -> bool:
        return self.llm.offline

    def load_prompt(self) -> str:
        return (_PROMPTS_DIR / self.prompt_filename).read_text()

    @staticmethod
    def parse_json(text: str) -> dict:
        """Parse JSON from an LLM response, tolerating ```json fences / preamble."""
        text = text.strip()
        fenced = _JSON_FENCE_RE.search(text)
        if fenced:
            text = fenced.group(1).strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start, end = text.find("{"), text.rfind("}")
            if start != -1 and end != -1 and end > start:
                return json.loads(text[start : end + 1])
            raise
