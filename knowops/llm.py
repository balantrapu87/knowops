"""
OpenRouter chat client for the LLM agents.

A thin wrapper over the OpenRouter chat-completions API (Claude Sonnet by
default). The agents call ``complete()`` only in live mode; in offline mode each
agent runs deterministic logic instead, so this client is never hit. ``offline``
is true whenever KNOWOPS_OFFLINE is set or no OPENROUTER_API_KEY is configured.
"""

from __future__ import annotations

from typing import Optional

import httpx

from knowops.config import SETTINGS, Settings


class LLMClient:
    def __init__(self, settings: Settings = SETTINGS, offline: Optional[bool] = None):
        self.settings = settings
        self.offline = settings.llm_offline if offline is None else offline
        self.model = settings.openrouter_model
        self.base_url = settings.openrouter_base_url.rstrip("/")
        self.api_key = settings.openrouter_api_key

    def complete(
        self,
        system_prompt: str,
        user_content: str,
        json_mode: bool = False,
        temperature: float = 0.2,
        timeout: float = 60.0,
    ) -> str:
        """Return the assistant message text for a system+user prompt.

        Raises RuntimeError in offline mode (agents must branch before calling)
        and httpx.HTTPStatusError on a non-2xx OpenRouter response.
        """
        if self.offline:
            raise RuntimeError("LLMClient.complete() called in offline mode")

        payload: dict = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "temperature": temperature,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        resp = httpx.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
