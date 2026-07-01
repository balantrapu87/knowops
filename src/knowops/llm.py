"""
OpenRouter chat client for the LLM agents.

A thin wrapper over the OpenRouter chat-completions API (Claude Sonnet by
default). The agents call ``complete()`` only in live mode; in offline mode each
agent runs deterministic logic instead, so this client is never hit. ``offline``
is true whenever KNOWOPS_OFFLINE is set or no OPENROUTER_API_KEY is configured.
"""

from __future__ import annotations

import json
import logging
from typing import Optional

import httpx

from knowops.config import SETTINGS, Settings

log = logging.getLogger("knowops.llm")


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
        timeout: Optional[float] = None,
    ) -> str:
        """Return the assistant message text for a system+user prompt.

        Raises RuntimeError in offline mode (agents must branch before calling)
        and httpx.HTTPStatusError on a non-2xx OpenRouter response.
        """
        if self.offline:
            raise RuntimeError("LLMClient.complete() called in offline mode")

        to = timeout if timeout is not None else self.settings.openrouter_timeout
        print(f"[LLM DEBUG] Calling LLM complete: model={self.model}, timeout={to}")

        log.info("── LLM call ▶  model=%s  json_mode=%s", self.model, json_mode)
        log.debug("   user_content: %s", user_content[:300])

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

        import signal

        class LLMTimeoutException(Exception):
            pass

        def alarm_handler(signum, frame):
            raise LLMTimeoutException(f"LLM call strictly timed out after {to} seconds")

        has_alarm = hasattr(signal, "alarm")
        if has_alarm:
            signal.signal(signal.SIGALRM, alarm_handler)
            signal.alarm(max(1, int(to)))

        try:
            resp = httpx.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=to,
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
        finally:
            if has_alarm:
                signal.alarm(0)

        # Pretty-print JSON responses; fall back to raw text
        if json_mode:
            try:
                parsed = json.loads(content)
                log.info("   ◀ response [%s]:\n%s", self.model,
                         json.dumps(parsed, indent=2, ensure_ascii=False))
            except json.JSONDecodeError:
                log.info("   ◀ response [%s] (raw): %s", self.model, content[:500])
        else:
            log.info("   ◀ response [%s]: %s…", self.model, content[:200])

        return content
