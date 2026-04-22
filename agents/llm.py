"""Shared OpenAI-compatible LLM client for Voyager (Qwen3-30B).

Phase 1 used ASU's Voyager endpoint (https://openai.rc.asu.edu/v1) with the
`openai` Python SDK against `qwen3-30b-a3b-instruct-2507`. Phase 2 keeps the
same transport so the three LLM-backed agents (Planner, Diagnostician,
Patcher) share one configuration path.

Install: pip install openai
"""

from __future__ import annotations

import logging
import os
import time
from typing import Optional

from openai import OpenAI

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "qwen3-30b-a3b-instruct-2507"
DEFAULT_BASE_URL = "https://openai.rc.asu.edu/v1"


def make_client(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> OpenAI:
    """Construct an OpenAI client pointed at Voyager.

    api_key  → OPENAI_API_KEY env var if None
    base_url → OPENAI_API_BASE env var if None, else DEFAULT_BASE_URL
    """
    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set — cannot reach Voyager. "
            "Put it in .env or export it in your shell."
        )
    url = base_url or os.environ.get("OPENAI_API_BASE", DEFAULT_BASE_URL)
    return OpenAI(api_key=key, base_url=url)


def chat(
    client: OpenAI,
    model: str,
    system: str,
    user: str,
    temperature: float = 0.2,
    max_tokens: int = 4096,
    max_retries: int = 2,
    agent_name: str = "llm",
) -> str:
    """Single chat completion with exponential backoff. Returns the message text."""
    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        if attempt > 0:
            wait = 2 ** attempt
            logger.warning("%s retry %d/%d after %ds",
                           agent_name, attempt, max_retries, wait)
            time.sleep(wait)
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content or ""
        except Exception as exc:
            logger.error("%s call failed (attempt %d): %s",
                         agent_name, attempt, exc)
            last_exc = exc
    raise RuntimeError(
        f"{agent_name} failed after {max_retries + 1} attempts"
    ) from last_exc
