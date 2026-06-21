"""Unified LLM client configuration for MiniMax via LiteLLM."""
from __future__ import annotations

import os
from crewai import LLM

from alphaquant.infrastructure.config import Settings, get_settings


def get_llm(
    temperature: float = 0.2,
    max_tokens: int = 4096,
    settings: Settings | None = None,
) -> LLM:
    """Get a configured CrewAI LLM pointing at MiniMax-M3 via LiteLLM.

    Args:
        temperature: 0.0–1.0. Use 0.0 for deterministic numeric extraction.
        max_tokens: Max output tokens.
        settings: Optional Settings override (for testing).

    Returns:
        LLM instance configured for MiniMax-M3.
    """
    cfg = settings or get_settings()
    os.environ.setdefault("OPENAI_API_KEY", cfg.minimax_api_key)
    os.environ.setdefault("OPENAI_API_BASE", cfg.litellm_api_base)

    return LLM(
        model=cfg.litellm_model,
        temperature=temperature,
        max_tokens=max_tokens,
        api_key=cfg.minimax_api_key,
        base_url=cfg.litellm_api_base,
        timeout=cfg.litellm_timeout,
    )
