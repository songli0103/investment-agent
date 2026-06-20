"""LLM token usage and cost tracking."""
from __future__ import annotations

from dataclasses import dataclass

import structlog

from alphaquant.config import get_settings

log = structlog.get_logger()


# Pricing (USD per 1M tokens) is sourced from Settings so operators can override
# via env vars (MINIMAX_INPUT_PRICE_PER_M / MINIMAX_OUTPUT_PRICE_PER_M).
_settings = get_settings()
INPUT_PRICE_PER_M: float = _settings.minimax_input_price_per_m
OUTPUT_PRICE_PER_M: float = _settings.minimax_output_price_per_m

# Guard so the placeholder warning fires exactly once per process.
_placeholder_warning_emitted = False


def _warn_if_placeholder_pricing() -> None:
    """Emit a single structlog warning if pricing is still at the placeholder defaults.

    Spec §4.4 flagged these as placeholders. We refuse to silently publish wrong
    numbers — operators must explicitly opt out by overriding the env vars.
    """
    global _placeholder_warning_emitted
    if _placeholder_warning_emitted:
        return
    if INPUT_PRICE_PER_M == 3.0 or OUTPUT_PRICE_PER_M == 15.0:
        log.warning(
            "cost_tracker_using_placeholder_pricing",
            input_price_per_m=INPUT_PRICE_PER_M,
            output_price_per_m=OUTPUT_PRICE_PER_M,
            hint="Set MINIMAX_INPUT_PRICE_PER_M and MINIMAX_OUTPUT_PRICE_PER_M to verified values.",
        )
    _placeholder_warning_emitted = True


@dataclass
class TokenUsage:
    input_tokens: int
    output_tokens: int

    @property
    def cost_usd(self) -> float:
        return (
            self.input_tokens * INPUT_PRICE_PER_M / 1_000_000
            + self.output_tokens * OUTPUT_PRICE_PER_M / 1_000_000
        )


def track_usage(
    agent: str,
    input_tokens: int,
    output_tokens: int,
    latency_ms: int,
    request_id: str | None = None,
) -> TokenUsage:
    """Log an LLM call's cost. Returns the TokenUsage object."""
    _warn_if_placeholder_pricing()
    usage = TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens)
    log.info(
        "llm_call",
        agent=agent,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=round(usage.cost_usd, 6),
        latency_ms=latency_ms,
        request_id=request_id,
    )
    return usage