"""LLM token usage and cost tracking."""
from __future__ import annotations

from dataclasses import dataclass

import structlog

log = structlog.get_logger()


# Placeholder pricing — VERIFY with actual MiniMax pricing
INPUT_PRICE_PER_M = 3.0   # $3 per 1M input tokens
OUTPUT_PRICE_PER_M = 15.0  # $15 per 1M output tokens


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