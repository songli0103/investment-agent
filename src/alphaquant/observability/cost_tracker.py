"""LLM token 使用量与成本跟踪。"""
from __future__ import annotations

from dataclasses import dataclass

import structlog

from alphaquant.infrastructure.config import get_settings

log = structlog.get_logger()


# 单价(美元 / 百万 token)从 Settings 中读取,运维人员可通过环境变量
# (MINIMAX_INPUT_PRICE_PER_M / MINIMAX_OUTPUT_PRICE_PER_M) 进行覆盖。
_settings = get_settings()
INPUT_PRICE_PER_M: float = _settings.minimax_input_price_per_m
OUTPUT_PRICE_PER_M: float = _settings.minimax_output_price_per_m

# 守卫变量,确保占位符警告在每个进程中恰好触发一次。
_placeholder_warning_emitted = False


def _warn_if_placeholder_pricing() -> None:
    """如果价格仍为占位符默认值,则发出一次 structlog 警告。

    规范 §4.4 将这些值标记为占位符。我们拒绝静默发布错误的数字——
    运维人员必须显式通过覆盖环境变量来退出该警告。
    """
    global _placeholder_warning_emitted
    if _placeholder_warning_emitted:
        return
    if INPUT_PRICE_PER_M == 3.0 or OUTPUT_PRICE_PER_M == 15.0:
        log.warning(
            "cost_tracker_using_placeholder_pricing",
            input_price_per_m=INPUT_PRICE_PER_M,
            output_price_per_m=OUTPUT_PRICE_PER_M,
            hint="请将 MINIMAX_INPUT_PRICE_PER_M 和 MINIMAX_OUTPUT_PRICE_PER_M 设置为已核实的数值。",
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
    """记录一次 LLM 调用的成本。返回 TokenUsage 对象。"""
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