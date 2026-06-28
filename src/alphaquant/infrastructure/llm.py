"""通过 LiteLLM(OpenAI 兼容)统一配置 LLM 客户端。

在运行时通过环境变量配置(参见 ``.env``):
- ``LITELLM_MODEL``  — OpenAI 兼容形式的 provider/model
  (例如 ``openai/MiniMax-M2.7-highspeed``)
- ``LITELLM_API_BASE`` — OpenAI 兼容的 chat-completions 基础 URL
- ``MINIMAX_API_KEY``  — bearer token(为向后兼容保留此名称)
"""
from __future__ import annotations

import os
from crewai import LLM

from alphaquant.infrastructure.config import Settings, get_settings


def get_llm(
    temperature: float = 0.2,
    max_tokens: int = 4096,
    settings: Settings | None = None,
) -> LLM:
    """通过 LiteLLM(OpenAI 兼容)获取配置好的 CrewAI LLM。

    参数:
        temperature:0.0–1.0。对于确定性数字提取,使用 0.0。
        max_tokens:最大输出 token 数。
        settings:可选的 Settings 覆盖(用于测试)。

    返回:
        根据 ``LITELLM_MODEL`` 中指定的模型配置的 LLM 实例。
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
