"""从环境变量加载的配置。"""
from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """从 .env 文件和环境加载的应用设置。"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # 必填项
    minimax_api_key: str = Field(..., min_length=1)

    # LLM(带默认值;可在 .env 中覆盖)。``openai/`` 前缀告诉 LiteLLM 使用哪个 provider;
    # 模型名称本身会被转发到 ``LITELLM_API_BASE`` 指定的 OpenAI 兼容网关。
    litellm_model: str = "openai/MiniMax-M2.7-highspeed"
    litellm_api_base: str = "https://token.juda.dev/v1"
    litellm_timeout: int = 60

    # 成本跟踪定价(美元 / 百万 token)——占位符默认值,可通过环境变量覆盖。
    # 环境变量:MINIMAX_INPUT_PRICE_PER_M / MINIMAX_OUTPUT_PRICE_PER_M
    minimax_input_price_per_m: float = 3.0
    minimax_output_price_per_m: float = 15.0

    # 可选数据源
    alpha_vantage_api_key: str | None = None
    finnhub_api_key: str | None = None
    news_api_key: str | None = None

    # 日志
    log_level: str = "INFO"
    env: str = "development"


def get_settings() -> Settings:
    """加载设置。如果 MINIMAX_API_KEY 缺失,则快速失败。"""
    return Settings()  # type: ignore[call-arg]