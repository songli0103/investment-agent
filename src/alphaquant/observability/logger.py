"""通过 structlog 进行结构化日志记录。"""
from __future__ import annotations

import logging
import sys

import structlog

from alphaquant.infrastructure.config import get_settings


def configure_logging() -> None:
    """配置 structlog 以输出 JSON 格式。启动时调用一次。"""
    settings = get_settings()
    level = getattr(logging, settings.log_level.upper(), logging.INFO)

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """获取一个 logger 实例。"""
    return structlog.get_logger(name) if name else structlog.get_logger()