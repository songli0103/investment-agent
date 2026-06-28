"""进程内的令牌桶限流器(规范 §5.4)。

每客户端 IP 每分钟 10 个请求。纯 Python(无 Redis,无 slowapi)。

令牌桶以恒定速率补充至最大容量。每个请求消耗一个令牌;
如果桶为空,则以 HTTP 429 Too Many Requests 拒绝请求。
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass

from fastapi import HTTPException, Request


RATE_LIMIT_PER_MINUTE = 10
RATE_LIMIT_CAPACITY = RATE_LIMIT_PER_MINUTE  # 突发 == 持续速率
REFILL_INTERVAL_SECONDS = 60.0 / RATE_LIMIT_PER_MINUTE  # 每个令牌 6 秒


@dataclass
class _Bucket:
    tokens: float
    last_refill: float


class TokenBucketRateLimiter:
    """每键的令牌桶限流器。线程安全。

    键通常是客户端 IP 地址。存储在进程内;桶字典由锁保护以支持
    并发的 FastAPI 工作线程。
    """

    def __init__(self, capacity: int = RATE_LIMIT_CAPACITY) -> None:
        self._capacity = float(capacity)
        self._lock = threading.Lock()
        self._buckets: dict[str, _Bucket] = {}

    def _refill(self, bucket: _Bucket, now: float) -> None:
        elapsed = now - bucket.last_refill
        if elapsed <= 0:
            return
        # 添加的令牌与经过的时间成正比,以容量为上限。
        bucket.tokens = min(
            self._capacity,
            bucket.tokens + elapsed / REFILL_INTERVAL_SECONDS,
        )
        bucket.last_refill = now

    def consume(self, key: str, tokens: float = 1.0) -> None:
        """为 ``key`` 从桶中消耗 ``tokens``,否则抛出 429。"""
        now = time.monotonic()
        with self._lock:
            bucket = self._buckets.get(key)
            if bucket is None:
                bucket = _Bucket(tokens=self._capacity - tokens, last_refill=now)
                self._buckets[key] = bucket
                return
            self._refill(bucket, now)
            if bucket.tokens >= tokens:
                bucket.tokens -= tokens
                return
            # 桶为空:拒绝。
            raise HTTPException(
                status_code=429,
                detail={
                    "code": "RATE_LIMITED",
                    "message": "超出速率限制:每个客户端每分钟最多 10 个请求",
                },
            )


# 模块级单例 —— 每个进程一个限流器。
_rate_limiter = TokenBucketRateLimiter()


def rate_limit_analyze(request: Request) -> None:
    """FastAPI 依赖项:在 /analyze 上强制按 IP 速率限制。"""
    # 在某些测试工具中 ``request.client`` 可能为 None;回退到 sentinel 键,
    # 以确保限流器仍能工作。
    client = request.client
    key = client.host if client is not None else "unknown"
    _rate_limiter.consume(key)


def reset_rate_limiter() -> None:
    """测试辅助函数:清除所有按 IP 的桶。"""
    with _rate_limiter._lock:
        _rate_limiter._buckets.clear()
