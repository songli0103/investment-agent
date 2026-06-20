"""In-process token-bucket rate limiter (spec §5.4).

10 requests/minute per client IP. Pure Python (no Redis, no slowapi).

A token bucket refills at a constant rate up to a maximum capacity. Each
request consumes one token; if the bucket is empty, the request is rejected
with HTTP 429 Too Many Requests.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass

from fastapi import HTTPException, Request


RATE_LIMIT_PER_MINUTE = 10
RATE_LIMIT_CAPACITY = RATE_LIMIT_PER_MINUTE  # burst == sustained rate
REFILL_INTERVAL_SECONDS = 60.0 / RATE_LIMIT_PER_MINUTE  # 6 seconds per token


@dataclass
class _Bucket:
    tokens: float
    last_refill: float


class TokenBucketRateLimiter:
    """Per-key token bucket limiter. Thread-safe.

    Keys are typically client IP addresses. Stored in-process; the bucket
    dict is protected by a lock for concurrent FastAPI workers.
    """

    def __init__(self, capacity: int = RATE_LIMIT_CAPACITY) -> None:
        self._capacity = float(capacity)
        self._lock = threading.Lock()
        self._buckets: dict[str, _Bucket] = {}

    def _refill(self, bucket: _Bucket, now: float) -> None:
        elapsed = now - bucket.last_refill
        if elapsed <= 0:
            return
        # Tokens added proportional to elapsed time, capped at capacity.
        bucket.tokens = min(
            self._capacity,
            bucket.tokens + elapsed / REFILL_INTERVAL_SECONDS,
        )
        bucket.last_refill = now

    def consume(self, key: str, tokens: float = 1.0) -> None:
        """Consume ``tokens`` from the bucket for ``key`` or raise 429."""
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
            # Bucket empty: reject.
            raise HTTPException(
                status_code=429,
                detail={
                    "code": "RATE_LIMITED",
                    "message": "Rate limit exceeded: max 10 requests/minute per client",
                },
            )


# Module-level singleton — one limiter per process.
_rate_limiter = TokenBucketRateLimiter()


def rate_limit_analyze(request: Request) -> None:
    """FastAPI dependency: enforce the per-IP rate limit on /analyze."""
    # ``request.client`` may be None in some test harnesses; fall back to a
    # sentinel key so the limiter still functions.
    client = request.client
    key = client.host if client is not None else "unknown"
    _rate_limiter.consume(key)


def reset_rate_limiter() -> None:
    """Test helper: clear all per-IP buckets."""
    with _rate_limiter._lock:
        _rate_limiter._buckets.clear()
