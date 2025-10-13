"""Simple in-memory rate limiting helpers."""

import asyncio
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Deque, Dict

from fastapi import HTTPException, Request, status


@dataclass
class RateLimitStatus:
    """Details about the applied rate limit for a request."""

    limit: int
    remaining: int
    reset_epoch: float


class RateLimiter:
    """Naive sliding-window rate limiter for FastAPI endpoints."""

    def __init__(self, *, requests: int, window_seconds: int) -> None:
        if requests < 1:
            raise ValueError("requests must be >= 1")
        if window_seconds < 1:
            raise ValueError("window_seconds must be >= 1")
        self._limit = requests
        self._window = window_seconds
        self._records: Dict[str, Deque[float]] = defaultdict(deque)
        self._lock = asyncio.Lock()

    async def __call__(self, request: Request) -> RateLimitStatus:
        now = time.time()
        key = self._make_key(request)

        async with self._lock:
            bucket = self._records[key]
            # Trim entries outside the window
            while bucket and now - bucket[0] > self._window:
                bucket.popleft()

            if len(bucket) >= self._limit:
                retry_after = max(1, int(self._window - (now - bucket[0])))
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Too many requests. Please slow down.",
                    headers={"Retry-After": str(retry_after)},
                )

            bucket.append(now)
            remaining = max(0, self._limit - len(bucket))
            reset = (bucket[0] + self._window) if bucket else (now + self._window)

        return RateLimitStatus(limit=self._limit, remaining=remaining, reset_epoch=reset)

    def _make_key(self, request: Request) -> str:
        session_header = request.headers.get("x-session-id")
        if session_header:
            return f"hdr:{session_header}"

        auth_header = request.headers.get("authorization")
        if auth_header:
            return f"auth:{auth_header}"

        client = request.client
        host = client.host if client else "unknown"
        return f"ip:{host}"


# Shared limiter: 60 requests per minute per key
rate_limiter = RateLimiter(requests=60, window_seconds=60)

__all__ = ["RateLimiter", "RateLimitStatus", "rate_limiter"]
