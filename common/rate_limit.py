"""Global token-bucket rate limiter."""
from __future__ import annotations

import threading
import time


class TokenBucket:
    """
    Allow `rate_per_second` acquisitions on average, with a burst of `capacity`.
    A non-positive rate disables waiting.
    """

    def __init__(self, rate_per_second: float = 0.0, capacity: float = 1.0):
        self.rate = float(rate_per_second)
        self.capacity = float(max(capacity, 1.0))
        self._tokens = self.capacity
        self._updated = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self, tokens: float = 1.0) -> float:
        """Block until `tokens` are available. Returns seconds waited."""
        if self.rate <= 0:
            return 0.0
        waited = 0.0
        need = float(tokens)
        while True:
            with self._lock:
                now = time.monotonic()
                elapsed = now - self._updated
                self._updated = now
                self._tokens = min(self.capacity, self._tokens + elapsed * self.rate)
                if self._tokens >= need:
                    self._tokens -= need
                    return waited
                missing = need - self._tokens
                delay = missing / self.rate
            time.sleep(delay)
            waited += delay


_BUCKET: TokenBucket | None = None


def get_bucket() -> TokenBucket:
    global _BUCKET
    if _BUCKET is None:
        _BUCKET = TokenBucket(0.0, 1.0)
    return _BUCKET


def configure_bucket(rate_per_second: float, capacity: float = 1.0) -> TokenBucket:
    global _BUCKET
    _BUCKET = TokenBucket(rate_per_second, capacity)
    return _BUCKET
