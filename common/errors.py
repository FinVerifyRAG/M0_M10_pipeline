"""Failures that paper runs must not swallow."""
from __future__ import annotations


class StrictFailure(RuntimeError):
    """Raised when a strict paper run would otherwise invent a number."""
