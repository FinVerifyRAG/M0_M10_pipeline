"""
decision/__init__.py
---------------------
M8 Decision package: router, combine, and answer rewrite.

Public API
----------
    from decision.router import route, RouteResult, SUPPORTED, UNCERTAIN, ABSTAINED
    from decision.combine import combine, decide, AnswerDecision, AtomDecision
    from decision.answer_rewrite import rewrite_answer, RewrittenAnswer
"""
from decision.router import route, RouteResult, SUPPORTED, UNCERTAIN, ABSTAINED
from decision.combine import (
    combine, decide, AnswerDecision, AtomDecision,
    FULLY_SUPPORTED, PARTIALLY_VERIFIED, ANSWER_ABSTAINED,
)
from decision.answer_rewrite import rewrite_answer, RewrittenAnswer

__all__ = [
    # router
    "route", "RouteResult", "SUPPORTED", "UNCERTAIN", "ABSTAINED",
    # combine
    "combine", "decide", "AnswerDecision", "AtomDecision",
    "FULLY_SUPPORTED", "PARTIALLY_VERIFIED", "ANSWER_ABSTAINED",
    # answer_rewrite
    "rewrite_answer", "RewrittenAnswer",
]
