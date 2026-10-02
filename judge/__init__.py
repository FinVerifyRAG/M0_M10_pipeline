"""
judge/__init__.py
------------------
M8 Judge package — V3 LLM fact-verifier for UNCERTAIN atoms.

Public API
----------
    from judge.schemas import JudgeResult, VERIFIED, NOT_VERIFIED
    from judge.judge_llm import JudgeLLM  # requires openai
"""
# Lightweight schemas — importable without openai
from judge.schemas import JudgeResult, VERIFIED, NOT_VERIFIED

# JudgeLLM requires openai; import explicitly when needed:
#   from judge.judge_llm import JudgeLLM

__all__ = ["JudgeResult", "VERIFIED", "NOT_VERIFIED"]
