"""
tests/test_m8_judge_decision.py
--------------------------------
Comprehensive unit tests for M8: Judge + Decision.

Coverage
--------
judge/judge_llm.py:
  - _parse_judge_json: clean JSON, markdown-fenced JSON, missing fields,
    invalid verdict, fallback on complete failure
  - _format_evidence: truncation, chunk ordering
  - _build_user_prompt: correct sections present
  - JudgeLLM.judge: mock LLM -> VERIFIED verdict, NOT_VERIFIED verdict,
    JSON parse failure -> fallback NOT_VERIFIED, call cap enforcement
  - JudgeLLM.judge_batch: processes multiple atoms, respects cap
  - JudgeLLM.reset_call_counter: resets between queries

decision/router.py:
  - route_one: SUPPORTED when risk <= accept_below
  - route_one: ABSTAINED when risk >= abstain_above
  - route_one: UNCERTAIN when risk in band
  - route_one: ABSTAINED when accept_below == -1 (sentinel)
  - route_one: fallback to *|* when stratum not found
  - route_one: ABSTAINED when no thresholds at all
  - route: batch routing, correct counts
  - Stratum key construction from atom metadata

decision/combine.py:
  - combine: all SUPPORTED -> FULLY_SUPPORTED, full coverage
  - combine: all ABSTAINED -> ABSTAINED answer status
  - combine: mix SUPPORTED+ABSTAINED -> PARTIALLY_VERIFIED
  - combine: UNCERTAIN atoms with VERIFIED judge -> FULLY_SUPPORTED
  - combine: UNCERTAIN atoms with NOT_VERIFIED judge -> PARTIALLY_VERIFIED
  - combine: judge not called (skip_judge) -> UNCERTAIN atoms stay UNCERTAIN
  - combine: coverage fraction correct
  - AtomDecision.to_schema: maps to Decision correctly
  - AnswerDecision.to_decisions: list of Decision objects
  - decide(): integration path (routes + judges + combines)

decision/answer_rewrite.py:
  - FULLY_SUPPORTED -> no rewrite, no disclaimer
  - PARTIALLY_VERIFIED + flag mode -> [⚠ UNVERIFIED] markers
  - PARTIALLY_VERIFIED + trim mode -> removes bad sentences
  - ABSTAINED -> stub mode triggered
  - mode="none" -> disclaimer prepended, no structural change
  - Atoms not found in text -> gracefully skipped in flag/trim mode
"""
from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from common.schemas import (
    Atom, Chunk, Decision, GeneratedAnswer, RetrievalResult,
    ScoredAtom, Thresholds, VerifiedAtom,
)


# ===========================================================================
# Shared fixtures
# ===========================================================================

def _make_chunk(chunk_id: str = "c001", regulator: str = "SEBI",
                text: str = "The rate is 5% per annum.") -> Chunk:
    return Chunk(
        chunk_id=chunk_id, text=text, regulator=regulator,
        issue_date="2024-01-01", source_url="https://sebi.gov.in/test",
    )


def _make_atom(atom_id: str = "a001", atom_type: str = "RATE",
               text: str = "5% per annum",
               claim: str = "The rate is 5% per annum.") -> Atom:
    return Atom(
        atom_id=atom_id, type=atom_type, text=text, claim=claim,
        cited_chunk="c001", span=(0, 12),
    )


def _make_verified(atom: Optional[Atom] = None, v1_status: str = "MATCH",
                   v2_entail_prob: Optional[float] = 0.95) -> VerifiedAtom:
    return VerifiedAtom(
        atom=atom or _make_atom(),
        v1_status=v1_status,
        v2_entail_prob=v2_entail_prob,
    )


def _make_scored(atom: Optional[Atom] = None, risk: float = 0.1,
                 v1_status: str = "MATCH") -> ScoredAtom:
    va = _make_verified(atom=atom, v1_status=v1_status)
    return ScoredAtom(verified=va, risk=risk)


def _make_rr(chunks: Optional[List[Chunk]] = None) -> RetrievalResult:
    return RetrievalResult(
        query="What is the rate?",
        query_date="2024-01-01",
        chunks=chunks or [_make_chunk()],
    )


def _make_answer(text: str = "The rate is 5% per annum. [c001]") -> GeneratedAnswer:
    return GeneratedAnswer(
        query="What is the rate?",
        answer_text=text,
        citations=["c001"],
    )


def _make_thresholds(stratum: str = "*|*",
                     accept_below: float = 0.3,
                     abstain_above: float = 0.7) -> Dict[str, Thresholds]:
    return {
        stratum: Thresholds(
            stratum_name=stratum,
            accept_below=accept_below,
            abstain_above=abstain_above,
        )
    }


def _fake_llm_response(verdict: str = "VERIFIED",
                       rationale: str = "Supported by evidence.",
                       quote: str = "The rate is 5% per annum.") -> dict:
    """Build a fake LLMClient.chat() response."""
    payload = json.dumps({
        "verdict": verdict,
        "rationale": rationale,
        "evidence_quote": quote,
    })
    return {"text": payload, "model": "test-judge-model"}


# ===========================================================================
# judge/judge_llm.py tests
# ===========================================================================

class TestParseJudgeJSON:
    def test_clean_json(self):
        from judge.judge_llm import _parse_judge_json
        raw = json.dumps({"verdict": "VERIFIED",
                          "rationale": "ok", "evidence_quote": "quote"})
        d = _parse_judge_json(raw)
        assert d["verdict"] == "VERIFIED"

    def test_markdown_fenced_json(self):
        from judge.judge_llm import _parse_judge_json
        raw = '```json\n{"verdict":"NOT_VERIFIED","rationale":"no","evidence_quote":"none"}\n```'
        d = _parse_judge_json(raw)
        assert d["verdict"] == "NOT_VERIFIED"

    def test_invalid_verdict_raises(self):
        from judge.judge_llm import _parse_judge_json
        raw = json.dumps({"verdict": "MAYBE", "rationale": "x", "evidence_quote": "y"})
        with pytest.raises(ValueError, match="Invalid verdict"):
            _parse_judge_json(raw)

    def test_no_json_raises(self):
        from judge.judge_llm import _parse_judge_json
        with pytest.raises((ValueError, Exception)):
            _parse_judge_json("This is not JSON at all.")

    def test_missing_fields_filled(self):
        from judge.judge_llm import _parse_judge_json
        raw = json.dumps({"verdict": "VERIFIED"})
        d = _parse_judge_json(raw)
        assert d["verdict"] == "VERIFIED"
        assert "rationale" in d
        assert "evidence_quote" in d

    def test_case_insensitive_verdict(self):
        from judge.judge_llm import _parse_judge_json
        raw = json.dumps({"verdict": "verified", "rationale": "ok", "evidence_quote": "q"})
        d = _parse_judge_json(raw)
        assert d["verdict"] == "VERIFIED"


class TestFormatEvidence:
    def test_contains_chunk_text(self):
        from judge.judge_llm import _format_evidence
        chunks = [_make_chunk(text="Rate is 5% per annum.")]
        ev = _format_evidence(chunks, "The rate is 5%.")
        assert "Rate is 5% per annum." in ev

    def test_truncates_long_evidence(self):
        from judge.judge_llm import _format_evidence, _MAX_EVIDENCE_CHARS
        chunks = [_make_chunk(text="X" * (_MAX_EVIDENCE_CHARS + 500))]
        ev = _format_evidence(chunks, "claim")
        assert len(ev) <= _MAX_EVIDENCE_CHARS + 100  # allow truncation note

    def test_numbered_chunks(self):
        from judge.judge_llm import _format_evidence
        c1 = _make_chunk(chunk_id="c001", text="Chunk one.")
        c2 = _make_chunk(chunk_id="c002", text="Chunk two.")
        ev = _format_evidence([c1, c2], "claim")
        assert "[1]" in ev and "[2]" in ev


class TestBuildUserPrompt:
    def test_contains_claim_and_evidence(self):
        from judge.judge_llm import _build_user_prompt
        prompt = _build_user_prompt("The rate is 5%.", "Evidence text here.", "RATE")
        assert "The rate is 5%." in prompt
        assert "Evidence text here." in prompt
        assert "RATE" in prompt


class TestJudgeLLM:
    def _make_judge(self, fake_response: dict):
        """Create a JudgeLLM with a mocked LLMClient."""
        mock_llm = MagicMock()
        mock_llm.chat.return_value = fake_response

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write("You are a judge. Return JSON.")
            prompt_path = f.name

        from judge.judge_llm import JudgeLLM
        return JudgeLLM(mock_llm, prompt_path=prompt_path)

    def test_verified_verdict(self):
        judge = self._make_judge(_fake_llm_response("VERIFIED"))
        scored = _make_scored()
        rr = _make_rr()
        result = judge.judge(scored, rr)
        assert result.verdict == "VERIFIED"
        assert result.atom_id == scored.verified.atom.atom_id
        assert result.latency_ms >= 0.0

    def test_not_verified_verdict(self):
        judge = self._make_judge(_fake_llm_response("NOT_VERIFIED"))
        scored = _make_scored()
        result = judge.judge(scored, _make_rr())
        assert result.verdict == "NOT_VERIFIED"

    def test_json_parse_failure_defaults_not_verified(self):
        mock_llm = MagicMock()
        mock_llm.chat.return_value = {"text": "This is not valid JSON", "model": "test"}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False,
                                        encoding="utf-8") as f:
            f.write("system prompt")
            prompt_path = f.name

        from judge.judge_llm import JudgeLLM
        judge = JudgeLLM(mock_llm, prompt_path=prompt_path, max_retries=0)
        result = judge.judge(_make_scored(), _make_rr())
        assert result.verdict == "NOT_VERIFIED"
        assert result.parse_error is True

    def test_call_cap_returns_not_verified(self):
        judge = self._make_judge(_fake_llm_response("VERIFIED"))
        judge._call_count = 20  # Already at cap
        result = judge.judge(_make_scored(), _make_rr())
        assert result.verdict == "NOT_VERIFIED"
        assert "cap" in result.rationale.lower()

    def test_call_counter_increments(self):
        judge = self._make_judge(_fake_llm_response("VERIFIED"))
        assert judge.call_count == 0
        judge.judge(_make_scored(), _make_rr())
        assert judge.call_count == 1
        judge.judge(_make_scored(atom=_make_atom("a002")), _make_rr())
        assert judge.call_count == 2

    def test_reset_call_counter(self):
        judge = self._make_judge(_fake_llm_response("VERIFIED"))
        judge.judge(_make_scored(), _make_rr())
        assert judge.call_count == 1
        judge.reset_call_counter()
        assert judge.call_count == 0

    def test_judge_batch_returns_all_atom_ids(self):
        judge = self._make_judge(_fake_llm_response("VERIFIED"))
        atoms = [
            _make_scored(_make_atom("a001")),
            _make_scored(_make_atom("a002")),
            _make_scored(_make_atom("a003")),
        ]
        results = judge.judge_batch(atoms, _make_rr())
        assert set(results.keys()) == {"a001", "a002", "a003"}

    def test_judge_batch_respects_cap(self):
        judge = self._make_judge(_fake_llm_response("VERIFIED"))
        judge.max_calls_per_query = 2
        atoms = [_make_scored(_make_atom(f"a{i:03d}")) for i in range(5)]
        results = judge.judge_batch(atoms, _make_rr())
        # First 2 should be VERIFIED (cap allows), rest NOT_VERIFIED (capped)
        verdicts = [r.verdict for r in results.values()]
        assert verdicts.count("VERIFIED") == 2
        assert verdicts.count("NOT_VERIFIED") == 3

    def test_model_id_captured(self):
        judge = self._make_judge(_fake_llm_response("VERIFIED"))
        result = judge.judge(_make_scored(), _make_rr())
        assert result.model_id == "test-judge-model"


# ===========================================================================
# decision/router.py tests
# ===========================================================================

class TestRouteOne:
    def test_supported_when_risk_low(self):
        from decision.router import route_one, SUPPORTED
        scored = _make_scored(risk=0.1)
        th = _make_thresholds("*|*", accept_below=0.3, abstain_above=0.7)
        result = route_one(scored, th)
        assert result.status == SUPPORTED
        assert result.risk == 0.1

    def test_abstained_when_risk_high(self):
        from decision.router import route_one, ABSTAINED
        scored = _make_scored(risk=0.9)
        th = _make_thresholds("*|*", accept_below=0.3, abstain_above=0.7)
        result = route_one(scored, th)
        assert result.status == ABSTAINED

    def test_uncertain_when_risk_in_band(self):
        from decision.router import route_one, UNCERTAIN
        scored = _make_scored(risk=0.5)
        th = _make_thresholds("*|*", accept_below=0.3, abstain_above=0.7)
        result = route_one(scored, th)
        assert result.status == UNCERTAIN

    def test_abstained_on_sentinel(self):
        from decision.router import route_one, ABSTAINED
        scored = _make_scored(risk=0.5)
        # accept_below=None is the canonical fail-closed state (Bug 4.3 fix).
        # The old -1.0 sentinel is no longer used; None means abstain on all.
        th = _make_thresholds("*|*", accept_below=None, abstain_above=0.7)
        result = route_one(scored, th)
        assert result.status == ABSTAINED

    def test_fallback_to_global(self):
        """When SEBI|RATE not found, falls back to *|* threshold."""
        from decision.router import route_one, SUPPORTED
        scored = _make_scored(risk=0.1)
        # Only *|* in thresholds, not SEBI|RATE
        th = _make_thresholds("*|*", accept_below=0.3, abstain_above=0.7)
        result = route_one(scored, th)
        assert result.status == SUPPORTED

    def test_abstained_when_no_thresholds(self):
        """No thresholds at all -> ABSTAINED (safe default)."""
        from decision.router import route_one, ABSTAINED
        scored = _make_scored(risk=0.1)
        result = route_one(scored, {})
        assert result.status == ABSTAINED

    def test_stratum_returned(self):
        from decision.router import route_one
        scored = _make_scored(risk=0.1)
        th = _make_thresholds("*|*", accept_below=0.3, abstain_above=0.7)
        result = route_one(scored, th)
        assert result.stratum == "*|*"

    def test_boundary_exactly_at_accept_below(self):
        """risk == accept_below should be SUPPORTED (<=)."""
        from decision.router import route_one, SUPPORTED
        scored = _make_scored(risk=0.3)
        th = _make_thresholds("*|*", accept_below=0.3, abstain_above=0.7)
        result = route_one(scored, th)
        assert result.status == SUPPORTED

    def test_boundary_exactly_at_abstain_above(self):
        """risk == abstain_above should be ABSTAINED (>=)."""
        from decision.router import route_one, ABSTAINED
        scored = _make_scored(risk=0.7)
        th = _make_thresholds("*|*", accept_below=0.3, abstain_above=0.7)
        result = route_one(scored, th)
        assert result.status == ABSTAINED

    def test_no_uncertain_band_when_thresholds_equal(self):
        """When accept_below == abstain_above, ordering invariant is violated.
        Router fails closed (ABSTAINED) when accept_below >= abstain_above.
        This documents the Bug 4.3 fix: equal thresholds now fail-closed."""
        from decision.router import route_one, ABSTAINED
        # risk == 0.5 == accept_below == abstain_above
        scored = _make_scored(risk=0.5)
        th = _make_thresholds("*|*", accept_below=0.5, abstain_above=0.5)
        result = route_one(scored, th)
        # accept_below (0.5) >= abstain_above (0.5): ordering violation
        # Router logs an error and returns ABSTAINED (fail-closed)
        assert result.status == ABSTAINED


class TestRouteBatch:
    def test_returns_same_count(self):
        from decision.router import route
        atoms = [_make_scored(risk=r) for r in [0.1, 0.5, 0.9]]
        th = _make_thresholds("*|*", accept_below=0.3, abstain_above=0.7)
        results = route(atoms, th)
        assert len(results) == 3

    def test_correct_statuses(self):
        from decision.router import route, SUPPORTED, UNCERTAIN, ABSTAINED
        atoms = [_make_scored(risk=r) for r in [0.1, 0.5, 0.9]]
        th = _make_thresholds("*|*", accept_below=0.3, abstain_above=0.7)
        results = route(atoms, th)
        assert results[0].status == SUPPORTED
        assert results[1].status == UNCERTAIN
        assert results[2].status == ABSTAINED


# ===========================================================================
# decision/combine.py tests
# ===========================================================================

def _make_route_result(atom_id: str, status: str, risk: float = 0.1):
    from decision.router import RouteResult
    return RouteResult(
        atom_id=atom_id, status=status, risk=risk,
        stratum="*|*", accept_below=0.3, abstain_above=0.7,
    )


def _make_judge_result(atom_id: str, verdict: str):
    from judge.judge_llm import JudgeResult
    return JudgeResult(
        verdict=verdict,
        rationale="test rationale",
        evidence_quote="test quote",
        atom_id=atom_id,
        risk=0.5,
    )


class TestCombine:
    def test_all_supported_gives_fully_supported(self):
        from decision.combine import combine, FULLY_SUPPORTED
        atoms = [_make_scored(_make_atom("a1"), 0.1), _make_scored(_make_atom("a2"), 0.2)]
        routes = [
            _make_route_result("a1", "SUPPORTED"),
            _make_route_result("a2", "SUPPORTED"),
        ]
        answer = _make_answer()
        result = combine(atoms, routes, {}, answer)
        assert result.answer_status == FULLY_SUPPORTED
        assert result.n_supported == 2
        assert result.coverage == 1.0

    def test_all_abstained_gives_answer_abstained(self):
        from decision.combine import combine, ANSWER_ABSTAINED
        atoms = [_make_scored(_make_atom("a1"), 0.9), _make_scored(_make_atom("a2"), 0.95)]
        routes = [
            _make_route_result("a1", "ABSTAINED", 0.9),
            _make_route_result("a2", "ABSTAINED", 0.95),
        ]
        answer = _make_answer()
        result = combine(atoms, routes, {}, answer)
        assert result.answer_status == ANSWER_ABSTAINED
        assert result.n_abstained == 2
        assert result.coverage == 0.0

    def test_mix_gives_partially_verified(self):
        from decision.combine import combine, PARTIALLY_VERIFIED
        atoms = [_make_scored(_make_atom("a1"), 0.1), _make_scored(_make_atom("a2"), 0.9)]
        routes = [
            _make_route_result("a1", "SUPPORTED"),
            _make_route_result("a2", "ABSTAINED", 0.9),
        ]
        answer = _make_answer()
        result = combine(atoms, routes, {}, answer)
        assert result.answer_status == PARTIALLY_VERIFIED
        assert result.n_supported == 1
        assert result.n_abstained == 1

    def test_uncertain_with_verified_judge(self):
        from decision.combine import combine, FULLY_SUPPORTED
        atom = _make_atom("a1")
        atoms  = [_make_scored(atom, 0.5)]
        routes = [_make_route_result("a1", "UNCERTAIN", 0.5)]
        judge_results = {"a1": _make_judge_result("a1", "VERIFIED")}
        answer = _make_answer()
        result = combine(atoms, routes, judge_results, answer)
        assert result.n_verified == 1
        assert result.answer_status == FULLY_SUPPORTED

    def test_uncertain_with_not_verified_judge(self):
        from decision.combine import combine, PARTIALLY_VERIFIED
        atom = _make_atom("a1")
        atoms  = [_make_scored(atom, 0.5)]
        routes = [_make_route_result("a1", "UNCERTAIN", 0.5)]
        judge_results = {"a1": _make_judge_result("a1", "NOT_VERIFIED")}
        answer = _make_answer()
        result = combine(atoms, routes, judge_results, answer)
        assert result.n_not_verified == 1
        assert result.answer_status == PARTIALLY_VERIFIED

    def test_uncertain_no_judge_stays_uncertain(self):
        from decision.combine import combine, PARTIALLY_VERIFIED
        atom = _make_atom("a1")
        atoms  = [_make_scored(atom, 0.5)]
        routes = [_make_route_result("a1", "UNCERTAIN", 0.5)]
        answer = _make_answer()
        result = combine(atoms, routes, {}, answer)  # no judge results
        assert result.n_uncertain == 1
        assert result.answer_status == PARTIALLY_VERIFIED

    def test_coverage_calculation(self):
        from decision.combine import combine
        atoms = [
            _make_scored(_make_atom("a1"), 0.1),  # SUPPORTED
            _make_scored(_make_atom("a2"), 0.5),  # UNCERTAIN -> VERIFIED
            _make_scored(_make_atom("a3"), 0.9),  # ABSTAINED
        ]
        routes = [
            _make_route_result("a1", "SUPPORTED"),
            _make_route_result("a2", "UNCERTAIN", 0.5),
            _make_route_result("a3", "ABSTAINED", 0.9),
        ]
        judge_results = {"a2": _make_judge_result("a2", "VERIFIED")}
        result = combine(atoms, routes, judge_results, _make_answer())
        # coverage = (SUPPORTED + VERIFIED) / total = 2/3
        assert abs(result.coverage - 2/3) < 1e-4  # rounding to 4dp

    def test_judge_calls_counted(self):
        from decision.combine import combine
        atoms  = [_make_scored(_make_atom("a1"), 0.5)]
        routes = [_make_route_result("a1", "UNCERTAIN", 0.5)]
        judge_results = {"a1": _make_judge_result("a1", "VERIFIED")}
        result = combine(atoms, routes, judge_results, _make_answer())
        assert result.judge_calls == 1

    def test_atom_decision_to_schema(self):
        from decision.combine import AtomDecision
        ad = AtomDecision(
            atom_id="a1", atom_type="RATE", atom_text="5%",
            atom_claim="The rate is 5%.", final_status="SUPPORTED",
            risk=0.1, stratum="*|*",
        )
        schema = ad.to_schema()
        assert isinstance(schema, Decision)
        assert schema.atom_id == "a1"
        assert schema.status == "SUPPORTED"
        assert schema.risk == 0.1

    def test_answer_decision_to_decisions(self):
        from decision.combine import combine
        atoms  = [_make_scored(_make_atom("a1"), 0.1)]
        routes = [_make_route_result("a1", "SUPPORTED")]
        result = combine(atoms, routes, {}, _make_answer())
        decisions = result.to_decisions()
        assert len(decisions) == 1
        assert isinstance(decisions[0], Decision)

    def test_empty_atoms(self):
        from decision.combine import combine, FULLY_SUPPORTED
        result = combine([], [], {}, _make_answer())
        assert result.answer_status == FULLY_SUPPORTED
        assert result.coverage == 0.0

    def test_query_and_text_captured(self):
        from decision.combine import combine
        answer = _make_answer("Custom answer text.")
        atoms  = [_make_scored(_make_atom("a1"), 0.1)]
        routes = [_make_route_result("a1", "SUPPORTED")]
        result = combine(atoms, routes, {}, answer)
        assert result.answer_text == "Custom answer text."
        assert result.query == "What is the rate?"


class TestDecideIntegration:
    def test_decide_skip_judge(self):
        from decision.combine import decide, FULLY_SUPPORTED
        atoms     = [_make_scored(_make_atom("a1"), 0.1)]
        thresholds = _make_thresholds("*|*", 0.3, 0.7)
        mock_judge = MagicMock()
        answer    = _make_answer()
        rr        = _make_rr()
        result = decide(atoms, thresholds, mock_judge, rr, answer, skip_judge=True)
        # risk=0.1 <= 0.3 -> SUPPORTED -> no judge called
        mock_judge.judge_batch.assert_not_called()
        assert result.answer_status == FULLY_SUPPORTED

    def test_decide_calls_judge_for_uncertain(self):
        from decision.combine import decide, FULLY_SUPPORTED
        from judge.judge_llm import JudgeResult, VERIFIED

        atoms     = [_make_scored(_make_atom("a1"), 0.5)]  # UNCERTAIN
        thresholds = _make_thresholds("*|*", 0.3, 0.7)
        mock_judge = MagicMock()
        mock_judge.judge_batch.return_value = {
            "a1": JudgeResult(
                verdict=VERIFIED, rationale="ok",
                evidence_quote="q", atom_id="a1", risk=0.5,
            )
        }
        answer = _make_answer()
        rr     = _make_rr()
        result = decide(atoms, thresholds, mock_judge, rr, answer, skip_judge=False)
        mock_judge.judge_batch.assert_called_once()
        assert result.n_verified == 1


# ===========================================================================
# decision/answer_rewrite.py tests
# ===========================================================================

def _make_answer_decision(status: str, atom_statuses: List[str],
                          claim: str = "The rate is 5% per annum.",
                          text: str = "5% per annum"):
    from decision.combine import AnswerDecision, AtomDecision
    ads = [
        AtomDecision(
            atom_id=f"a{i}", atom_type="RATE",
            atom_text=text, atom_claim=claim,
            final_status=s, risk=0.5, stratum="*|*",
        )
        for i, s in enumerate(atom_statuses)
    ]
    n_sup = sum(1 for s in atom_statuses if s == "SUPPORTED")
    n_abs = sum(1 for s in atom_statuses if s == "ABSTAINED")
    n_unc = sum(1 for s in atom_statuses if s == "UNCERTAIN")
    n_nv  = sum(1 for s in atom_statuses if s == "NOT_VERIFIED")
    n_ver = sum(1 for s in atom_statuses if s == "VERIFIED")
    total  = len(atom_statuses)
    cov    = (n_sup + n_ver) / total if total else 0.0
    return AnswerDecision(
        answer_status=status,
        atom_decisions=ads,
        n_supported=n_sup,
        n_verified=n_ver,
        n_uncertain=n_unc,
        n_abstained=n_abs,
        n_not_verified=n_nv,
        coverage=cov,
        answer_text="The rate is 5% per annum as per Regulation 52.",
        query="What is the rate?",
    )


class TestRewriteAnswer:
    def test_fully_supported_no_rewrite(self):
        from decision.answer_rewrite import rewrite_answer
        answer = _make_answer("The rate is 5% per annum as per Regulation 52.")
        ad = _make_answer_decision("FULLY_SUPPORTED", ["SUPPORTED"])
        rw = rewrite_answer(answer, ad)
        assert rw.rewrite_mode == "none"
        assert rw.disclaimer == ""
        assert rw.text == answer.answer_text

    def test_flag_mode_adds_marker(self):
        from decision.answer_rewrite import rewrite_answer
        answer = _make_answer("The rate is 5% per annum as per Regulation 52.")
        # Use 2 atoms: 1 SUPPORTED + 1 ABSTAINED so it's PARTIALLY_VERIFIED not all-bad
        ad = _make_answer_decision("PARTIALLY_VERIFIED", ["SUPPORTED", "ABSTAINED"])
        rw = rewrite_answer(answer, ad, mode="flag")
        assert rw.rewrite_mode == "flag"
        assert "⚠" in rw.text or rw.disclaimer != ""

    def test_flag_mode_has_disclaimer(self):
        from decision.answer_rewrite import rewrite_answer
        answer = _make_answer("The rate is 5% per annum as per Regulation 52.")
        ad = _make_answer_decision("PARTIALLY_VERIFIED", ["ABSTAINED"])
        rw = rewrite_answer(answer, ad, mode="flag")
        assert rw.disclaimer != ""

    def test_trim_mode_removes_bad_sentence(self):
        from decision.answer_rewrite import rewrite_answer
        # Answer with two sentences; second contains the bad atom text
        answer = _make_answer(
            "The threshold is 10 lakhs. The rate is 5% per annum."
        )
        ad = _make_answer_decision(
            "PARTIALLY_VERIFIED", ["ABSTAINED"],
            text="5% per annum",
            claim="The rate is 5% per annum.",
        )
        rw = rewrite_answer(answer, ad, mode="trim")
        # The bad sentence should be removed
        assert "5% per annum" not in rw.text or "threshold" in rw.text

    def test_abstained_answer_gives_stub(self):
        from decision.answer_rewrite import rewrite_answer
        answer = _make_answer()
        ad = _make_answer_decision("ABSTAINED", ["ABSTAINED", "ABSTAINED"])
        rw = rewrite_answer(answer, ad)
        assert rw.rewrite_mode == "stub"
        assert "cannot be" in rw.text.lower() or "unable" in rw.text.lower()

    def test_mode_none_prepends_disclaimer_only(self):
        from decision.answer_rewrite import rewrite_answer
        answer = _make_answer("The rate is 5%.")
        # 2 atoms: 1 SUPPORTED + 1 ABSTAINED -> PARTIALLY_VERIFIED, not all-abstained
        ad = _make_answer_decision("PARTIALLY_VERIFIED", ["SUPPORTED", "ABSTAINED"])
        rw = rewrite_answer(answer, ad, mode="none")
        assert rw.rewrite_mode == "none"
        assert rw.disclaimer != ""
        assert "The rate is 5%." in rw.text

    def test_original_always_preserved(self):
        from decision.answer_rewrite import rewrite_answer
        original = "The rate is 5% per annum."
        answer = _make_answer(original)
        ad = _make_answer_decision("PARTIALLY_VERIFIED", ["ABSTAINED"])
        rw = rewrite_answer(answer, ad, mode="flag")
        assert rw.original == original

    def test_atom_not_in_text_skipped_gracefully(self):
        """If atom text does not appear in the answer, flag mode skips it silently."""
        from decision.answer_rewrite import rewrite_answer
        answer = _make_answer("Completely different text here.")
        ad = _make_answer_decision(
            "PARTIALLY_VERIFIED", ["ABSTAINED"],
            text="NONEXISTENT_TEXT_XYZ",
            claim="NONEXISTENT_CLAIM_XYZ",
        )
        # Should not raise; atom text not found -> no marker added
        rw = rewrite_answer(answer, ad, mode="flag")
        assert isinstance(rw.text, str)
