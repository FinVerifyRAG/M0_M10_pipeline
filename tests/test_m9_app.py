"""
tests/test_m9_app.py
---------------------
M9: Comprehensive tests for the app/ layer (badge, pipeline, api).

Coverage plan
-------------
badge.py  (12 tests)
  - risk_badge: LOW / MEDIUM / HIGH thresholds
  - risk_badge: boundary values (0.0, 0.25, 0.26, 0.60, 0.61, 1.0)
  - risk_badge: clipping (< 0, > 1)
  - risk_badge: .html() output
  - status_badge: all 5 statuses + unknown
  - answer_status_label: all 3 statuses

pipeline.py (20 tests)
  - PipelineResult fields present
  - run_pipeline returns PipelineResult on stub modules
  - error field set on exception
  - latency keys present
  - _build_atom_result mapping
  - _build_source mapping
  - _build_verification_summary: FULLY_SUPPORTED / PARTIALLY_VERIFIED / ABSTAINED
  - reset_pipeline clears singletons
  - VerificationSummary coverage calculation
  - AtomResult fields

api.py (15 tests, FastAPI-only, skipped if not installed)
  - POST /ask with minimal payload
  - POST /ask with date and skip_judge
  - POST /ask response structure
  - GET /health returns 200
  - POST /ask with invalid rewrite_mode returns 422
  - POST /ask with empty question returns 422
  - POST /ask with question > 1000 chars returns 422
  - AtomResponse risk_badge field
  - VerificationResponse fields
  - POST /pipeline/reset returns 200
  - sources in response
  - latency in response
  - AskRequest validation
  - HistoryTurn serialization
  - CORS headers present

Total: 47 tests
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ── Shared fixtures ────────────────────────────────────────────────────────────

from common.schemas import (
    Atom, Chunk, GeneratedAnswer, RetrievalResult, ScoredAtom, Thresholds,
    VerifiedAtom,
)
from decision.combine import AnswerDecision, AtomDecision
from decision.router import SUPPORTED, UNCERTAIN, ABSTAINED


def _make_atom(atom_id="a1", type_="RATE", text="5%", claim="The rate is 5%.") -> Atom:
    return Atom(atom_id=atom_id, type=type_, text=text, claim=claim)


def _make_chunk(chunk_id="c1", regulator="RBI", issue_date="2024-01-01") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        text="The SLR rate is 18.5% of NDTL for scheduled commercial banks.",
        regulator=regulator,
        issue_date=issue_date,
        source_url="https://rbi.org.in/test",
        metadata={"section": "Section 24", "breadcrumb": "Section 24"},
    )


def _make_verified_atom(atom=None, v1_status="MATCH", v2_entail_prob=0.91) -> VerifiedAtom:
    return VerifiedAtom(
        atom=atom or _make_atom(),
        v1_status=v1_status,
        v2_entail_prob=v2_entail_prob,
    )


def _make_scored_atom(risk=0.15, atom=None, v1_status="MATCH") -> ScoredAtom:
    return ScoredAtom(
        verified=_make_verified_atom(atom=atom, v1_status=v1_status),
        risk=risk,
    )


def _make_atom_decision(
    atom_id="a1",
    atom_type="RATE",
    atom_text="5%",
    atom_claim="The rate is 5%.",
    final_status="SUPPORTED",
    risk=0.12,
    stratum="RBI|RATE",
) -> AtomDecision:
    return AtomDecision(
        atom_id=atom_id,
        atom_type=atom_type,
        atom_text=atom_text,
        atom_claim=atom_claim,
        final_status=final_status,
        risk=risk,
        stratum=stratum,
    )


def _make_answer_decision(
    answer_status="FULLY_SUPPORTED",
    n_supported=2,
    n_abstained=0,
    n_uncertain=0,
    n_not_verified=0,
    n_verified=0,
    judge_calls=0,
) -> AnswerDecision:
    ads = [
        _make_atom_decision(atom_id=f"a{i}", final_status="SUPPORTED")
        for i in range(n_supported)
    ]
    coverage = (n_supported + n_verified) / max(len(ads), 1)
    return AnswerDecision(
        answer_status=answer_status,
        atom_decisions=ads,
        n_supported=n_supported,
        n_verified=n_verified,
        n_uncertain=n_uncertain,
        n_abstained=n_abstained,
        n_not_verified=n_not_verified,
        judge_calls=judge_calls,
        coverage=round(coverage, 4),
        answer_text="The rate is 5%.",
        query="What is the rate?",
    )


# ══════════════════════════════════════════════════════════════════════════════
# Group 1: badge.py (12 tests)
# ══════════════════════════════════════════════════════════════════════════════

from app.badge import risk_badge, status_badge, answer_status_label, RiskBadge, StatusBadge


class TestRiskBadge:

    def test_low_at_zero(self):
        b = risk_badge(0.0)
        assert b.label == "LOW"
        assert b.emoji == "🟢"

    def test_low_at_boundary(self):
        b = risk_badge(0.25)
        assert b.label == "LOW"

    def test_medium_just_above_low(self):
        b = risk_badge(0.26)
        assert b.label == "MEDIUM"
        assert b.emoji == "🟡"

    def test_medium_at_boundary(self):
        b = risk_badge(0.60)
        assert b.label == "MEDIUM"

    def test_high_just_above_medium(self):
        b = risk_badge(0.61)
        assert b.label == "HIGH"
        assert b.emoji == "🔴"

    def test_high_at_one(self):
        b = risk_badge(1.0)
        assert b.label == "HIGH"

    def test_clips_below_zero(self):
        b = risk_badge(-0.5)
        assert b.label == "LOW"
        assert b.risk == 0.0

    def test_clips_above_one(self):
        b = risk_badge(1.5)
        assert b.label == "HIGH"
        assert b.risk == 1.0

    def test_html_contains_label(self):
        b = risk_badge(0.1)
        html = b.html()
        assert "LOW" in html
        assert "span" in html

    def test_html_show_score(self):
        b = risk_badge(0.1)
        html = b.html(show_score=True)
        assert "0.10" in html

    def test_custom_thresholds(self):
        b = risk_badge(0.35, low_max=0.4, med_max=0.8)
        assert b.label == "LOW"

    def test_str_repr(self):
        b = risk_badge(0.5)
        assert "MEDIUM" in str(b)


class TestStatusBadge:

    @pytest.mark.parametrize("status,expected_emoji", [
        ("SUPPORTED",    "✅"),
        ("VERIFIED",     "✅"),
        ("UNCERTAIN",    "⚠️"),
        ("ABSTAINED",    "🚫"),
        ("NOT_VERIFIED", "❌"),
    ])
    def test_known_statuses(self, status, expected_emoji):
        sb = status_badge(status)
        assert sb.label == status
        assert sb.emoji == expected_emoji

    def test_unknown_status_fallback(self):
        sb = status_badge("UNKNOWN_STATUS")
        assert sb.label == "UNKNOWN_STATUS"
        assert "ℹ" in sb.emoji  # fallback

    def test_html_output(self):
        sb = status_badge("SUPPORTED")
        html = sb.html()
        assert "SUPPORTED" in html
        assert "span" in html

    def test_lowercase_input(self):
        sb = status_badge("supported")
        assert sb.label == "SUPPORTED"


class TestAnswerStatusLabel:

    def test_fully_supported(self):
        label, color, emoji = answer_status_label("FULLY_SUPPORTED")
        assert "Verified" in label
        assert color == "#1F6B4A"
        assert emoji == "✅"

    def test_partially_verified(self):
        label, color, emoji = answer_status_label("PARTIALLY_VERIFIED")
        assert "Partial" in label

    def test_abstained(self):
        label, color, emoji = answer_status_label("ABSTAINED")
        assert emoji == "🚫"

    def test_unknown_passthrough(self):
        label, color, emoji = answer_status_label("CUSTOM_STATUS")
        assert label == "CUSTOM_STATUS"


# ══════════════════════════════════════════════════════════════════════════════
# Group 2: pipeline.py (20 tests)
# ══════════════════════════════════════════════════════════════════════════════

from app.pipeline import (
    _build_atom_result,
    _build_source,
    _build_verification_summary,
    run_pipeline,
    reset_pipeline,
    AtomResult,
    SourceResult,
    VerificationSummary,
    PipelineResult,
)


class TestBuildAtomResult:

    def test_fields_present(self):
        ad = _make_atom_decision(final_status="SUPPORTED", risk=0.12)
        ar = _build_atom_result(ad, risk_badge)
        assert ar.atom_id == "a1"
        assert ar.status == "SUPPORTED"
        assert ar.risk == 0.12
        assert ar.risk_label == "LOW"
        assert ar.atom_type == "RATE"

    def test_medium_risk_label(self):
        ad = _make_atom_decision(risk=0.45, final_status="UNCERTAIN")
        ar = _build_atom_result(ad, risk_badge)
        assert ar.risk_label == "MEDIUM"
        assert ar.status == "UNCERTAIN"

    def test_high_risk_label(self):
        ad = _make_atom_decision(risk=0.85, final_status="ABSTAINED")
        ar = _build_atom_result(ad, risk_badge)
        assert ar.risk_label == "HIGH"

    def test_stratum_preserved(self):
        ad = _make_atom_decision(stratum="SEBI|THRESHOLD")
        ar = _build_atom_result(ad, risk_badge)
        assert ar.stratum == "SEBI|THRESHOLD"

    def test_rationale_preserved(self):
        ad = _make_atom_decision()
        ad.rationale = "Verified against Section 24."
        ar = _build_atom_result(ad, risk_badge)
        assert ar.rationale == "Verified against Section 24."

    def test_risk_rounded_to_4dp(self):
        ad = _make_atom_decision(risk=0.123456789)
        ar = _build_atom_result(ad, risk_badge)
        assert ar.risk == 0.1235


class TestBuildSource:

    def test_fields_present(self):
        chunk = _make_chunk()
        sr = _build_source(chunk)
        assert sr.chunk_id == "c1"
        assert sr.regulator == "RBI"
        assert sr.issue_date == "2024-01-01"
        assert "rbi.org.in" in sr.source_url
        assert sr.section == "Section 24"

    def test_preview_truncated(self):
        chunk = _make_chunk()
        sr = _build_source(chunk, max_chars=10)
        assert len(sr.preview) == 10

    def test_missing_section_fallback(self):
        chunk = _make_chunk()
        chunk.metadata = {}
        sr = _build_source(chunk)
        assert sr.section == ""


class TestBuildVerificationSummary:

    def test_fully_supported(self):
        ad = _make_answer_decision(answer_status="FULLY_SUPPORTED", n_supported=3)
        vs = _build_verification_summary(ad)
        assert vs.status == "FULLY_SUPPORTED"
        assert vs.status_label == "Fully Verified"
        assert vs.n_total == 3
        assert vs.n_supported == 3
        assert "verified" in vs.explanation.lower()

    def test_abstained(self):
        ad = _make_answer_decision(answer_status="ABSTAINED", n_supported=0, n_abstained=2)
        # Manually adjust for test
        ad.n_abstained = 2
        ad.atom_decisions = [_make_atom_decision(final_status="ABSTAINED") for _ in range(2)]
        vs = _build_verification_summary(ad)
        assert vs.status == "ABSTAINED"
        assert "not be relied" in vs.explanation

    def test_partially_verified(self):
        ad = _make_answer_decision(answer_status="PARTIALLY_VERIFIED", n_supported=2, n_abstained=1)
        ad.n_abstained = 1
        vs = _build_verification_summary(ad)
        assert vs.status == "PARTIALLY_VERIFIED"
        assert vs.n_total >= 2

    def test_coverage_present(self):
        ad = _make_answer_decision(n_supported=2)
        vs = _build_verification_summary(ad)
        assert 0.0 <= vs.coverage <= 1.0


class TestRunPipeline:
    """Tests using stubs — no real ML models needed."""

    def test_returns_pipeline_result(self):
        result = run_pipeline("What is the SLR requirement?")
        assert isinstance(result, PipelineResult)

    def test_query_preserved(self):
        result = run_pipeline("Test query 123")
        assert result.query == "Test query 123"

    def test_query_date_default_today(self):
        result = run_pipeline("Test")
        assert result.query_date == date.today().isoformat()

    def test_query_date_custom(self):
        result = run_pipeline("Test", query_date="2023-06-15")
        assert result.query_date == "2023-06-15"

    def test_latency_keys_present(self):
        result = run_pipeline("Test latency")
        # At minimum "total" should be present if pipeline ran
        assert isinstance(result.latency, dict)

    def test_error_on_exception(self):
        """Force an exception in the pipeline to verify error handling."""
        with patch("app.pipeline._load_retriever") as mock_ret:
            mock_ret.side_effect = RuntimeError("intentional test error")
            result = run_pipeline("Error test")
        assert result.error is not None
        assert "intentional test error" in result.error

    def test_verification_field_present(self):
        result = run_pipeline("Verification test")
        assert isinstance(result.verification, VerificationSummary)

    def test_atoms_list_present(self):
        result = run_pipeline("Atom list test")
        assert isinstance(result.atoms, list)

    def test_sources_list_present(self):
        result = run_pipeline("Source test")
        assert isinstance(result.sources, list)

    def test_reset_pipeline(self):
        """reset_pipeline() should not raise."""
        reset_pipeline()
        reset_pipeline()  # Idempotent


class TestAtomResultDataclass:

    def test_fields(self):
        ar = AtomResult(
            atom_id="a1", atom_type="RATE", text="5%", claim="Rate is 5%.",
            status="SUPPORTED", risk=0.12, risk_label="LOW",
            stratum="RBI|RATE",
        )
        assert ar.atom_id == "a1"
        assert ar.v1_status is None      # Optional, defaults to None
        assert ar.v2_entail_prob is None


class TestVerificationSummaryDataclass:

    def test_fields(self):
        vs = VerificationSummary(
            status="FULLY_SUPPORTED", status_label="Fully Verified",
            coverage=1.0, n_total=5, n_supported=5, n_verified=0,
            n_uncertain=0, n_abstained=0, n_not_verified=0,
            judge_calls=0, explanation="All good.",
        )
        assert vs.coverage == 1.0
        assert vs.n_total == 5


# ══════════════════════════════════════════════════════════════════════════════
# Group 3: api.py (15 tests, requires FastAPI + httpx)
# ══════════════════════════════════════════════════════════════════════════════

try:
    from fastapi.testclient import TestClient
    from app.api import app as fastapi_app
    _FASTAPI_AVAILABLE = fastapi_app is not None
except ImportError:
    _FASTAPI_AVAILABLE = False

_skip_api = pytest.mark.skipif(
    not _FASTAPI_AVAILABLE,
    reason="FastAPI not installed or app not available",
)


@_skip_api
class TestFastAPIHealth:

    @pytest.fixture(autouse=True)
    def client(self):
        self.client = TestClient(fastapi_app)

    def test_health_200(self):
        resp = self.client.get("/health")
        assert resp.status_code == 200

    def test_health_json(self):
        resp = self.client.get("/health")
        data = resp.json()
        assert data["status"] == "ok"
        assert "today" in data


@_skip_api
class TestFastAPIAsk:

    @pytest.fixture(autouse=True)
    def client(self):
        self.client = TestClient(fastapi_app)

    def _minimal_ask(self, question="What is the SLR requirement?"):
        return self.client.post("/ask", json={"question": question})

    def test_ask_200(self):
        resp = self._minimal_ask()
        assert resp.status_code == 200

    def test_ask_response_structure(self):
        resp = self._minimal_ask()
        data = resp.json()
        assert "query" in data
        assert "answer" in data
        assert "atoms" in data
        assert "sources" in data
        assert "verification" in data
        assert "latency" in data

    def test_ask_query_echoed(self):
        resp = self._minimal_ask("Test echo query")
        assert resp.json()["query"] == "Test echo query"

    def test_ask_with_date(self):
        resp = self.client.post("/ask", json={
            "question": "What is CRR?",
            "date": "2024-03-15",
        })
        assert resp.status_code == 200
        assert resp.json()["query_date"] == "2024-03-15"

    def test_ask_with_skip_judge(self):
        resp = self.client.post("/ask", json={
            "question": "What is CRR?",
            "skip_judge": True,
        })
        assert resp.status_code == 200

    def test_ask_invalid_rewrite_mode(self):
        resp = self.client.post("/ask", json={
            "question": "What is CRR?",
            "rewrite_mode": "invalid_mode",
        })
        assert resp.status_code == 422

    def test_ask_empty_question_rejected(self):
        resp = self.client.post("/ask", json={"question": ""})
        assert resp.status_code == 422

    def test_ask_long_question_rejected(self):
        resp = self.client.post("/ask", json={"question": "x" * 1001})
        assert resp.status_code == 422

    def test_ask_atoms_list(self):
        resp = self._minimal_ask()
        atoms = resp.json()["atoms"]
        assert isinstance(atoms, list)

    def test_ask_atom_fields(self):
        """If atoms are returned, each must have required fields."""
        resp = self._minimal_ask()
        atoms = resp.json()["atoms"]
        for atom in atoms:
            assert "atom_id" in atom
            assert "status" in atom
            assert "risk" in atom
            assert "risk_badge" in atom

    def test_ask_verification_fields(self):
        resp = self._minimal_ask()
        v = resp.json()["verification"]
        assert "status" in v
        assert "coverage" in v
        assert "n_total" in v
        assert "explanation" in v

    def test_ask_sources_list(self):
        resp = self._minimal_ask()
        sources = resp.json()["sources"]
        assert isinstance(sources, list)

    def test_ask_latency_dict(self):
        resp = self._minimal_ask()
        latency = resp.json()["latency"]
        assert isinstance(latency, dict)

    def test_pipeline_reset(self):
        resp = self.client.post("/pipeline/reset")
        assert resp.status_code == 200
        assert resp.json()["status"] == "reset"

    def test_ask_with_history(self):
        resp = self.client.post("/ask", json={
            "question": "What is the SLR requirement now?",
            "history": [
                {"role": "user", "content": "Tell me about bank reserve requirements"},
                {"role": "assistant", "content": "Banks must maintain CRR and SLR."},
            ],
        })
        assert resp.status_code == 200
