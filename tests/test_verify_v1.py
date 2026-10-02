"""
tests/test_verify_v1.py
-----------------------
Unit tests for verify/v1_deterministic.py — MATCH / MISMATCH / NOT_FOUND / NA.

Run with:
    pytest tests/test_verify_v1.py -v
"""
import pytest
from common.schemas import Atom, Chunk, RetrievalResult
from verify.v1_deterministic import v1_check, MATCH, MISMATCH, NOT_FOUND, NA


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_chunk(text: str, chunk_id: str = "c1", regulator: str = "SEBI") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        text=text,
        regulator=regulator,
        issue_date="01/01/2024",
        source_url="https://sebi.gov.in/test",
    )


def make_rr(*chunks: Chunk) -> RetrievalResult:
    return RetrievalResult(query="test", query_date="2024-01-01", chunks=list(chunks))


def make_atom(text: str, atom_type: str, cited_chunk: str = "c1") -> Atom:
    return Atom(
        atom_id="a1",
        type=atom_type,
        text=text,
        claim=f"The regulation states {text}.",
        cited_chunk=cited_chunk,
    )


# ---------------------------------------------------------------------------
# RATE tests
# ---------------------------------------------------------------------------

class TestV1Rate:
    def test_match_exact_percent(self):
        chunk = make_chunk("The interest rate is 5% per annum.")
        atom = make_atom("5%", "RATE")
        assert v1_check(atom, make_rr(chunk)) == MATCH

    def test_match_per_cent_variant(self):
        chunk = make_chunk("The rate shall be 7.5 per cent.")
        atom = make_atom("7.5%", "RATE")
        assert v1_check(atom, make_rr(chunk)) == MATCH

    def test_mismatch_wrong_value(self):
        chunk = make_chunk("The rate is 5% per annum.")
        atom = make_atom("10%", "RATE")
        assert v1_check(atom, make_rr(chunk)) == MISMATCH

    def test_not_found_no_rate_in_chunk(self):
        chunk = make_chunk("This circular covers disclosure requirements.")
        atom = make_atom("5%", "RATE")
        assert v1_check(atom, make_rr(chunk)) == NOT_FOUND


# ---------------------------------------------------------------------------
# THRESHOLD tests
# ---------------------------------------------------------------------------

class TestV1Threshold:
    def test_match_lakh(self):
        chunk = make_chunk("Minimum net worth of ₹5 lakh is required.")
        atom = make_atom("₹5 lakh", "THRESHOLD")
        assert v1_check(atom, make_rr(chunk)) == MATCH

    def test_match_crore(self):
        chunk = make_chunk("The portfolio value must exceed ₹1 crore.")
        atom = make_atom("₹1 crore", "THRESHOLD")
        assert v1_check(atom, make_rr(chunk)) == MATCH

    def test_mismatch_different_amount(self):
        chunk = make_chunk("Minimum investment of ₹10 lakh.")
        atom = make_atom("₹5 lakh", "THRESHOLD")
        assert v1_check(atom, make_rr(chunk)) == MISMATCH


# ---------------------------------------------------------------------------
# DATE tests
# ---------------------------------------------------------------------------

class TestV1Date:
    def test_match_same_date_different_format(self):
        chunk = make_chunk("Effective from 12 March 2024.")
        atom = make_atom("12/03/2024", "DATE")
        assert v1_check(atom, make_rr(chunk)) == MATCH

    def test_not_found_no_date(self):
        chunk = make_chunk("The regulation applies to all listed entities.")
        atom = make_atom("12/03/2024", "DATE")
        assert v1_check(atom, make_rr(chunk)) == NOT_FOUND


# ---------------------------------------------------------------------------
# SECTION tests
# ---------------------------------------------------------------------------

class TestV1Section:
    def test_match_section(self):
        chunk = make_chunk("As per Regulation 52(4), every issuer shall...")
        atom = make_atom("Regulation 52(4)", "SECTION")
        assert v1_check(atom, make_rr(chunk)) == MATCH

    def test_not_found_different_section(self):
        chunk = make_chunk("As per Regulation 48, the board shall...")
        atom = make_atom("Regulation 52(4)", "SECTION")
        # Regulation 48 ≠ 52, so either MISMATCH or NOT_FOUND depending on normalizer
        result = v1_check(atom, make_rr(chunk))
        assert result in (MISMATCH, NOT_FOUND)


# ---------------------------------------------------------------------------
# NA atoms (ENTITY / APPLICABILITY)
# ---------------------------------------------------------------------------

class TestV1NA:
    def test_entity_returns_na(self):
        chunk = make_chunk("SEBI regulates the securities market.")
        atom = make_atom("SEBI", "ENTITY")
        assert v1_check(atom, make_rr(chunk)) == NA

    def test_applicability_returns_na(self):
        chunk = make_chunk("This applies to all listed companies.")
        atom = make_atom("listed companies", "APPLICABILITY")
        assert v1_check(atom, make_rr(chunk)) == NA


# ---------------------------------------------------------------------------
# Multi-chunk: cited chunk first, fallback to others
# ---------------------------------------------------------------------------

class TestV1MultiChunk:
    def test_match_in_non_cited_chunk(self):
        cited = make_chunk("This chunk has no rate information.", chunk_id="c1")
        other = make_chunk("The penalty rate is 12%.", chunk_id="c2")
        atom = make_atom("12%", "RATE", cited_chunk="c1")
        assert v1_check(atom, make_rr(cited, other)) == MATCH

    def test_match_in_cited_chunk_takes_priority(self):
        cited = make_chunk("The deposit rate is 6%.", chunk_id="c1")
        other = make_chunk("The penalty rate is 12%.", chunk_id="c2")
        atom = make_atom("6%", "RATE", cited_chunk="c1")
        assert v1_check(atom, make_rr(cited, other)) == MATCH
