"""
tests/test_verify_cascade.py
-----------------------------
Integration tests for the full V1 → V2 cascade (verify/cascade.py).

V2 NLI is run in skip_v2=True mode here so tests pass without a GPU/model.
A separate integration test (tests/test_verify_cascade_nli.py) exercises the
real NLI model and is marked @pytest.mark.slow.

Run with:
    pytest tests/test_verify_cascade.py -v
"""
import pytest
from common.schemas import Atom, Chunk, RetrievalResult, VerifiedAtom
from verify.cascade import verify
from verify.v1_deterministic import MATCH, MISMATCH, NOT_FOUND, NA


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_chunk(text: str, chunk_id: str = "c1") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        text=text,
        regulator="SEBI",
        issue_date="01/01/2024",
        source_url="https://sebi.gov.in/test",
    )


def make_rr(*chunks) -> RetrievalResult:
    return RetrievalResult(query="q", query_date="2024-01-01", chunks=list(chunks))


def make_atom(text: str, atom_type: str, claim: str = "", cited: str = "c1") -> Atom:
    return Atom(
        atom_id=f"a_{text[:5]}",
        type=atom_type,
        text=text,
        claim=claim or f"The regulation states {text}.",
        cited_chunk=cited,
    )


# ---------------------------------------------------------------------------
# Tests — skip_v2=True (no model required)
# ---------------------------------------------------------------------------

class TestCascadeV1Only:
    """All atoms resolved by V1; V2 skipped."""

    def test_match_rate_atom(self):
        chunk = make_chunk("The rate is 5% per annum.")
        atoms = [make_atom("5%", "RATE")]
        results = verify(atoms, make_rr(chunk), skip_v2=True)

        assert len(results) == 1
        va = results[0]
        assert isinstance(va, VerifiedAtom)
        assert va.v1_status == MATCH
        assert va.v2_entail_prob is None   # V1 definitive → no V2

    def test_mismatch_rate_atom(self):
        chunk = make_chunk("The rate is 5% per annum.")
        atoms = [make_atom("10%", "RATE")]
        results = verify(atoms, make_rr(chunk), skip_v2=True)
        assert results[0].v1_status == MISMATCH

    def test_entity_atom_goes_to_v2_but_skipped(self):
        chunk = make_chunk("SEBI regulates listed companies.")
        atoms = [make_atom("SEBI", "ENTITY")]
        results = verify(atoms, make_rr(chunk), skip_v2=True)

        va = results[0]
        assert va.v1_status == NA
        # V2 was *intended* but skipped; entail_prob should be None
        assert va.v2_entail_prob is None
        assert va.metadata.get("v2_ran") is False

    def test_no_atoms_returns_empty(self):
        rr = make_rr(make_chunk("Some text"))
        assert verify([], rr, skip_v2=True) == []

    def test_all_atom_types_covered(self):
        """Every atom type must produce a VerifiedAtom — no silent drops."""
        chunk = make_chunk("Rate 5%, threshold ₹5 lakh, date 12/03/2024, section 52(4).")
        atoms = [
            make_atom("5%", "RATE"),
            make_atom("₹5 lakh", "THRESHOLD"),
            make_atom("12/03/2024", "DATE"),
            make_atom("Regulation 52(4)", "SECTION"),
            make_atom("SEBI", "ENTITY"),
            make_atom("all listed companies", "APPLICABILITY"),
        ]
        results = verify(atoms, make_rr(chunk), skip_v2=True)
        assert len(results) == len(atoms)
        for va in results:
            assert isinstance(va, VerifiedAtom)
            assert va.v1_status in {MATCH, MISMATCH, NOT_FOUND, NA}

    def test_order_preserved(self):
        """Output order must match input order."""
        chunk = make_chunk("Rate 5%, amount ₹10 lakh.")
        atoms = [
            make_atom("5%", "RATE", cited="c1"),
            make_atom("₹10 lakh", "THRESHOLD", cited="c1"),
        ]
        results = verify(atoms, make_rr(chunk), skip_v2=True)
        assert results[0].atom.text == "5%"
        assert results[1].atom.text == "₹10 lakh"

    def test_not_found_rate_no_rate_in_chunk(self):
        chunk = make_chunk("This section covers eligibility criteria.")
        atoms = [make_atom("15%", "RATE")]
        results = verify(atoms, make_rr(chunk), skip_v2=True)
        assert results[0].v1_status == NOT_FOUND

    def test_metadata_v2_ran_flag(self):
        chunk = make_chunk("Rate is 5%.")
        atoms = [make_atom("5%", "RATE")]
        results = verify(atoms, make_rr(chunk), skip_v2=True)
        # MATCH → V2 should NOT have been run
        assert results[0].metadata.get("v2_ran") is False
