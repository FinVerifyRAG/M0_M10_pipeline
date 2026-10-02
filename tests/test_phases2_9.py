"""
tests/test_phases2_9.py
------------------------
Tests for Phase 2–9 implementations:
  Phase 2: Structured atomization schema + COMPUTED type
  Phase 3: Statistical certification split (agg_val)
  Phase 4: Leakage prevention (disjoint document splitting)
  Phase 5: Version graph & legal drift events
  Phase 6: Retrieval signals (temporal entropy)
  Phase 7: Judge — COMPUTED atom annotation
  Phase 8: Router/rewrite finalization (OUTDATED, REJECTED)
  Phase 9: Benchmark coverage guard (ensure_min_calibration)

Run with:
    pytest tests/test_phases2_9.py -v
"""
from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
from unittest.mock import MagicMock

import pytest


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class _MockChunk:
    chunk_id:       str
    text:           str
    regulator:      str = "RBI"
    issue_date:     str = "2024-01-01"
    source_url:     str = "https://rbi.org.in/test"
    metadata:       dict = field(default_factory=dict)
    effective_from: Optional[str] = None
    effective_to:   Optional[str] = None


def _make_question(qid="q0001", family="RATE", chunk_id="docA_chunk_0001", issue_date="2024-01-01"):
    from bench.question_gen.generator import QuestionItem
    return QuestionItem(
        question_id=qid,
        question="What is the CRR rate?",
        family=family,
        regulator="RBI",
        source_chunk_id=chunk_id,
        issue_date=issue_date,
        gold_chunk_ids=[chunk_id],
        expected_atoms=["RATE"],
    )


def _make_questions(n=60):
    families = ["RATE", "THRESHOLD", "SECTION", "DATE", "APPLICABILITY", "ENTITY"]
    from bench.question_gen.generator import QuestionItem
    return [
        QuestionItem(
            question_id=f"q_{i:04d}",
            question=f"Question {i}?",
            family=families[i % len(families)],
            regulator="RBI",
            source_chunk_id=f"doc{i // 5}_chunk_{i:04d}",
            issue_date=f"202{i % 4 + 1}-01-01",
            gold_chunk_ids=[f"doc{i // 5}_chunk_{i:04d}"],
            expected_atoms=["RATE"],
        )
        for i in range(n)
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2: Structured atomization schema + COMPUTED type
# ─────────────────────────────────────────────────────────────────────────────

class TestPhase2AtomSchema:
    """Tests for COMPUTED type and structured semantic fields in common.schemas."""

    def test_computed_atom_evidence_defaults(self):
        """ComputedAtomEvidence can be created with minimal fields."""
        from common.schemas import ComputedAtomEvidence
        ev = ComputedAtomEvidence(
            inputs=["5%", "18.5%"],
            operation="sum",
            formula="5% + 18.5%",
            result="23.5%",
        )
        assert ev.result == "23.5%"
        assert ev.verified is False   # not yet verified
        assert ev.result_numeric is None
        assert ev.constants == {}
        assert ev.input_evidence == []

    def test_atom_with_computed_evidence(self):
        """Atom accepts type=COMPUTED and carries ComputedAtomEvidence."""
        from common.schemas import Atom, ComputedAtomEvidence
        ev = ComputedAtomEvidence(
            inputs=["₹5 lakh", "₹1 lakh"],
            operation="sum",
            formula="₹5 lakh + ₹1 lakh",
            result="₹6 lakh",
            result_numeric=600000.0,
        )
        atom = Atom(
            atom_id="test_atom_001",
            type="COMPUTED",
            text="₹6 lakh",
            claim="The total is ₹6 lakh.",
            computed=ev,
        )
        assert atom.type == "COMPUTED"
        assert atom.computed.result == "₹6 lakh"
        assert atom.computed.result_numeric == 600000.0

    def test_atom_operator_field_accepts_enum_values(self):
        """Atom.operator field accepts AtomOperator string values."""
        from common.schemas import Atom, AtomOperator
        atom = Atom(
            atom_id="op_atom",
            type="THRESHOLD",
            text="up to ₹5 lakh",
            claim="Threshold is up to ₹5 lakh.",
            operator=AtomOperator.UP_TO,
            value="5",
            unit="lakh",
        )
        assert atom.operator == AtomOperator.UP_TO

    def test_up_to_ne_exceeding(self):
        """UP_TO and EXCEEDING are distinct AtomOperator values."""
        from common.schemas import AtomOperator
        assert AtomOperator.UP_TO != AtomOperator.EXCEEDING

    def test_computed_in_atom_types(self):
        """COMPUTED is registered in ATOM_TYPES."""
        from common.schemas import ATOM_TYPES
        assert "COMPUTED" in ATOM_TYPES

    def test_atom_structured_fields_are_optional(self):
        """Atom can be created without structured fields (backward compat)."""
        from common.schemas import Atom
        atom = Atom(
            atom_id="minimal",
            type="RATE",
            text="4.5%",
            claim="CRR is 4.5%.",
        )
        assert atom.operator is None or atom.operator == "unknown"
        assert atom.computed is None
        assert atom.subject is None

    def test_llm_extractor_operator_normalisation(self):
        """LLMAtomExtractor._normalise_operator maps known strings to AtomOperator values."""
        from atoms.llm_extract import _normalise_operator
        from common.schemas import AtomOperator
        assert _normalise_operator("up to")     == AtomOperator.UP_TO
        assert _normalise_operator("exceeding") == AtomOperator.EXCEEDING
        assert _normalise_operator("at least")  == AtomOperator.GTE
        assert _normalise_operator("=")         == AtomOperator.EQ
        assert _normalise_operator("unknown")   == AtomOperator.UNKNOWN

    def test_try_float_parses_numeric(self):
        """_try_float handles valid and invalid values."""
        from atoms.llm_extract import _try_float
        assert _try_float("3.14")  == pytest.approx(3.14)
        assert _try_float(100)     == pytest.approx(100.0)
        assert _try_float(None)    is None
        assert _try_float("abc")   is None


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3: Statistical certification split (agg_val)
# ─────────────────────────────────────────────────────────────────────────────

class TestPhase3AggValSplit:
    """Tests for the agg_val sub-split carved from agg_train."""

    def test_agg_val_present_in_splits(self):
        """split_questions now returns agg_val key."""
        from bench.splits import split_questions
        qs = _make_questions(60)
        splits = split_questions(qs)
        assert "agg_val" in splits

    def test_agg_val_non_empty_with_enough_questions(self):
        """agg_val should have at least 1 item when agg_train is non-trivial."""
        from bench.splits import split_questions
        qs = _make_questions(60)
        splits = split_questions(qs)
        # agg_val = 20% of agg_train; with 60 questions, agg_train ~30 → val ≥ 1
        assert len(splits["agg_val"]) >= 1

    def test_agg_val_disjoint_from_calibration_and_test(self):
        """agg_val questions must not appear in calibration or test."""
        from bench.splits import split_questions
        qs = _make_questions(120)
        splits = split_questions(qs)
        val_ids  = {q.question_id for q in splits["agg_val"]}
        calib_ids = {q.question_id for q in splits["calibration"]}
        test_ids  = {q.question_id for q in splits["test"]}
        assert val_ids.isdisjoint(calib_ids), "agg_val leaks into calibration"
        assert val_ids.isdisjoint(test_ids),  "agg_val leaks into test"

    def test_total_items_preserved(self):
        """Total items across all splits equals input length."""
        from bench.splits import split_questions
        qs = _make_questions(60)
        splits = split_questions(qs)
        total = sum(len(v) for v in splits.values())
        assert total == len(qs)

    def test_agg_val_frac_validation(self):
        """SplitConfig rejects invalid agg_val_frac."""
        from bench.splits import SplitConfig
        with pytest.raises(ValueError):
            SplitConfig(agg_val_frac=0.0)
        with pytest.raises(ValueError):
            SplitConfig(agg_val_frac=1.5)

    def test_empty_input_returns_empty_agg_val(self):
        """Empty question list returns empty agg_val."""
        from bench.splits import split_questions
        splits = split_questions([])
        assert splits["agg_val"] == []


# ─────────────────────────────────────────────────────────────────────────────
# Phase 4: Leakage prevention (disjoint document splitting)
# ─────────────────────────────────────────────────────────────────────────────

class TestPhase4DisjointSplit:
    """Tests for disjoint_document_split and ensure_min_calibration."""

    def _make_chunks(self, doc_ids: List[str]) -> List[_MockChunk]:
        """Create mock chunks with doc-prefixed chunk IDs."""
        chunks = []
        for doc_id in doc_ids:
            for i in range(3):
                chunks.append(_MockChunk(
                    chunk_id=f"{doc_id}_chunk_{i:04d}",
                    text=f"Regulatory text from {doc_id}.",
                ))
        return chunks

    def test_nli_train_chunks_disjoint_from_calibration(self):
        """NLI train chunks must not come from calibration/test documents."""
        from bench.splits import disjoint_document_split, SplitConfig
        # Create 8 distinct documents
        all_doc_ids = [f"docA{i}" for i in range(8)]
        all_chunks  = self._make_chunks(all_doc_ids)
        # Questions are from first 6 docs
        questions = [
            _make_question(qid=f"q{i}", chunk_id=f"docA{i}_chunk_0000")
            for i in range(6)
        ]
        result = disjoint_document_split(all_chunks, questions)
        assert "nli_train_chunks" in result
        assert "held_out_chunks"  in result
        assert "banned_doc_ids"   in result

        # Verify: banned docs are not in nli_train
        banned = set(result["banned_doc_ids"])
        for chunk in result["nli_train_chunks"]:
            doc_prefix = chunk.chunk_id.split("_")[0]
            assert doc_prefix not in banned, (
                f"Chunk {chunk.chunk_id} from banned doc {doc_prefix} in nli_train"
            )

    def test_held_out_chunks_are_from_banned_docs(self):
        """All held-out chunks come from banned (calib/test) documents."""
        from bench.splits import disjoint_document_split
        all_chunks = self._make_chunks([f"doc{i}" for i in range(5)])
        questions  = [_make_question(chunk_id=f"doc{i}_chunk_0000") for i in range(5)]
        result = disjoint_document_split(all_chunks, questions)
        banned = set(result["banned_doc_ids"])
        for chunk in result["held_out_chunks"]:
            doc_prefix = chunk.chunk_id.split("_")[0]
            assert doc_prefix in banned

    def test_ensure_min_calibration_ok(self):
        """ensure_min_calibration returns True when calibration is large enough."""
        from bench.splits import ensure_min_calibration
        splits = {"calibration": list(range(120))}
        assert ensure_min_calibration(splits, min_calib=100) is True

    def test_ensure_min_calibration_fails(self):
        """ensure_min_calibration returns False and warns when too small."""
        from bench.splits import ensure_min_calibration
        splits = {"calibration": list(range(50))}
        assert ensure_min_calibration(splits, min_calib=100) is False

    def test_empty_questions_returns_all_chunks_as_nli_train(self):
        """With no questions, no docs are banned → all chunks are nli_train."""
        from bench.splits import disjoint_document_split
        all_chunks = self._make_chunks(["docX", "docY"])
        result = disjoint_document_split(all_chunks, questions=[])
        assert len(result["held_out_chunks"]) == 0
        assert len(result["nli_train_chunks"]) == len(all_chunks)


# ─────────────────────────────────────────────────────────────────────────────
# Phase 5: Version graph & legal drift
# ─────────────────────────────────────────────────────────────────────────────

class TestPhase5VersionGraph:
    """Tests for VersionGraph instance methods and amendment_events."""

    def _make_graph(self):
        from ingest.version_graph.build import VersionGraph, VersionNode
        g = VersionGraph()
        g.add_node(VersionNode(
            doc_id="doc_2022",
            section_id="SEBI_Chapter1_Reg52",
            effective_from="2022-01-01",
            effective_to="2023-06-30",
            supersedes=[],
            amends=[],
            regulator="SEBI",
        ))
        g.add_node(VersionNode(
            doc_id="doc_2023",
            section_id="SEBI_Chapter1_Reg52",
            effective_from="2023-07-01",
            effective_to=None,
            supersedes=["doc_2022"],
            amends=[],
            regulator="SEBI",
        ))
        g.add_node(VersionNode(
            doc_id="rbi_doc_2023",
            section_id="RBI_General_Sec42",
            effective_from="2023-01-01",
            effective_to=None,
            supersedes=[],
            amends=["rbi_old"],
            regulator="RBI",
        ))
        return g

    def test_history_returns_sorted(self):
        """history() returns versions sorted ascending by effective_from."""
        g = self._make_graph()
        h = g.history("SEBI_Chapter1_Reg52")
        assert len(h) == 2
        assert h[0].effective_from < h[1].effective_from

    def test_current_version_finds_active(self):
        """current_version() returns the in-force doc_id on a given date."""
        g = self._make_graph()
        assert g.current_version("SEBI_Chapter1_Reg52", "2024-01-01") == "doc_2023"
        assert g.current_version("SEBI_Chapter1_Reg52", "2022-06-01") == "doc_2022"

    def test_current_version_unknown_section(self):
        """current_version() returns None for unknown section."""
        g = self._make_graph()
        assert g.current_version("NONEXISTENT_SECTION", "2024-01-01") is None

    def test_superseded_sections_correct(self):
        """superseded_sections() identifies sections with expired versions."""
        g = self._make_graph()
        sup = g.superseded_sections()
        assert "SEBI_Chapter1_Reg52" in sup

    def test_amendment_events_non_empty(self):
        """amendment_events() returns at least one event for an amending version."""
        g = self._make_graph()
        events = g.amendment_events()
        assert len(events) > 0
        for e in events:
            assert "date" in e
            assert "type" in e
            assert e["type"] in ("supersedes", "amends")

    def test_amendment_events_since_filter(self):
        """amendment_events(since_date) excludes events before that date."""
        g = self._make_graph()
        all_events   = g.amendment_events()
        since_events = g.amendment_events(since_date="2023-07-01")
        assert len(since_events) <= len(all_events)
        for e in since_events:
            assert e["date"] >= "2023-07-01"

    def test_query_in_force_delegates(self):
        """query.in_force() delegates to graph.current_version()."""
        from ingest.version_graph.query import in_force
        g = self._make_graph()
        result = in_force(g, "SEBI_Chapter1_Reg52", "2024-01-01")
        assert result == "doc_2023"

    def test_query_history_delegates(self):
        """query.history() delegates to graph.history()."""
        from ingest.version_graph.query import history
        g = self._make_graph()
        h = history(g, "SEBI_Chapter1_Reg52")
        assert len(h) == 2

    def test_graph_save_load_roundtrip(self, tmp_path):
        """VersionGraph saves and loads correctly."""
        from ingest.version_graph.build import VersionGraph
        g = self._make_graph()
        path = str(tmp_path / "vg.json")
        g.save(path)
        g2 = VersionGraph.load(path)
        assert "SEBI_Chapter1_Reg52" in g2.nodes
        assert len(g2.nodes["SEBI_Chapter1_Reg52"]) == 2


# ─────────────────────────────────────────────────────────────────────────────
# Phase 6: Retrieval signals — temporal entropy
# ─────────────────────────────────────────────────────────────────────────────

class TestPhase6TemporalEntropy:
    """Tests for temporal_entropy_score and enhanced temporal_filter."""

    def _make_chunk(self, chunk_id, issue_date, effective_from=None, effective_to=None):
        return _MockChunk(
            chunk_id=chunk_id,
            text="Regulatory text.",
            issue_date=issue_date,
            effective_from=effective_from,
            effective_to=effective_to,
        )

    def test_recent_chunk_lower_score(self):
        """A more recent chunk has a lower temporal entropy score."""
        from retrieval.temporal import temporal_entropy_score
        recent = self._make_chunk("c1", "2024-01-01")
        old    = self._make_chunk("c2", "2015-01-01")
        s_recent = temporal_entropy_score(recent, "2024-06-01")
        s_old    = temporal_entropy_score(old, "2024-06-01")
        assert s_recent < s_old

    def test_active_chunk_lower_score_than_expired(self):
        """A still-active chunk (no effective_to) scores lower than an expired one."""
        from retrieval.temporal import temporal_entropy_score
        active  = self._make_chunk("c1", "2023-01-01", effective_from="2023-01-01", effective_to=None)
        expired = self._make_chunk("c2", "2020-01-01", effective_from="2020-01-01", effective_to="2022-12-31")
        s_active  = temporal_entropy_score(active,  "2024-01-01")
        s_expired = temporal_entropy_score(expired, "2024-01-01")
        assert s_active < s_expired

    def test_score_in_unit_interval(self):
        """temporal_entropy_score returns a value in [0, 1]."""
        from retrieval.temporal import temporal_entropy_score
        for year in ("2010", "2020", "2024"):
            chunk = self._make_chunk(f"c_{year}", f"{year}-01-01")
            s = temporal_entropy_score(chunk, "2024-06-01")
            assert 0.0 <= s <= 1.0

    def test_temporal_filter_drops_expired(self):
        """temporal_filter drops chunks with effective_to before query_date."""
        from retrieval.temporal import temporal_filter
        from ingest.version_graph.build import VersionGraph
        expired = self._make_chunk("exp", "2020-01-01", effective_from="2020-01-01", effective_to="2021-12-31")
        active  = self._make_chunk("act", "2023-01-01", effective_from="2023-01-01", effective_to=None)
        g = VersionGraph()
        result = temporal_filter([expired, active], g, query_date="2024-01-01")
        ids = [c.chunk_id for c in result]
        assert "exp" not in ids
        assert "act" in ids

    def test_temporal_filter_drops_future(self):
        """temporal_filter drops chunks with effective_from after query_date."""
        from retrieval.temporal import temporal_filter
        from ingest.version_graph.build import VersionGraph
        future = self._make_chunk("fut", "2025-01-01", effective_from="2025-01-01", effective_to=None)
        g = VersionGraph()
        result = temporal_filter([future], g, query_date="2024-01-01")
        assert result == []


# ─────────────────────────────────────────────────────────────────────────────
# Phase 7: Judge — COMPUTED atom annotation
# ─────────────────────────────────────────────────────────────────────────────

class TestPhase7JudgeComputed:
    """Tests for _build_computed_annotation in judge_llm."""

    def _make_computed_atom(self):
        from common.schemas import Atom, ComputedAtomEvidence
        ev = ComputedAtomEvidence(
            inputs=["₹5 lakh", "₹1.5 lakh"],
            operation="sum",
            formula="₹5 lakh + ₹1.5 lakh",
            result="₹6.5 lakh",
            result_numeric=650000.0,
            constants={},
        )
        return Atom(
            atom_id="computed_001",
            type="COMPUTED",
            text="₹6.5 lakh",
            claim="The total deduction is ₹6.5 lakh.",
            computed=ev,
        )

    def test_annotation_non_empty_for_computed(self):
        """_build_computed_annotation returns a non-empty string for COMPUTED atoms."""
        from judge.judge_llm import _build_computed_annotation
        atom = self._make_computed_atom()
        ann  = _build_computed_annotation(atom)
        assert "Formula" in ann
        assert "Operation" in ann
        assert "Result" in ann

    def test_annotation_empty_for_rate_atom(self):
        """_build_computed_annotation returns '' for non-COMPUTED atoms."""
        from judge.judge_llm import _build_computed_annotation
        from common.schemas import Atom
        atom = Atom(atom_id="r1", type="RATE", text="4.5%", claim="CRR is 4.5%.")
        assert _build_computed_annotation(atom) == ""

    def test_annotation_includes_inputs(self):
        """The annotation block lists all inputs."""
        from judge.judge_llm import _build_computed_annotation
        atom = self._make_computed_atom()
        ann  = _build_computed_annotation(atom)
        assert "₹5 lakh" in ann
        assert "₹1.5 lakh" in ann

    def test_user_prompt_with_computed_annotation(self):
        """_build_user_prompt appends the annotation for COMPUTED atoms."""
        from judge.judge_llm import _build_user_prompt, _build_computed_annotation
        atom = self._make_computed_atom()
        ann  = _build_computed_annotation(atom)
        prompt = _build_user_prompt(atom.claim, "EVIDENCE TEXT", atom.type, ann)
        assert "Formula" in prompt
        assert "COMPUTED" in prompt

    def test_annotation_contains_formula(self):
        """The annotation includes the formula string."""
        from judge.judge_llm import _build_computed_annotation
        atom = self._make_computed_atom()
        ann  = _build_computed_annotation(atom)
        assert "₹5 lakh + ₹1.5 lakh" in ann


# ─────────────────────────────────────────────────────────────────────────────
# Phase 8: Router/rewrite finalization (OUTDATED, REJECTED)
# ─────────────────────────────────────────────────────────────────────────────

class TestPhase8RewriteFinalization:
    """Tests for OUTDATED and REJECTED status handling in answer_rewrite."""

    def _make_atom_decision(self, atom_id="a1", status="ABSTAINED", text="4.5%"):
        from decision.combine import AtomDecision
        return AtomDecision(
            atom_id=atom_id,
            atom_type="RATE",
            atom_text=text,
            atom_claim=f"The rate is {text}.",
            final_status=status,
            risk=0.8,
            stratum="RBI|RATE",
        )

    def _make_answer_decision(self, atom_statuses):
        from decision.combine import AnswerDecision
        from decision.answer_rewrite import ANSWER_ABSTAINED
        ads = [self._make_atom_decision(f"a{i}", s, f"{i}%") for i, s in enumerate(atom_statuses)]
        n_supported = sum(1 for s in atom_statuses if s == "SUPPORTED")
        n_abstained = sum(1 for s in atom_statuses if s == "ABSTAINED")
        n_outdated  = sum(1 for s in atom_statuses if s == "OUTDATED")
        n_rejected  = sum(1 for s in atom_statuses if s == "REJECTED")
        all_bad = n_abstained + n_outdated + n_rejected
        if all_bad == len(atom_statuses):
            answer_status = "ABSTAINED"
        elif n_supported == len(atom_statuses):
            answer_status = "FULLY_SUPPORTED"
        else:
            answer_status = "PARTIALLY_VERIFIED"
        from decision.combine import FULLY_SUPPORTED, ANSWER_ABSTAINED, PARTIALLY_VERIFIED
        return AnswerDecision(
            answer_status=answer_status,
            atom_decisions=ads,
            n_supported=n_supported,
            n_abstained=n_abstained,
        )

    def test_outdated_gets_special_tag(self):
        """OUTDATED atoms get [OUTDATED: ...] tag in flag mode."""
        from decision.answer_rewrite import _flag_mode
        atom_dec = self._make_atom_decision("a1", "OUTDATED", "5%")
        text = "The SLR rate is 5%."
        rewritten, flagged = _flag_mode(text, [atom_dec])
        assert "[OUTDATED:" in rewritten
        assert "a1" in flagged

    def test_rejected_gets_special_tag(self):
        """REJECTED atoms get [REJECTED: ...] tag in flag mode."""
        from decision.answer_rewrite import _flag_mode
        atom_dec = self._make_atom_decision("a1", "REJECTED", "5%")
        text = "The SLR rate is 5%."
        rewritten, flagged = _flag_mode(text, [atom_dec])
        assert "[REJECTED:" in rewritten

    def test_outdated_disclaimer_used_when_outdated_present(self):
        """rewrite_answer uses the OUTDATED-specific disclaimer."""
        from common.schemas import GeneratedAnswer
        from decision.answer_rewrite import rewrite_answer, _DISCLAIMER_OUTDATED
        from decision.combine import AnswerDecision, AtomDecision

        # Use a SUPPORTED + OUTDATED mix so it takes PARTIALLY_VERIFIED path,
        # not the full-abstention stub path.
        atoms = [
            AtomDecision("a0", "RATE", "18.5%", "SLR is 18.5%.", "SUPPORTED", 0.1, "RBI|RATE"),
            AtomDecision("a1", "RATE",  "5%",   "Rate is 5%.",   "OUTDATED",  0.9, "RBI|RATE"),
        ]
        answer_decision = AnswerDecision(
            answer_status="PARTIALLY_VERIFIED",
            atom_decisions=atoms,
            n_supported=1, n_abstained=0,
        )
        answer = GeneratedAnswer(
            query="What is SLR?",
            answer_text="The SLR rate is 18.5%. Rate is 5%.",
            citations=[],
            token_logprobs=None,
            model_id="test",
        )
        rw = rewrite_answer(answer, answer_decision, mode="flag")
        assert _DISCLAIMER_OUTDATED in rw.disclaimer

    def test_rejected_disclaimer_takes_priority_over_outdated(self):
        """REJECTED disclaimer is chosen when both REJECTED and OUTDATED atoms exist."""
        from common.schemas import GeneratedAnswer
        from decision.answer_rewrite import rewrite_answer, _DISCLAIMER_REJECTED
        from decision.combine import AnswerDecision, AtomDecision

        # Mix in a SUPPORTED atom so we hit PARTIALLY_VERIFIED, not full-abstention stub.
        atoms = [
            AtomDecision("a0", "RATE", "18.5%", "SLR is 18.5%.", "SUPPORTED", 0.1, "RBI|RATE"),
            AtomDecision("a1", "RATE", "5%",    "Rate is 5%.",   "OUTDATED",  0.9, "RBI|RATE"),
            AtomDecision("a2", "RATE", "4%",    "Rate is 4%.",   "REJECTED",  0.95, "RBI|RATE"),
        ]
        ad = AnswerDecision(
            answer_status="PARTIALLY_VERIFIED",
            atom_decisions=atoms,
            n_supported=1, n_abstained=0,
        )
        answer = GeneratedAnswer(
            query="What is SLR?",
            answer_text="The SLR rate is 18.5%. Rate is 5%. Rate is 4%.",
            citations=[], token_logprobs=None, model_id="test",
        )
        rw = rewrite_answer(answer, ad, mode="flag")
        assert _DISCLAIMER_REJECTED in rw.disclaimer

    def test_fully_supported_no_rewrite(self):
        """Fully supported answer is returned unchanged."""
        from common.schemas import GeneratedAnswer
        from decision.answer_rewrite import rewrite_answer
        from decision.combine import AnswerDecision, AtomDecision

        ad_obj = AtomDecision("a1", "RATE", "5%", "Rate is 5%.", "SUPPORTED", 0.1, "RBI|RATE")
        answer_decision = AnswerDecision(
            answer_status="FULLY_SUPPORTED",
            atom_decisions=[ad_obj],
            n_supported=1,
        )
        answer = GeneratedAnswer(
            query="Q?",
            answer_text="Rate is 5%.",
            citations=[],
            token_logprobs=None,
            model_id="test",
        )
        rw = rewrite_answer(answer, answer_decision, mode="flag")
        assert rw.rewrite_mode == "none"
        assert rw.disclaimer == ""


# ─────────────────────────────────────────────────────────────────────────────
# Phase 9: Benchmark coverage (ensure_min_calibration + agg_val save/load)
# ─────────────────────────────────────────────────────────────────────────────

class TestPhase9BenchmarkCoverage:
    """Phase 9 tests: splits are saved/loaded with agg_val, coverage guard works."""

    def test_save_load_includes_agg_val(self, tmp_path):
        """save_split / load_split round-trips agg_val correctly."""
        from bench.splits import split_questions, save_split, load_split
        qs = _make_questions(80)
        splits = split_questions(qs)
        save_split(splits, str(tmp_path))
        loaded = load_split(str(tmp_path))
        # Must have agg_val key
        assert "agg_val" in loaded
        assert len(loaded["agg_val"]) == len(splits["agg_val"])

    def test_all_question_ids_unique_across_splits(self):
        """No question_id appears in more than one split."""
        from bench.splits import split_questions
        qs = _make_questions(120)
        splits = split_questions(qs)
        all_ids = [q.question_id for s in splits.values() for q in s]
        assert len(all_ids) == len(set(all_ids)), "Duplicate question IDs across splits"

    def test_coverage_guard_warns_when_small(self):
        """ensure_min_calibration returns False for small calibration sets."""
        from bench.splits import ensure_min_calibration
        result = ensure_min_calibration({"calibration": list(range(10))}, min_calib=50)
        assert result is False

    def test_split_by_document_produces_four_keys(self):
        """split_by='document' produces the four expected keys."""
        from bench.splits import split_questions, SplitConfig
        qs = _make_questions(80)
        splits = split_questions(qs, SplitConfig(split_by="document"))
        assert set(splits.keys()) >= {"agg_train", "agg_val", "calibration", "test"}

    def test_split_by_random_produces_four_keys(self):
        """split_by='random' produces the four expected keys."""
        from bench.splits import split_questions, SplitConfig
        qs = _make_questions(80)
        splits = split_questions(qs, SplitConfig(split_by="random"))
        assert set(splits.keys()) >= {"agg_train", "agg_val", "calibration", "test"}
