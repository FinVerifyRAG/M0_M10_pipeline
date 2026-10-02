"""
common/schemas.py
-----------------
Core Pydantic schemas for the RegGuard pipeline.

Schema changelog
----------------
v1.1 (Bug 4 / Design Gap 6 / Section 40 fixes)
  - Atom: added structured fields (subject, operator, value, unit, condition,
          effective_period, atom_type enum, source_span).
  - ATOM_TYPES: added COMPUTED type.
  - Thresholds: accept_below is now Optional[float] (None = fail-closed).
  - Decision: status vocabulary expanded to explicit states.
  - DecisionState enum added with documented semantics.
  - ComputedAtomEvidence added for COMPUTED atom type.
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Any, Dict, Tuple
from enum import Enum


# ---------------------------------------------------------------------------
# Decision state vocabulary  (Section 40)
# ---------------------------------------------------------------------------

class DecisionState(str, Enum):
    """
    Explicit decision states with documented semantics.

    No implicit state should be interpreted as SUPPORTED.

    SUPPORTED   : risk <= accept_below; certified by LTT guarantee.
                  Transition rule: emit atom as verified.
    UNCERTAIN   : risk in (accept_below, abstain_above); send to judge.
                  Transition rule: judge decides SUPPORTED or ABSTAINED.
    ABSTAINED   : risk >= abstain_above, or judge cap reached, or judge
                  returned ABSTAINED.  Do not show atom as fact.
                  Transition rule: remove or flag in answer rewrite.
    REJECTED    : V1 returned MISMATCH (explicit contradiction in evidence).
                  Transition rule: treat as a factual error.
    OUTDATED    : Evidence chunk is temporally ineligible for the query date.
                  Transition rule: abstain with OUTDATED reason code.
    """
    SUPPORTED = "SUPPORTED"
    UNCERTAIN = "UNCERTAIN"
    ABSTAINED = "ABSTAINED"
    REJECTED  = "REJECTED"
    OUTDATED  = "OUTDATED"


# ---------------------------------------------------------------------------
# Atom operator vocabulary  (Design Gap 6.1)
# ---------------------------------------------------------------------------

class AtomOperator(str, Enum):
    """
    Semantic operator for the atom proposition.
    '₹5 lakh' means '= 5 lakh'; 'up to ₹5 lakh' means '<= 5 lakh'.
    These are NOT equivalent and must be distinguished.
    """
    EQ          = "="
    LT          = "<"
    LTE         = "<="
    GT          = ">"
    GTE         = ">="
    UP_TO       = "up_to"        # <= semantically (inclusive upper bound)
    EXCEEDING   = "exceeding"    # > semantically
    BETWEEN     = "between"
    APPROXIMATELY = "approximately"
    UNKNOWN     = "unknown"


# ---------------------------------------------------------------------------
# Atom types
# ---------------------------------------------------------------------------

ATOM_TYPES = {
    "RATE",          # Percentage / rate value
    "THRESHOLD",     # Rupee / time amount
    "SECTION",       # Section / regulation reference
    "DATE",          # Any date
    "ENTITY",        # Named entity
    "APPLICABILITY", # Scope / applicability clause
    "COMPUTED",      # Derived/calculated value (Design Gap 7)
}


# ---------------------------------------------------------------------------
# Computed atom evidence  (Design Gap 7)
# ---------------------------------------------------------------------------

class ComputedAtomEvidence(BaseModel):
    """
    Evidence for a COMPUTED atom type.

    A COMPUTED atom is one whose value does not literally appear in the
    evidence but is derived from evidence through a deterministic calculation.

    Design Gap 7 requirement: inputs, operation, constants, formula, result,
    and evidence for each input must all be recorded.

    Do NOT treat a computed result as ordinary lexical evidence.
    """
    inputs:         List[str]            # Raw evidence spans used as inputs
    operation:      str                  # E.g. "sum", "multiply", "slab_tax"
    formula:        str                  # Human-readable formula
    constants:      Dict[str, Any] = Field(default_factory=dict)
    result:         str                  # Computed result as string
    result_numeric: Optional[float] = None   # Parsed numeric result
    input_evidence: List[str] = Field(default_factory=list)  # chunk_ids
    verified:       bool = False         # True if deterministic check passed


# ---------------------------------------------------------------------------
# Structured Atom schema  (Design Gap 6.1)
# ---------------------------------------------------------------------------

class Atom(BaseModel):
    """
    Structured atom.

    Design Gap 6 requirement: atoms must no longer be unstructured text
    fragments.  The semantic slots below allow V1 to match the structured
    tuple inside the best-aligned sentence, not the whole chunk.

    Operator examples:
      'up to ₹5 lakh'    -> operator=UP_TO,  value='5', unit='lakh'
      'exceeding ₹5 lakh' -> operator=EXCEEDING, value='5', unit='lakh'
    These are NOT equivalent and must not produce the same atom.
    """
    atom_id:          str
    type:             str   # One of ATOM_TYPES
    text:             str   # Exact substring copied from the answer

    # --- structured semantic fields (Design Gap 6.1) ---
    subject:          Optional[str] = None   # What the atom is about
    operator:         str = AtomOperator.UNKNOWN  # Relational operator
    value:            Optional[str] = None   # Numeric / date / section value
    unit:             Optional[str] = None   # %, lakh, crore, bps, ...
    condition:        Optional[str] = None   # Qualifying condition
    effective_period: Optional[str] = None   # E.g. "FY 2024-25"
    source_span:      Optional[str] = None   # Verbatim span from evidence

    # --- legacy / NLI claim ---
    claim:            str = ""               # Full self-contained sentence
    cited_chunk:      Optional[str] = None   # chunk_id nearest to the fact
    span:             Optional[Tuple[int, int]] = None  # char offsets in answer

    # --- computed atom evidence ---
    computed:         Optional[ComputedAtomEvidence] = None

    # --- metadata ---
    metadata:         Dict[str, Any] = Field(default_factory=dict)
    schema_version:   str = "1.1"

    def is_computed(self) -> bool:
        return self.type == "COMPUTED"


# ---------------------------------------------------------------------------
# Chunk schema
# ---------------------------------------------------------------------------

class Chunk(BaseModel):
    chunk_id:      str
    text:          str
    regulator:     str
    issue_date:    str
    source_url:    str
    effective_from: Optional[str] = None
    effective_to:   Optional[str] = None
    superseded_by:  Optional[str] = None
    metadata:       Dict[str, Any] = Field(default_factory=dict)
    schema_version: str = "1.0"


class RetrievalResult(BaseModel):
    query:         str
    query_date:    str
    chunks:        List[Chunk]
    metadata:      Dict[str, Any] = Field(default_factory=dict)
    schema_version: str = "1.0"


class GeneratedAnswer(BaseModel):
    query:          str
    answer_text:    str
    citations:      List[str]
    token_logprobs: Optional[List[float]] = None
    model_id:       Optional[str] = None
    metadata:       Dict[str, Any] = Field(default_factory=dict)
    schema_version: str = "1.0"


class VerifiedAtom(BaseModel):
    atom:           Atom
    v1_status:      str   # MATCH, MISMATCH, NOT_FOUND, NA
    v2_entail_prob: Optional[float] = None
    metadata:       Dict[str, Any] = Field(default_factory=dict)
    schema_version: str = "1.0"


class ScoredAtom(BaseModel):
    verified:       VerifiedAtom
    risk:           float
    metadata:       Dict[str, Any] = Field(default_factory=dict)
    schema_version: str = "1.0"


class Thresholds(BaseModel):
    """
    Certified thresholds for a Mondrian stratum.

    accept_below  : Optional[float]
        Largest certified lambda.  None means no lambda was certified;
        ALL atoms in this stratum must be ABSTAINED (fail-closed).
        MUST NEVER be compared with None in downstream code:
            if accept_below is not None and risk <= accept_below: ...

    abstain_above : float
        Judge-budget threshold (Section 10, Option B).
        This is a RESOURCE-CONTROL threshold, NOT a second certified
        risk guarantee.  Atoms above this are directly ABSTAINED without
        invoking the judge.

    Invariant: when accept_below is not None,
        accept_below < abstain_above
    """
    stratum_name:  str
    accept_below:  Optional[float] = None   # Bug 4 fix: now Optional
    abstain_above: float                    # Always set; judge-budget threshold
    metadata:      Dict[str, Any] = Field(default_factory=dict)
    schema_version: str = "1.1"


class Decision(BaseModel):
    """
    Final decision for one atom.

    status: one of DecisionState.  Use the enum values to avoid typos.
    """
    atom_id:        str
    status:         str   # DecisionState value
    risk:           float
    reason:         Optional[str] = None   # Human-readable reason
    schema_version: str = "1.1"
