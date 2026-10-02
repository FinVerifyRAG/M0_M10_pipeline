"""
decision/router.py
-------------------
M8 Decision Router: maps each ScoredAtom to SUPPORTED / UNCERTAIN / ABSTAINED
using certified M7 thresholds.

Logic (per atom, per stratum)
------------------------------
    stratum = regulator|atom_type  (e.g. "SEBI|RATE")
    t = lookup(stratum, thresholds)   # fallback chain: reg|type -> *|type -> *|*

    if   risk <= t.accept_below:  -> SUPPORTED   (within certified error bound)
    elif risk >= t.abstain_above: -> ABSTAINED   (too risky; do not show as fact)
    else:                         -> UNCERTAIN   (send to V3 judge)

Edge cases
----------
- accept_below == -1.0: stratum could not be certified; everything is ABSTAINED.
- abstain_above == accept_below: no UNCERTAIN band; atom is SUPPORTED or ABSTAINED.
- Thresholds not found for stratum: fall back to global ``*|*``; if that also
  missing, default to ABSTAINED (safe default).

Guarantee note
--------------
SUPPORTED atoms carry the M7 guarantee:
    P(selective risk > eps | calibration data) <= delta.
UNCERTAIN atoms sent to the judge are outside the certified bound unless the
end-to-end pipeline has been jointly calibrated (see judge_llm.py).

Public API
----------
    from decision.router import route, RouteResult, SUPPORTED, UNCERTAIN, ABSTAINED

    results = route(scored_atoms, thresholds)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

from common.schemas import ScoredAtom, Thresholds, DecisionState
from guarantee.mondrian import get_stratum, fallback_hierarchy

logger = logging.getLogger("decision.router")

# Status constants (use DecisionState values for consistency)
SUPPORTED  = DecisionState.SUPPORTED.value
UNCERTAIN  = DecisionState.UNCERTAIN.value
ABSTAINED  = DecisionState.ABSTAINED.value
REJECTED   = DecisionState.REJECTED.value
OUTDATED   = DecisionState.OUTDATED.value


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class RouteResult:
    """Routing decision for a single ScoredAtom."""
    atom_id:       str
    status:        str    # SUPPORTED | UNCERTAIN | ABSTAINED
    risk:          float
    stratum:       str    # The stratum used (may be a fallback)
    accept_below:  Optional[float] = None
    abstain_above: Optional[float] = None


# ---------------------------------------------------------------------------
# Threshold lookup with fallback chain
# ---------------------------------------------------------------------------

def _lookup(
    stratum_name: str,
    thresholds:   Dict[str, Thresholds],
) -> Optional[Thresholds]:
    """
    Walk the fallback chain: reg|type -> *|type -> *|* until a match is found.
    Returns None only if thresholds dict is completely empty.
    """
    for fb in fallback_hierarchy(stratum_name):
        if fb in thresholds:
            return thresholds[fb]
    return None


# ---------------------------------------------------------------------------
# Single-atom routing
# ---------------------------------------------------------------------------

def route_one(
    scored_atom: ScoredAtom,
    thresholds:  Dict[str, Thresholds],
) -> RouteResult:
    """
    Route a single ScoredAtom to SUPPORTED / UNCERTAIN / ABSTAINED.

    Parameters
    ----------
    scored_atom : ScoredAtom from M6.
    thresholds  : Dict[stratum_name -> Thresholds] from M7 certify.

    Returns
    -------
    RouteResult.
    """
    atom     = scored_atom.verified.atom
    risk_val = scored_atom.risk

    # Build the stratum key from atom metadata or atom type
    regulator = ""
    if hasattr(atom, "metadata") and atom.metadata:
        regulator = str(atom.metadata.get("regulator", "")).upper()
    # Also try the VerifiedAtom's metadata (set by cascade.py / signals)
    if not regulator and hasattr(scored_atom.verified, "metadata"):
        regulator = str(scored_atom.verified.metadata.get("regulator", "")).upper()

    atom_type    = atom.type
    stratum_name = get_stratum(regulator, atom_type)

    # Threshold lookup with fallback
    th = _lookup(stratum_name, thresholds)
    used_stratum = stratum_name  # May differ if fallback used

    if th is None:
        logger.warning(
            "No threshold found for stratum %s (and no *|* fallback). "
            "Defaulting to ABSTAINED.",
            stratum_name,
        )
        return RouteResult(
            atom_id=atom.atom_id,
            status=ABSTAINED,
            risk=risk_val,
            stratum=stratum_name,
        )

    # Track which stratum was actually used
    used_stratum = th.stratum_name

    accept_below  = th.accept_below
    abstain_above = th.abstain_above

    # Bug 4.3 fix: accept_below=None is the canonical fail-closed state.
    # NEVER do: risk_val <= None.
    # The old -1.0 sentinel is removed; None is now the schema type.
    if accept_below is None:
        # No certified threshold for this stratum -> fail closed
        status = ABSTAINED
    elif accept_below >= abstain_above:
        # Invariant violation: log and fail closed
        logger.error(
            "Threshold ordering violated for stratum %s: "
            "accept_below=%.4f >= abstain_above=%.4f. Failing closed.",
            used_stratum, accept_below, abstain_above,
        )
        status = ABSTAINED
    elif risk_val <= accept_below:
        status = SUPPORTED
    elif risk_val >= abstain_above:
        status = ABSTAINED
    else:
        status = UNCERTAIN  # Falls in (accept_below, abstain_above) -> send to judge

    logger.debug(
        "atom=%s  stratum=%s  risk=%.3f  accept_below=%.3f  "
        "abstain_above=%.3f  -> %s",
        atom.atom_id, used_stratum, risk_val, accept_below, abstain_above, status,
    )

    return RouteResult(
        atom_id=atom.atom_id,
        status=status,
        risk=risk_val,
        stratum=used_stratum,
        accept_below=accept_below,
        abstain_above=abstain_above,
    )


# ---------------------------------------------------------------------------
# Batch routing
# ---------------------------------------------------------------------------

def route(
    scored_atoms: List[ScoredAtom],
    thresholds:   Dict[str, Thresholds],
) -> List[RouteResult]:
    """
    Route all ScoredAtoms to their statuses.

    Parameters
    ----------
    scored_atoms : list from M6 aggregator (all atoms for a query).
    thresholds   : loaded from thresholds.json via guarantee.certify.load_thresholds.

    Returns
    -------
    List[RouteResult] in the same order as input.
    """
    results = [route_one(sa, thresholds) for sa in scored_atoms]

    n_supported  = sum(1 for r in results if r.status == SUPPORTED)
    n_uncertain  = sum(1 for r in results if r.status == UNCERTAIN)
    n_abstained  = sum(1 for r in results if r.status == ABSTAINED)

    logger.info(
        "Routing complete: %d atoms | SUPPORTED=%d  UNCERTAIN=%d  ABSTAINED=%d",
        len(results), n_supported, n_uncertain, n_abstained,
    )
    return results
