"""
guarantee/mondrian.py
----------------------
Mondrian stratification for the guarantee layer.

Strata are defined as ``regulator|atom_type`` (e.g. ``SEBI|RATE``).
When a stratum has too few calibration atoms (< n_min), it merges upward
using a fallback hierarchy:

    regulator|atom_type  ->  *|atom_type  ->  *|*

Delta is split across strata using Bonferroni correction so that the
family-wise error rate is at most delta:
    delta_stratum = delta / n_strata

Public API
----------
    from guarantee.mondrian import (
        get_stratum,
        build_strata,
        allocate_delta,
        get_delta_for_stratum,
    )
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger("guarantee.mondrian")

# Default minimum calibration atoms required in a stratum
DEFAULT_N_MIN = 100


# ---------------------------------------------------------------------------
# Stratum key helpers
# ---------------------------------------------------------------------------

def get_stratum(regulator: str, atom_type: str) -> str:
    """
    Return the canonical stratum name for a (regulator, atom_type) pair.

    Examples
    --------
    >>> get_stratum("SEBI", "RATE")
    'SEBI|RATE'
    >>> get_stratum("", "DATE")
    '*|DATE'
    """
    reg  = regulator.upper().strip() if regulator else "*"
    atype = atom_type.upper().strip() if atom_type else "*"
    return f"{reg}|{atype}"


def fallback_hierarchy(stratum: str) -> List[str]:
    """
    Return the fallback chain for a stratum, from most-specific to
    least-specific (global fallback ``*|*``).

    Examples
    --------
    >>> fallback_hierarchy("SEBI|RATE")
    ['SEBI|RATE', '*|RATE', '*|*']
    """
    parts = stratum.split("|", 1)
    reg   = parts[0] if len(parts) > 0 else "*"
    atype = parts[1] if len(parts) > 1 else "*"
    chain = [f"{reg}|{atype}"]
    if reg != "*":
        chain.append(f"*|{atype}")
    if chain[-1] != "*|*":
        chain.append("*|*")
    return chain


# ---------------------------------------------------------------------------
# Stratum dataclass
# ---------------------------------------------------------------------------

@dataclass
class Stratum:
    """Holds per-stratum calibration data and computed thresholds."""
    name: str                    # e.g. "SEBI|RATE"
    risk:  np.ndarray            # risk scores (calibration atoms)
    wrong: np.ndarray            # labels (1=wrong, 0=correct)
    accept_below:  Optional[float] = None   # LTT-certified threshold
    abstain_above: Optional[float] = None
    n_atoms: int = 0             # filled after __post_init__
    merged_from: Optional[str] = None  # original stratum if this was merged

    def __post_init__(self):
        self.n_atoms = len(self.risk)

    @property
    def empirical_risk(self) -> float:
        """Empirical risk (fraction wrong) in this stratum."""
        if self.n_atoms == 0:
            return float("nan")
        return float(self.wrong.sum() / self.n_atoms)


# ---------------------------------------------------------------------------
# Build strata from a DataFrame / array of atoms
# ---------------------------------------------------------------------------

def build_strata(
    risk:       np.ndarray,
    wrong:      np.ndarray,
    regulators: List[str],
    atom_types: List[str],
    n_min:      int = DEFAULT_N_MIN,
) -> Dict[str, Stratum]:
    """
    Partition calibration atoms into Mondrian strata by (regulator, atom_type).
    Small strata (<n_min atoms) are merged upward to the global fallback.

    Parameters
    ----------
    risk       : risk scores, shape (N,).
    wrong      : binary labels (1=wrong), shape (N,).
    regulators : regulator name per atom, shape (N,), e.g. ["SEBI", "RBI", ...].
    atom_types : atom type per atom, shape (N,), e.g. ["RATE", "DATE", ...].
    n_min      : minimum number of atoms for a stratum to be kept standalone.

    Returns
    -------
    Dict mapping stratum_name -> Stratum.  Includes the global ``*|*`` stratum.
    """
    risk       = np.asarray(risk,  dtype=float)
    wrong      = np.asarray(wrong, dtype=float)
    regulators = list(regulators)
    atom_types = list(atom_types)

    # 1. Build raw strata
    raw: Dict[str, list] = {}
    for i, (reg, atype) in enumerate(zip(regulators, atom_types)):
        key = get_stratum(reg, atype)
        raw.setdefault(key, [])
        raw[key].append(i)

    strata: Dict[str, Stratum] = {}
    merged_into: Dict[str, str] = {}  # original -> merged_name

    # 2. Merge small strata upward
    for name, idxs in sorted(raw.items()):
        if len(idxs) >= n_min:
            strata[name] = Stratum(
                name=name,
                risk=risk[idxs],
                wrong=wrong[idxs],
            )
        else:
            # Find the fallback that already has enough atoms (after merging)
            fallbacks = fallback_hierarchy(name)
            target = fallbacks[-1]  # default to *|*
            for fb in fallbacks[1:]:
                # How many total atoms would the fallback have after merging?
                existing = len(raw.get(fb, []))
                incoming = len(idxs)
                if existing + incoming >= n_min or fb == "*|*":
                    target = fb
                    break

            if target not in raw:
                raw[target] = []
            raw[target].extend(idxs)
            merged_into[name] = target
            logger.info(
                "Stratum %s has only %d atoms (<n_min=%d); merging into %s",
                name, len(idxs), n_min, target,
            )

    # 3. Build merged strata (those that absorbed small strata)
    for name, idxs in sorted(raw.items()):
        if name in strata:
            continue  # Already built as a standalone stratum
        if any(v == name for v in merged_into.values()):
            strata[name] = Stratum(
                name=name,
                risk=risk[idxs],
                wrong=wrong[idxs],
            )

    # 4. Ensure the global *|* stratum exists
    if "*|*" not in strata:
        all_idxs = list(range(len(risk)))
        strata["*|*"] = Stratum(name="*|*", risk=risk[all_idxs], wrong=wrong[all_idxs])

    logger.info("Built %d strata: %s", len(strata), list(strata.keys()))
    return strata


# ---------------------------------------------------------------------------
# Delta allocation (Bonferroni)
# ---------------------------------------------------------------------------

def allocate_delta(delta: float, strata: Dict[str, Stratum]) -> Dict[str, float]:
    """
    Split the family-wise error rate ``delta`` across all strata using
    Bonferroni correction:
        delta_stratum = delta / n_strata.

    The global ``*|*`` stratum is included in the count.

    Returns
    -------
    Dict mapping stratum_name -> delta_stratum.
    """
    n = len(strata)
    per_stratum = delta / n if n > 0 else delta
    return {name: per_stratum for name in strata}


def get_delta_for_stratum(
    stratum_name: str,
    delta_map:    Dict[str, float],
) -> float:
    """
    Retrieve the delta for the given stratum, walking the fallback chain if
    the exact stratum is not in ``delta_map``.
    """
    for fb in fallback_hierarchy(stratum_name):
        if fb in delta_map:
            return delta_map[fb]
    return delta_map.get("*|*", 0.05)


# ---------------------------------------------------------------------------
# Threshold lookup (with fallback chain)
# ---------------------------------------------------------------------------

def lookup_threshold(
    stratum_name: str,
    strata:       Dict[str, Stratum],
) -> Optional[Stratum]:
    """
    Return the Stratum object for the best matching stratum, following the
    fallback chain:  regulator|type  ->  *|type  ->  *|*

    Returns None only if no stratum whatsoever is found (should not happen
    when ``*|*`` is always present).
    """
    for fb in fallback_hierarchy(stratum_name):
        if fb in strata:
            return strata[fb]
    return None
