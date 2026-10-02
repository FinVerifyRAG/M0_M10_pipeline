"""
aggregate/ — Multi-signal aggregator that combines all signals into a
single atom-level risk score in [0, 1].

Public API:
    from aggregate.model import score
    scored_atoms = score(verified_atoms, answer, rr)
"""
