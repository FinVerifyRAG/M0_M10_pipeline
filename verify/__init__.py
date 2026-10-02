"""
M5: verify/ — Verification Cascade (V1 deterministic + V2 NLI)

Public API:
    from verify.cascade import verify
    verified_atoms: list[VerifiedAtom] = verify(atoms, retrieval_result)
"""
from verify.cascade import verify

__all__ = ["verify"]
