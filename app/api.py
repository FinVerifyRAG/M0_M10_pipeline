"""
app/api.py
----------
M9: FastAPI application that exposes the RegGuard pipeline via a REST API.

Endpoints
---------
POST /ask
    Request:  { "question": str, "date": str?, "history": [...] }
    Response: { "answer": str, "atoms": [...], "sources": [...],
                "verification": {...}, "latency": {...} }

GET  /health
    Returns service status.

GET  /docs
    Auto-generated OpenAPI docs (built into FastAPI).

Run
---
    uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload

Or via the CLI helper:
    python -m app.api

Dependencies
------------
    pip install fastapi uvicorn
"""
from __future__ import annotations

import logging
import sys
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure the project root is on the path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field, field_validator
    _FASTAPI_AVAILABLE = True
except ImportError:
    _FASTAPI_AVAILABLE = False

from app.pipeline import run_pipeline, reset_pipeline, PipelineResult

logger = logging.getLogger("app.api")
logging.basicConfig(level=logging.INFO)

# ══════════════════════════════════════════════════════════════════════════════
# Pydantic I/O schemas (separate from common/schemas.py to keep API clean)
# ══════════════════════════════════════════════════════════════════════════════

class HistoryTurn(BaseModel):
    role:    str = Field(..., description="'user' or 'assistant'")
    content: str = Field(..., description="Turn text")


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000,
                          description="Regulatory question to answer")
    date:     Optional[str] = Field(None,
                                    description="Query date ISO-8601 (defaults to today)")
    history:  Optional[List[HistoryTurn]] = Field(None,
                                                   description="Conversation history")
    skip_judge: bool = Field(False,
                              description="Skip M8 judge for faster (less precise) responses")
    rewrite_mode: str = Field("flag",
                               description="Answer rewrite mode: flag | trim | none")

    @field_validator("rewrite_mode")
    @classmethod
    def _valid_mode(cls, v: str) -> str:
        allowed = {"flag", "trim", "stub_on_abstain", "none"}
        if v not in allowed:
            raise ValueError(f"rewrite_mode must be one of {allowed}")
        return v


class AtomResponse(BaseModel):
    atom_id:       str
    atom_type:     str
    text:          str
    claim:         str
    status:        str
    risk:          float
    risk_badge:    str          # "LOW" | "MEDIUM" | "HIGH"
    stratum:       str
    v1_status:     Optional[str]   = None
    v2_entail_prob: Optional[float] = None
    rationale:     Optional[str]   = None
    evidence_quote: Optional[str]  = None


class SourceResponse(BaseModel):
    chunk_id:   str
    regulator:  str
    issue_date: str
    source_url: str
    section:    str
    preview:    str


class VerificationResponse(BaseModel):
    status:         str
    status_label:   str
    coverage:       float
    n_total:        int
    n_supported:    int
    n_verified:     int
    n_uncertain:    int
    n_abstained:    int
    n_not_verified: int
    judge_calls:    int
    explanation:    str


class AskResponse(BaseModel):
    query:           str
    query_date:      str
    answer:          str
    original_answer: str
    disclaimer:      str
    rewrite_mode:    str
    atoms:           List[AtomResponse]
    sources:         List[SourceResponse]
    verification:    VerificationResponse
    latency:         Dict[str, float]
    error:           Optional[str] = None


# ══════════════════════════════════════════════════════════════════════════════
# FastAPI app
# ══════════════════════════════════════════════════════════════════════════════

if _FASTAPI_AVAILABLE:
    app = FastAPI(
        title="RegGuard API",
        description=(
            "Regulatory Q&A pipeline with atom-level verification, "
            "Learn-then-Test risk certificates, and a V3 LLM judge."
        ),
        version="1.0.0",
    )

    # CORS (allow all in dev; restrict in production)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── /health ───────────────────────────────────────────────────────────────

    @app.get("/health", tags=["status"])
    def health() -> Dict[str, Any]:
        """Returns service liveness and the current date."""
        return {"status": "ok", "today": date.today().isoformat()}

    # ── /pipeline/reset ───────────────────────────────────────────────────────

    @app.post("/pipeline/reset", tags=["admin"])
    def pipeline_reset() -> Dict[str, str]:
        """Force re-load of all pipeline singletons (e.g. after re-indexing)."""
        reset_pipeline()
        return {"status": "reset"}

    # ── /ask ──────────────────────────────────────────────────────────────────

    @app.post("/ask", response_model=AskResponse, tags=["qa"])
    def ask(req: AskRequest) -> AskResponse:
        """
        Run the full M2→M8 pipeline and return a structured, verified answer.

        - **question**: your regulatory query
        - **date**: optional ISO date for temporal filtering (defaults to today)
        - **history**: optional previous turns for follow-up resolution
        - **skip_judge**: set True for faster but less precise responses
        - **rewrite_mode**: how to handle unverified atoms in the answer text
        """
        query_date = req.date or date.today().isoformat()
        history = [{"role": t.role, "content": t.content} for t in (req.history or [])]

        result: PipelineResult = run_pipeline(
            question=req.question,
            query_date=query_date,
            history=history,
            skip_judge=req.skip_judge,
            rewrite_mode=req.rewrite_mode,
        )

        return AskResponse(
            query=result.query,
            query_date=result.query_date,
            answer=result.answer,
            original_answer=result.original_answer,
            disclaimer=result.disclaimer,
            rewrite_mode=result.rewrite_mode,
            atoms=[
                AtomResponse(
                    atom_id=a.atom_id,
                    atom_type=a.atom_type,
                    text=a.text,
                    claim=a.claim,
                    status=a.status,
                    risk=a.risk,
                    risk_badge=a.risk_label,
                    stratum=a.stratum,
                    v1_status=a.v1_status,
                    v2_entail_prob=a.v2_entail_prob,
                    rationale=a.rationale,
                    evidence_quote=a.evidence_quote,
                )
                for a in result.atoms
            ],
            sources=[
                SourceResponse(
                    chunk_id=s.chunk_id,
                    regulator=s.regulator,
                    issue_date=s.issue_date,
                    source_url=s.source_url,
                    section=s.section,
                    preview=s.preview,
                )
                for s in result.sources
            ],
            verification=VerificationResponse(**vars(result.verification)),
            latency=result.latency,
            error=result.error,
        )

    # ── Global error handler ──────────────────────────────────────────────────

    @app.exception_handler(Exception)
    async def _global_error(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled API error: %s", exc)
        return JSONResponse(
            status_code=500,
            content={"detail": f"Internal server error: {exc}"},
        )

else:
    # Graceful fallback when FastAPI is not installed
    app = None  # type: ignore[assignment]
    logger.warning(
        "FastAPI not installed. Install with: pip install fastapi uvicorn"
    )


# ══════════════════════════════════════════════════════════════════════════════
# CLI entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if not _FASTAPI_AVAILABLE:
        print("FastAPI not installed. Run: pip install fastapi uvicorn")
        sys.exit(1)
    try:
        import uvicorn
        uvicorn.run("app.api:app", host="0.0.0.0", port=8000, reload=True)
    except ImportError:
        print("uvicorn not installed. Run: pip install uvicorn")
        sys.exit(1)
