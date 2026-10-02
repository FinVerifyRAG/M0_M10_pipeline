"""
app/badge.py
------------
M9: Risk-badge mapping and atom-status badge rendering.

Maps a float risk score in [0, 1] to a LOW / MEDIUM / HIGH label and colour,
and maps atom-level decision statuses (SUPPORTED, VERIFIED, UNCERTAIN,
ABSTAINED, NOT_VERIFIED) to colour-coded badge strings for the UI.

Public API
----------
    from app.badge import risk_badge, status_badge, RiskBadge

    b = risk_badge(0.18)        # RiskBadge(label="LOW", color="#1F6B4A", emoji="🟢")
    html = b.html()             # '<span style="...">🟢 LOW</span>'

    sb = status_badge("SUPPORTED")  # StatusBadge(label="SUPPORTED", color="#1F6B4A")
    html = sb.html()
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# ── Thresholds (override via configs/app.yaml if needed) ─────────────────────
_LOW_MAX  = 0.25
_MED_MAX  = 0.60

# ── Colour palette ────────────────────────────────────────────────────────────
_FOREST   = "#1F6B4A"
_AMBER    = "#9A6B2F"
_BURGUNDY = "#8B2E2E"
_SLATE    = "#4A5A6A"

_STATUS_COLORS: dict[str, str] = {
    "SUPPORTED":    _FOREST,
    "VERIFIED":     _FOREST,
    "UNCERTAIN":    _AMBER,
    "ABSTAINED":    _BURGUNDY,
    "NOT_VERIFIED": _BURGUNDY,
}

_STATUS_EMOJIS: dict[str, str] = {
    "SUPPORTED":    "✅",
    "VERIFIED":     "✅",
    "UNCERTAIN":    "⚠️",
    "ABSTAINED":    "🚫",
    "NOT_VERIFIED": "❌",
}


# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class RiskBadge:
    """Risk badge for a float risk score."""
    label: str          # "LOW" | "MEDIUM" | "HIGH"
    color: str          # hex colour code
    emoji: str          # 🟢 | 🟡 | 🔴
    risk:  float        # raw risk score

    def html(self, show_score: bool = False) -> str:
        """Render as an HTML <span> badge."""
        score_str = f" ({self.risk:.2f})" if show_score else ""
        return (
            f'<span style="display:inline-block; padding:0.22rem 0.6rem; '
            f'border-radius:4px; font-size:0.74rem; font-weight:700; '
            f'letter-spacing:0.04em; color:{self.color}; '
            f'background:{self.color}1A; border:1px solid {self.color}55;">'
            f'{self.emoji} {self.label}{score_str}</span>'
        )

    def __str__(self) -> str:
        return f"{self.emoji} {self.label}"


@dataclass
class StatusBadge:
    """Status badge for an atom decision."""
    label:  str   # SUPPORTED | VERIFIED | UNCERTAIN | ABSTAINED | NOT_VERIFIED
    color:  str   # hex colour code
    emoji:  str   # ✅ | ⚠️ | 🚫 | ❌

    def html(self) -> str:
        return (
            f'<span style="display:inline-block; padding:0.18rem 0.5rem; '
            f'border-radius:4px; font-size:0.72rem; font-weight:700; '
            f'letter-spacing:0.03em; color:{self.color}; '
            f'background:{self.color}1A; border:1px solid {self.color}55;">'
            f'{self.emoji} {self.label}</span>'
        )

    def __str__(self) -> str:
        return f"{self.emoji} {self.label}"


# ── Public API ────────────────────────────────────────────────────────────────

def risk_badge(
    risk: float,
    low_max:  float = _LOW_MAX,
    med_max:  float = _MED_MAX,
) -> RiskBadge:
    """
    Map a float risk score to a RiskBadge.

    Parameters
    ----------
    risk    : float in [0, 1]; values outside this range are clipped.
    low_max : scores <= this -> LOW.
    med_max : scores <= this -> MEDIUM (and > low_max).

    Returns
    -------
    RiskBadge with .label, .color, .emoji, and .html() method.
    """
    risk = max(0.0, min(1.0, float(risk)))

    if risk <= low_max:
        return RiskBadge(label="LOW",    color=_FOREST,   emoji="🟢", risk=risk)
    if risk <= med_max:
        return RiskBadge(label="MEDIUM", color=_AMBER,    emoji="🟡", risk=risk)
    return     RiskBadge(label="HIGH",   color=_BURGUNDY, emoji="🔴", risk=risk)


def status_badge(status: str) -> StatusBadge:
    """
    Map an atom decision status string to a StatusBadge.

    Recognised statuses: SUPPORTED, VERIFIED, UNCERTAIN, ABSTAINED, NOT_VERIFIED.
    Unknown statuses fall back to a neutral slate badge.
    """
    status_upper = (status or "").upper().strip()
    color = _STATUS_COLORS.get(status_upper, _SLATE)
    emoji = _STATUS_EMOJIS.get(status_upper, "ℹ️")
    return StatusBadge(label=status_upper, color=color, emoji=emoji)


def answer_status_label(answer_status: str) -> tuple[str, str, str]:
    """
    Map an answer-level status to (label, color, emoji).

    Used by the Streamlit UI to render the overall answer badge.
    """
    mapping = {
        "FULLY_SUPPORTED":    ("Fully Verified",      _FOREST,   "✅"),
        "PARTIALLY_VERIFIED": ("Partially Verified",  _AMBER,    "⚠️"),
        "ABSTAINED":          ("Cannot Be Verified",  _BURGUNDY, "🚫"),
    }
    return mapping.get(answer_status, (answer_status, _SLATE, "ℹ️"))
