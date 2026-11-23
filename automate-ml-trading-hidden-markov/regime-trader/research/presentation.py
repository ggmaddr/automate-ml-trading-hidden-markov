"""Pure formatting helpers shared by the CLI and the Streamlit page."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from research.schemas import Evidence, QuantContext

RELATIONSHIP_LABELS = {
    "supportive": "Supportive",
    "partially_supportive": "Partially supportive",
    "conflicting": "Conflicting",
    "orthogonal": "Orthogonal (different dimensions)",
    "insufficient_evidence": "Insufficient evidence",
}

STANCE_LABELS = {
    "positive": "Positive evidence",
    "negative": "Negative evidence",
    "mixed": "Mixed evidence",
    "insufficient_evidence": "Insufficient evidence",
}


def age_label(generated_at: datetime, now: Optional[datetime] = None) -> str:
    now = now or datetime.now(timezone.utc)
    if generated_at.tzinfo is None:
        generated_at = generated_at.replace(tzinfo=timezone.utc)
    seconds = max(int((now - generated_at).total_seconds()), 0)
    if seconds < 90:
        return "just now"
    if seconds < 5400:
        return f"{seconds // 60} min ago"
    if seconds < 172800:
        return f"{seconds // 3600} h ago"
    return f"{seconds // 86400} d ago"


def evidence_markdown(e: Evidence) -> str:
    """One bullet with a visible, clickable source link attached to the claim."""
    date = f" · {e.published_at:%Y-%m-%d}" if e.published_at else ""
    domain = f" ({e.source_domain})" if e.source_domain and e.source_domain not in e.source_title else ""
    return (
        f"- **{e.category}** · _{e.impact}_ · conf {e.confidence:.0%} — {e.claim} "
        f"[{e.source_title}{domain}]({e.source_url}){date}"
    )


def quant_rows(q: QuantContext) -> list[tuple[str, str]]:
    def pct(v: Optional[float], d: int = 1) -> str:
        return "n/a" if v is None else f"{v * 100:.{d}f}%"

    return [
        ("Symbol", q.symbol),
        ("HMM regime", f"{q.regime_label} (state {q.regime_id})" if q.regime_id is not None else "n/a"),
        ("Confidence", pct(q.regime_probability, 0)),
        ("Expected volatility (per bar)", pct(q.expected_volatility, 2)),
        ("Volatility rank (0 calm – 1 turbulent)", "n/a" if q.volatility_rank is None else f"{q.volatility_rank:.2f}"),
        ("Stability", f"{'confirmed' if q.is_confirmed else 'UNCONFIRMED'}, {q.consecutive_bars} bars"),
        ("Flicker", f"{'FLICKERING' if q.is_flickering else 'stable'} ({'n/a' if q.flicker_rate is None else f'{q.flicker_rate:.0f}'} flips in window)"),
        ("Model version", q.model_version or "n/a"),
        ("Data as of", f"{q.timestamp:%Y-%m-%d}"),
    ]
