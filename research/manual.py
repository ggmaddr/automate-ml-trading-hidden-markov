"""No-API workflow: use Google Finance Deep Search by hand, keep everything else local.

Nothing here calls Gemini or touches Google Finance. `build_deep_search_prompt`
produces text for YOU to paste into Google Finance; `save_manual_report`
stores the report you paste back, next to the HMM snapshot, as a briefing
file. Both are read-only with respect to trading.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

from research.config import PROJECT_ROOT
from research.prompts import default_question, summarize_quant_line
from research.schemas import QuantContext

REFERENCE_DIR = PROJECT_ROOT / "evaluation" / "google_finance_reference"
_URL_RE = re.compile(r"https?://[^\s)\]>\"']+")


def build_deep_search_prompt(symbol: str, question: Optional[str], lookback_days: int) -> str:
    q = question or default_question(symbol, lookback_days)
    return (
        f"{q}\n\n"
        "Please include: dates for each development, the source for each claim, which items are reported fact "
        "versus interpretation, any conflicting reports, and what you could not find. "
        "Do not give buy/sell/hold advice."
    )


def extract_domains(text: str) -> list[str]:
    seen: list[str] = []
    for url in _URL_RE.findall(text):
        host = (urlparse(url.rstrip(".,")).hostname or "")
        host = host[4:] if host.startswith("www.") else host
        if host and host not in seen:
            seen.append(host)
    return seen


def save_manual_report(
    symbol: str, question: str, report: str, quant: Optional[QuantContext],
    directory: Path = REFERENCE_DIR, now: Optional[datetime] = None,
) -> Path:
    """Write `<SYMBOL>_<UTC timestamp>.md`: question, HMM snapshot line, then the pasted report."""
    now = now or datetime.now(timezone.utc)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{symbol.upper()}_{now:%Y%m%d_%H%M%S}.md"
    quant_line = summarize_quant_line(quant) if quant is not None else "(HMM snapshot unavailable)"
    path.write_text(
        f"# {symbol.upper()} — Google Finance Deep Search (manual)\n\n"
        f"Saved: {now:%Y-%m-%d %H:%M} UTC\n\n"
        f"## Question\n{question}\n\n"
        f"## HMM quant snapshot (independent of the report below)\n{quant_line}\n\n"
        f"## Pasted report\n{report.strip()}\n"
    )
    return path
