"""Escape-by-default HTML builders for the UI.

Everything dynamic (model output, search-result titles, URLs, tickers) is
UNTRUSTED. These helpers HTML-escape every interpolated string and only allow
http(s) links, so research text can never inject markup or script.
"""

from __future__ import annotations

import html as _html
from typing import Iterable, Optional
from urllib.parse import urlparse

TONES = {"ok", "warn", "halt", "neutral", "accent"}


def esc(value: object) -> str:
    return _html.escape("" if value is None else str(value), quote=True)


def safe_url(url: object) -> str:
    """The URL if it is plain http(s), else '#'."""
    text = str(url or "").strip()
    parsed = urlparse(text)
    if parsed.scheme in ("http", "https") and parsed.netloc and not any(c in text for c in ' "<>\n\r\t'):
        return _html.escape(text, quote=True)
    return "#"


def eyebrow(text: str) -> str:
    return f'<div class="rt-eyebrow">{esc(text)}</div>'


def hero(title: str, accent: str = ".") -> str:
    """Large tight headline ending in the accent-colored period."""
    return f'<h1 class="rt-hero">{esc(title)}<span class="rt-dot">{esc(accent)}</span></h1>'


def section_title(text: str) -> str:
    return f'<h2 class="rt-section">{esc(text)}</h2>'


def lead(text: str) -> str:
    return f'<p class="rt-lead">{esc(text)}</p>'


def pill(text: str, tone: str = "neutral") -> str:
    tone = tone if tone in TONES else "neutral"
    return f'<span class="rt-pill rt-{tone}">{esc(text)}</span>'


def kpi(label: str, value: object, sub: Optional[object] = None, tone: Optional[str] = None) -> str:
    tone_cls = f" rt-{tone}" if tone in TONES else ""
    sub_html = f'<div class="rt-kpi-sub{tone_cls}">{esc(sub)}</div>' if sub not in (None, "") else ""
    return (
        '<div class="rt-card rt-kpi">'
        f'<div class="rt-label">{esc(label)}</div>'
        f'<div class="rt-kpi-value">{esc(value)}</div>{sub_html}</div>'
    )


def kpi_row(items: Iterable[str], min_width: int = 170) -> str:
    return f'<div class="rt-kpi-row" style="grid-template-columns: repeat(auto-fit, minmax({int(min_width)}px, 1fr))">' + "".join(items) + "</div>"


def card(inner_html: str, title: Optional[str] = None) -> str:
    """`inner_html` must already be built from these helpers (escaped)."""
    head = f'<div class="rt-label">{esc(title)}</div>' if title else ""
    return f'<div class="rt-card">{head}{inner_html}</div>'


def source_chip(title: str, url: str, date: Optional[str] = None) -> str:
    when = f' <span class="rt-chip-date">{esc(date)}</span>' if date else ""
    return f'<a class="rt-chip" href="{safe_url(url)}" target="_blank" rel="noopener noreferrer nofollow">{esc(title)}</a>{when}'


def evidence_item(claim: str, category: str, impact: str, confidence: float, title: str, url: str,
                  date: Optional[str] = None) -> str:
    tone = {"positive": "ok", "negative": "halt", "mixed": "warn"}.get(impact, "neutral")
    return (
        '<div class="rt-evidence">'
        f'<div class="rt-evidence-meta">{pill(category, "accent")}{pill(impact, tone)}'
        f'<span class="rt-mono-muted">conf {confidence:.0%}</span></div>'
        f'<div class="rt-evidence-claim">{esc(claim)}</div>'
        f'<div>{source_chip(title, url, date)}</div></div>'
    )


def notice(text: str, tone: str = "neutral", title: Optional[str] = None) -> str:
    """Quiet inline message (replaces Streamlit's blue/yellow alert boxes)."""
    tone = tone if tone in TONES else "neutral"
    head = f'<div class="rt-notice-title">{esc(title)}</div>' if title else ""
    return f'<div class="rt-notice rt-notice-{tone}">{head}<div>{esc(text)}</div></div>'


def notice_list(lines: Iterable[str], tone: str = "neutral", title: Optional[str] = None) -> str:
    head = f'<div class="rt-notice-title">{esc(title)}</div>' if title else ""
    items = "".join(f"<li>{esc(line)}</li>" for line in lines)
    tone = tone if tone in TONES else "neutral"
    return f'<div class="rt-notice rt-notice-{tone}">{head}<ul>{items}</ul></div>'


def bullets(items: Iterable[str]) -> str:
    return "<ul class=\"rt-list\">" + "".join(f"<li>{esc(i)}</li>" for i in items) + "</ul>"


def tile(icon_html: str, title: str, sub: str = "", note: str = "") -> str:
    """Illustrated tile (icon + title + two quiet lines), like the reference's data tiles. `icon_html` is built by ui.svg."""
    sub_html = f'<div class="rt-tile-sub">{esc(sub)}</div>' if sub else ""
    note_html = f'<div class="rt-tile-note">{esc(note)}</div>' if note else ""
    return f'<div class="rt-tile"><div class="rt-tile-icon">{icon_html}</div><div class="rt-tile-title">{esc(title)}</div>{sub_html}{note_html}</div>'


def tile_row(tiles: Iterable[str]) -> str:
    return '<div class="rt-tile-row">' + "".join(tiles) + "</div>"


def flow(steps: Iterable[str]) -> str:
    """Steps (each already-built tile HTML) joined by dotted arrows."""
    return '<div class="rt-flow">' + '<div class="rt-arrow" aria-hidden="true"></div>'.join(steps) + "</div>"
