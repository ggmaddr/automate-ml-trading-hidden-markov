"""Pure-Python SVG builders: gauge dials and line icons.

Everything is generated from numbers/short labels we control and escaped, so
the output is safe to embed with st.html.
"""

from __future__ import annotations

import base64
import math

from ui import tokens as t
from ui.html import esc


def as_img(svg_markup: str, alt: str, css_class: str = "", style: str = "") -> str:
    """Embed SVG as a CSS background on a role="img" div.

    Streamlit's HTML sanitizer strips inline <svg> and data-URI <img>, but keeps
    `style` attributes. An SVG used as a CSS image can never run script, and the
    aria-label keeps it accessible."""
    uri = "data:image/svg+xml;base64," + base64.b64encode(svg_markup.encode()).decode()
    cls = f' class="{esc(css_class)}"' if css_class else ""
    label = f' role="img" aria-label="{esc(alt)}"' if alt else ' aria-hidden="true"'
    return (
        f'<div{cls}{label} style="background-image:url(&quot;{uri}&quot;);background-repeat:no-repeat;'
        f'background-position:center;background-size:contain;{style}"></div>'
    )


_TONE_COLORS = {"ok": "#7FAEB6", "warn": "#C8A862", "halt": t.HALT, "neutral": "#96B7BE"}

# 24x24 line icons (stroke only), drawn to echo the thin-line illustrations in the reference style.
_ICONS = {
    "shield": '<path d="M12 3l7 3v5c0 4.5-3 8-7 10-4-2-7-5.5-7-10V6z"/><path d="M8.5 12l2.5 2.5L15.5 9.5"/>',
    "layers": '<path d="M12 3l9 5-9 5-9-5z"/><path d="M3 13l9 5 9-5"/><path d="M3 17.5l9 5 9-5"/>',
    "coin": '<circle cx="12" cy="12" r="9"/><path d="M9 8.5h4.2a2.3 2.3 0 010 4.6H10m0 0h3.7a2.3 2.3 0 010 4.6H9M12 6.5v2M12 17.7v2"/>',
    "gauge": '<path d="M4 17a8 8 0 1116 0"/><path d="M12 17l4-5"/>',
    "chart": '<path d="M4 19V5"/><path d="M4 19h16"/><path d="M8 15l3-4 3 2 4-6"/>',
    "bank": '<path d="M3 9l9-5 9 5"/><path d="M5 10v7M9.5 10v7M14.5 10v7M19 10v7"/><path d="M3 20h18"/>',
    "doc": '<path d="M7 3h7l4 4v14H7z"/><path d="M14 3v4h4M10 12h5M10 15h5M10 18h3"/>',
    "search": '<circle cx="10.5" cy="10.5" r="6"/><path d="M15 15l6 6M10.5 8v5M8 10.5h5"/>',
    "chip": '<rect x="7" y="7" width="10" height="10" rx="1"/><path d="M10 10h4v4h-4zM9 3v4M15 3v4M9 17v4M15 17v4M3 9h4M3 15h4M17 9h4M17 15h4"/>',
    "ledger": '<path d="M6 3h9l3 3v15H6z"/><path d="M9 9h6M9 13h6M9 17h4"/>',
}


def icon(name: str, size: int = 28, color: str = t.TEAL) -> str:
    body = _ICONS.get(name, "")
    markup = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
        f'stroke="{color}" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round">{body}</svg>'
    )
    return as_img(markup, "", "rt-icon", f"width:{size}px;height:{size}px;display:inline-block;")


def _point(cx: float, cy: float, r: float, angle: float) -> tuple[float, float]:
    return cx + r * math.cos(angle), cy - r * math.sin(angle)


def gauge_fraction(current: float, limit: float) -> float:
    """Share of the limit used, clamped to [0, 1]."""
    if not limit or limit <= 0:
        return 0.0
    return max(0.0, min(current / limit, 1.0))


def gauge_tone(fraction: float) -> str:
    """Same thresholds the dashboard's risk badges always used: <50% OK, <90% WARNING, else HALT."""
    return "ok" if fraction < 0.5 else "warn" if fraction < 0.9 else "halt"


def gauge(fraction: float, title: str, value_text: str, caption: str, icon_name: str = "gauge",
          tone: str | None = None) -> str:
    """Half-circle dial. The needle and filled arc show `fraction` of the limit used."""
    fraction = max(0.0, min(float(fraction), 1.0))
    tone = tone or gauge_tone(fraction)
    color = _TONE_COLORS.get(tone, _TONE_COLORS["neutral"])
    cx, cy, r, stroke = 130.0, 124.0, 92.0, 20

    def arc(a0: float, a1: float) -> str:
        x0, y0 = _point(cx, cy, r, a0)
        x1, y1 = _point(cx, cy, r, a1)
        return f"M {x0:.2f} {y0:.2f} A {r} {r} 0 0 1 {x1:.2f} {y1:.2f}"

    end_angle = math.pi * (1 - fraction)
    ticks = []
    for i in range(0, 11):
        a = math.pi * (1 - i / 10)
        x0, y0 = _point(cx, cy, r + stroke / 2 + 3, a)
        x1, y1 = _point(cx, cy, r + stroke / 2 + (9 if i % 5 == 0 else 6), a)
        ticks.append(f'<line x1="{x0:.1f}" y1="{y0:.1f}" x2="{x1:.1f}" y2="{y1:.1f}" stroke="{t.HAIRLINE}" stroke-width="1.5"/>')
    nx, ny = _point(cx, cy, r - 18, end_angle)
    filled = f'<path d="{arc(math.pi, end_angle)}" stroke="{color}" stroke-width="{stroke}" fill="none"/>' if fraction > 0.004 else ""
    label = f"{title}: {value_text} ({caption}), {fraction:.0%} of limit used"
    dial = (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 260 150">'
        f'<path d="{arc(math.pi, 0)}" stroke="#E1EBED" stroke-width="{stroke}" fill="none"/>'
        f'{filled}{"".join(ticks)}'
        f'<line x1="{cx}" y1="{cy}" x2="{nx:.2f}" y2="{ny:.2f}" stroke="{t.TEAL_DEEP}" stroke-width="2.6" stroke-linecap="round"/>'
        f'<circle cx="{cx}" cy="{cy}" r="8" fill="#6F8FA3"/></svg>'
    )
    return (
        f'<div class="rt-gauge">{as_img(dial, label, "rt-gauge-dial")}'
        f'<div class="rt-gauge-icon">{icon(icon_name, 26)}</div>'
        f'<div class="rt-gauge-title">{esc(title)}</div>'
        f'<div class="rt-gauge-value rt-{tone}">{esc(value_text)}</div>'
        f'<div class="rt-gauge-caption">{esc(caption)}</div></div>'
    )


def gauge_row(items: list[str]) -> str:
    return '<div class="rt-gauge-row">' + "".join(items) + "</div>"
