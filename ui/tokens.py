"""Design tokens for the "editorial lab" look — the single source of truth.

Colors were sampled from the reference screenshots (canvas #F4F9FA, headline
ink #244248, section teal #4A838C, headline-period teal #60B0B3, chart teal
#7CB3BC, sand #C6B58E, navy/teal/gold bars from the core+sleeve slide). Two
deliberate departures keep text accessible (WCAG AA, 4.5:1): the screenshots'
mono labels and small grey text are far too light, so `LABEL` and `MUTED` are
darker than the originals.
"""

from __future__ import annotations

# ---- surfaces ---------------------------------------------------------------
CANVAS = "#F4F9FA"       # page background
CARD = "#FBFDFD"         # card surface
CARD_ALT = "#EEF5F6"     # inset / hover surface
HAIRLINE = "#CBD9DB"     # borders and rules
GRID = "#E1EBED"         # chart grid lines

# ---- ink --------------------------------------------------------------------
INK = "#244248"          # headlines, primary text
TEAL = "#4A838C"         # section titles, links, primary series
TEAL_DEEP = "#2F6B74"    # text-safe teal for small copy
ACCENT = "#60B0B3"       # the teal period after headlines, highlights (decorative)
BODY = "#3F5A60"         # body copy
MUTED = "#5A6E74"        # secondary copy (AA on CANVAS)
LABEL = "#5C7078"        # monospace eyebrow labels (AA on CANVAS)

# ---- data series ------------------------------------------------------------
SERIES_TEAL = "#7CB3BC"
SERIES_SAND = "#C6B58E"
SERIES_SLATE = "#8AA4B5"
NAVY = "#214162"
TEAL_BAR = "#459898"
GOLD = "#B18842"

# ---- status (kept distinct from brand teal) -----------------------------------
OK = "#2E7D5B"
WARN = "#9A6B12"
HALT = "#B23A3A"

SANS = "'Inter Tight', 'Helvetica Neue', Helvetica, Arial, sans-serif"
MONO = "'IBM Plex Mono', 'Courier Prime', Courier, monospace"
FONT_IMPORT = (
    "https://fonts.googleapis.com/css2?family=Inter+Tight:wght@400;500;600;700"
    "&family=IBM+Plex+Mono:wght@400;500&display=swap"
)

BRAND = "REGIME TRADER"
LAB = "REGIME LAB / SYSTEMATIC TRADING RESEARCH"


def _linear(channel: int) -> float:
    c = channel / 255
    return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4


def luminance(hex_color: str) -> float:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return 0.2126 * _linear(r) + 0.7152 * _linear(g) + 0.0722 * _linear(b)


def contrast_ratio(foreground: str, background: str) -> float:
    """WCAG 2.x contrast ratio between two #RRGGBB colors."""
    a, b = luminance(foreground), luminance(background)
    hi, lo = max(a, b), min(a, b)
    return (hi + 0.05) / (lo + 0.05)


# (foreground, background) pairs that carry real text and must stay >= 4.5:1.
TEXT_PAIRS: list[tuple[str, str, str]] = [
    ("INK/CANVAS", INK, CANVAS), ("INK/CARD", INK, CARD),
    ("BODY/CANVAS", BODY, CANVAS), ("BODY/CARD", BODY, CARD),
    ("MUTED/CANVAS", MUTED, CANVAS), ("MUTED/CARD", MUTED, CARD),
    ("LABEL/CANVAS", LABEL, CANVAS), ("LABEL/CARD", LABEL, CARD),
    ("TEAL_DEEP/CANVAS", TEAL_DEEP, CANVAS), ("TEAL_DEEP/CARD", TEAL_DEEP, CARD),
    ("OK/CARD", OK, CARD), ("WARN/CARD", WARN, CARD), ("HALT/CARD", HALT, CARD),
    ("INK/CARD_ALT", INK, CARD_ALT), ("MUTED/CARD_ALT", MUTED, CARD_ALT),
]
