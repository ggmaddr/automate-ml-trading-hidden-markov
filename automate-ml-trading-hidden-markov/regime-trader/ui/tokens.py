"""Design tokens for the "editorial lab" look — the single source of truth.

Colorway 3: the project's own brand colors — blue #2973B2, aqua #0CC0DF, and
a light-yellow/cream background #F3F2EC. Variable names are unchanged from
earlier colorways so every other file (ui/theme.py, ui/charts.py, ui/svg.py,
ui/widgets/) needs no edits — only the hex values moved. One deliberate
departure keeps text accessible (WCAG AA, 4.5:1): the brand blue itself is a
hair under 4.5:1 on the cream canvas, so `TEAL_DEEP` (used for small link
text) is a slightly deepened shade of it; `TEAL` keeps the exact brand blue
for headings/large text/chart lines, where that's not required.
"""

from __future__ import annotations

# ---- surfaces ---------------------------------------------------------------
CANVAS = "#F3F2EC"       # page background (brand light yellow/cream)
CARD = "#FBFAF6"         # card surface
CARD_ALT = "#ECEAE0"     # inset / hover surface
HAIRLINE = "#DAD6CA"     # borders and rules
GRID = "#E6E2D5"         # chart grid lines

# ---- ink --------------------------------------------------------------------
INK = "#1C3350"          # headlines, primary text (deep navy)
TEAL = "#2973B2"         # section titles, links, primary series (brand blue)
TEAL_DEEP = "#235F8E"    # text-safe shade of the brand blue, for small copy
ACCENT = "#0CC0DF"       # the period after headlines, highlights (decorative, brand aqua)
BODY = "#30465F"         # body copy
MUTED = "#5B6B7C"        # secondary copy (AA on CANVAS)
LABEL = "#5E6F82"        # monospace eyebrow labels (AA on CANVAS)

# ---- data series ------------------------------------------------------------
SERIES_TEAL = "#0CC0DF"  # brand aqua
SERIES_SAND = "#EC635E"  # coral (named _SAND for the unaffected callers; now the coral accent)
SERIES_SLATE = "#6FA8D6" # light steel blue
NAVY = "#16304C"
TEAL_BAR = "#2973B2"
GOLD = "#D9A646"         # warm amber, used only as a 6th/7th palette slot in multi-series charts

# ---- status (kept distinct from the brand blue) -----------------------------------
OK = "#2E7D5B"
WARN = "#946610"
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
