"""CSS skin for the app. All overrides live here (one file) and target only
Streamlit's stable `data-testid` hooks plus our own `rt-*` classes, so a
Streamlit upgrade is a one-file fix."""

from __future__ import annotations

import streamlit as st

from ui import tokens as t


def css() -> str:
    return f"""
@import url('{t.FONT_IMPORT}');

:root {{
  --rt-canvas: {t.CANVAS}; --rt-card: {t.CARD}; --rt-card-alt: {t.CARD_ALT}; --rt-hair: {t.HAIRLINE};
  --rt-ink: {t.INK}; --rt-teal: {t.TEAL}; --rt-teal-deep: {t.TEAL_DEEP}; --rt-accent: {t.ACCENT};
  --rt-body: {t.BODY}; --rt-muted: {t.MUTED}; --rt-label: {t.LABEL};
  --rt-ok: {t.OK}; --rt-warn: {t.WARN}; --rt-halt: {t.HALT};
  --rt-sans: {t.SANS}; --rt-mono: {t.MONO};
}}

/* ---- page chrome ------------------------------------------------------- */
.stApp {{ background: var(--rt-canvas); color: var(--rt-body); font-family: var(--rt-sans); }}
[data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"],
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"], footer {{ display: none !important; }}
[data-testid="stMainBlockContainer"] {{ max-width: 1320px; padding: 1.4rem 2.2rem 4rem; }}
h1, h2, h3, h4 {{ font-family: var(--rt-sans); color: var(--rt-ink); letter-spacing: -0.02em; }}
p, li, label, .stMarkdown {{ font-family: var(--rt-sans); }}
a {{ color: var(--rt-teal-deep); }}

/* ---- editorial primitives ------------------------------------------------ */
.rt-eyebrow {{ font-family: var(--rt-mono); font-size: 0.74rem; letter-spacing: 0.2em; text-transform: uppercase;
  color: var(--rt-label); margin: 0 0 0.35rem; }}
.rt-label {{ font-family: var(--rt-mono); font-size: 0.7rem; letter-spacing: 0.16em; text-transform: uppercase; color: var(--rt-label); }}
.rt-mono-muted {{ font-family: var(--rt-mono); font-size: 0.72rem; color: var(--rt-muted); }}
.rt-hero {{ font-family: var(--rt-sans); font-weight: 600; font-size: clamp(2.6rem, 6.2vw, 5.2rem); line-height: 1.0;
  letter-spacing: -0.035em; color: var(--rt-ink); margin: 0.1rem 0 1.2rem; padding: 0; }}
.rt-dot {{ color: var(--rt-accent); }}
.rt-section {{ font-weight: 500; font-size: clamp(1.4rem, 2.4vw, 2rem); letter-spacing: -0.03em; color: var(--rt-teal);
  margin: 2.2rem 0 0.8rem; padding: 0; }}
.rt-lead {{ font-size: 1.05rem; line-height: 1.55; color: var(--rt-muted); max-width: 62ch; margin: 0 0 1.2rem; }}

/* brand header */
.rt-brand {{ font-family: var(--rt-sans); font-weight: 600; font-size: clamp(2rem, 3.6vw, 3rem); letter-spacing: -0.04em;
  line-height: 1; color: var(--rt-ink); margin: 0.2rem 0 0.4rem; }}
.rt-rule {{ height: 1px; background: var(--rt-hair); margin: 0.6rem 0 1.4rem; border: 0; }}

/* nav: st.page_link rendered as mono text links */
[data-testid="stPageLink-NavLink"] {{ background: transparent !important; border-radius: 0; padding: 0.35rem 0.2rem;
  border-bottom: 2px solid transparent; justify-content: flex-end; text-decoration: none; }}
[data-testid="stPageLink-NavLink"] p {{ font-family: var(--rt-mono) !important; font-size: 0.95rem; color: var(--rt-muted);
  letter-spacing: 0.01em; margin: 0; }}
[data-testid="stPageLink-NavLink"]:hover p {{ color: var(--rt-teal-deep); }}
[data-testid="stPageLink-NavLink"]:focus-visible {{ outline: 2px solid var(--rt-teal); outline-offset: 2px; }}

/* ---- cards ------------------------------------------------------------------ */
.rt-card {{ background: var(--rt-card); border: 1px solid var(--rt-hair); border-radius: 6px; padding: 1.05rem 1.2rem;
  transition: border-color .18s ease, box-shadow .18s ease; }}
.rt-card:hover {{ border-color: var(--rt-accent); box-shadow: 0 1px 0 rgba(74,131,140,.12); }}
.rt-kpi-row {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 0.8rem; margin: 0.4rem 0 1.2rem; }}
.rt-kpi-value {{ font-family: var(--rt-sans); font-weight: 500; font-size: clamp(1.6rem, 2.6vw, 2.35rem); letter-spacing: -0.04em;
  color: var(--rt-teal); line-height: 1.1; margin-top: 0.35rem; }}
.rt-kpi-sub {{ font-family: var(--rt-mono); font-size: 0.74rem; color: var(--rt-muted); margin-top: 0.3rem; }}
.rt-ok {{ color: var(--rt-ok); }} .rt-warn {{ color: var(--rt-warn); }} .rt-halt {{ color: var(--rt-halt); }}

.rt-pill {{ display: inline-block; font-family: var(--rt-mono); font-size: 0.7rem; letter-spacing: 0.08em; text-transform: uppercase;
  padding: 0.12rem 0.5rem; border: 1px solid var(--rt-hair); border-radius: 999px; background: var(--rt-card-alt); color: var(--rt-muted);
  margin-right: 0.4rem; }}
.rt-pill.rt-ok {{ color: var(--rt-ok); border-color: currentColor; }} .rt-pill.rt-warn {{ color: var(--rt-warn); border-color: currentColor; }}
.rt-pill.rt-halt {{ color: var(--rt-halt); border-color: currentColor; }} .rt-pill.rt-accent {{ color: var(--rt-teal-deep); }}

.rt-evidence {{ background: var(--rt-card); border: 1px solid var(--rt-hair); border-left: 3px solid var(--rt-accent);
  border-radius: 4px; padding: 0.7rem 0.9rem; margin: 0.5rem 0; }}
.rt-evidence-meta {{ margin-bottom: 0.35rem; }} .rt-evidence-claim {{ color: var(--rt-ink); line-height: 1.45; margin-bottom: 0.35rem; }}
.rt-chip {{ font-family: var(--rt-mono); font-size: 0.76rem; color: var(--rt-teal-deep); text-decoration: underline;
  text-underline-offset: 3px; }}
.rt-chip:focus-visible {{ outline: 2px solid var(--rt-teal); outline-offset: 2px; }}
.rt-chip-date {{ font-family: var(--rt-mono); font-size: 0.72rem; color: var(--rt-muted); margin-left: 0.4rem; }}

/* ---- native widgets, skinned ----------------------------------------------- */
[data-testid="stMetric"] {{ background: var(--rt-card); border: 1px solid var(--rt-hair); border-radius: 6px; padding: 0.8rem 1rem; }}
[data-testid="stMetricLabel"] p {{ font-family: var(--rt-mono); font-size: 0.7rem; letter-spacing: 0.14em; text-transform: uppercase; color: var(--rt-label); }}
[data-testid="stMetricValue"] {{ font-family: var(--rt-sans); font-weight: 500; letter-spacing: -0.03em; color: var(--rt-teal); }}
[data-testid="stDataFrame"] {{ border: 1px solid var(--rt-hair); border-radius: 6px; }}
[data-testid="stExpander"] {{ border: 1px solid var(--rt-hair); border-radius: 6px; background: var(--rt-card); }}
[data-testid="stForm"] {{ background: var(--rt-card); border: 1px solid var(--rt-hair); border-radius: 6px; }}
[data-testid="stAlert"] {{ border-radius: 6px; border: 1px solid var(--rt-hair); background: var(--rt-card-alt); color: var(--rt-body); }}
.stButton > button, [data-testid="stFormSubmitButton"] > button {{ font-family: var(--rt-mono); letter-spacing: 0.04em;
  border: 1px solid var(--rt-teal); color: var(--rt-teal-deep); background: var(--rt-card); border-radius: 4px; transition: background .18s ease, color .18s ease; cursor: pointer; }}
.stButton > button:hover, [data-testid="stFormSubmitButton"] > button:hover {{ background: var(--rt-teal); color: #fff; border-color: var(--rt-teal); }}
.stButton > button:focus-visible, [data-testid="stFormSubmitButton"] > button:focus-visible {{ outline: 2px solid var(--rt-teal); outline-offset: 2px; }}
[data-testid="stCaptionContainer"], .stCaption {{ font-family: var(--rt-mono); color: var(--rt-muted); }}

/* ---- notices, tiles, flow, correlation grid ------------------------------------ */
.rt-notice {{ background: var(--rt-card-alt); border: 1px solid var(--rt-hair); border-left: 3px solid var(--rt-accent); border-radius: 4px;
  padding: 0.75rem 1rem; color: var(--rt-body); margin: 0.5rem 0 1rem; line-height: 1.5; }}
.rt-notice ul {{ margin: 0.3rem 0 0; padding-left: 1.1rem; }} .rt-notice li {{ margin: 0.15rem 0; }}
.rt-notice-title {{ font-family: var(--rt-mono); font-size: 0.72rem; letter-spacing: 0.14em; text-transform: uppercase; color: var(--rt-label); margin-bottom: 0.3rem; }}
.rt-notice-warn {{ border-left-color: #C8A862; }} .rt-notice-halt {{ border-left-color: var(--rt-halt); }} .rt-notice-ok {{ border-left-color: var(--rt-ok); }}
.rt-list {{ margin: 0.2rem 0 0.8rem; padding-left: 1.1rem; color: var(--rt-body); line-height: 1.5; }}
.rt-tile-row {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 1rem; margin: 0.6rem 0 1.2rem; }}
.rt-tile {{ background: var(--rt-card); border: 1px solid var(--rt-hair); border-radius: 6px; padding: 1.1rem 1.2rem; text-align: left; }}
.rt-tile-icon {{ margin-bottom: 0.7rem; }} .rt-tile-icon .rt-icon {{ width: 44px !important; height: 44px !important; }}
.rt-tile-title {{ font-weight: 500; font-size: 1.35rem; letter-spacing: -0.03em; color: var(--rt-teal); }}
.rt-tile-sub {{ color: var(--rt-body); margin-top: 0.25rem; }} .rt-tile-note {{ font-family: var(--rt-mono); font-size: 0.74rem; color: var(--rt-muted); margin-top: 0.3rem; }}
.rt-flow {{ display: flex; align-items: stretch; gap: 0.6rem; flex-wrap: wrap; margin: 0.8rem 0 1.4rem; }}
.rt-flow > .rt-tile {{ flex: 1 1 200px; }}
.rt-arrow {{ flex: 0 0 38px; align-self: center; height: 0; border-top: 2px dotted var(--rt-teal); position: relative; }}
.rt-arrow::after {{ content: ""; position: absolute; right: -2px; top: -6px; border-left: 8px solid var(--rt-teal); border-top: 5px solid transparent; border-bottom: 5px solid transparent; }}
@media (max-width: 760px) {{ .rt-arrow {{ display: none; }} }}
.rt-corr {{ display: grid; gap: 4px; align-items: center; margin-top: 0.5rem; }}
.rt-corr-head {{ font-family: var(--rt-mono); font-size: 0.62rem; overflow: hidden; text-overflow: ellipsis; letter-spacing: 0.12em; text-transform: uppercase; color: var(--rt-label); text-align: center; }}
.rt-corr-cell {{ font-family: var(--rt-mono); font-size: 0.82rem; color: var(--rt-ink); text-align: center; padding: 0.7rem 0.2rem; border-radius: 3px; border: 1px solid #fff; }}
[class*="st-key-rtcard"] {{ background: var(--rt-card); border: 1px solid var(--rt-hair); border-radius: 6px; padding: 1rem 1.1rem 0.4rem; }}
.rt-bignum {{ font-weight: 500; font-size: clamp(2rem, 3.4vw, 3rem); letter-spacing: -0.04em; color: var(--rt-teal); line-height: 1.05; }}
.rt-bignum small {{ font-family: var(--rt-mono); font-size: 0.8rem; color: var(--rt-muted); letter-spacing: 0; margin-left: 0.4rem; }}

/* ---- gauges ------------------------------------------------------------------ */
.rt-gauge-row {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 1.2rem; margin: 0.6rem 0 1.4rem; }}
.rt-gauge {{ text-align: center; padding: 0.4rem 0.2rem; }}
.rt-gauge-dial {{ width: 100%; max-width: 270px; aspect-ratio: 260 / 150; margin: 0 auto; }}
.rt-gauge-icon {{ margin-top: -0.2rem; opacity: .9; }}
.rt-gauge-title {{ font-family: var(--rt-sans); font-weight: 500; font-size: 1.45rem; letter-spacing: -0.03em; color: var(--rt-teal); margin-top: 0.2rem; }}
.rt-gauge-value {{ font-family: var(--rt-mono); font-size: 0.9rem; color: var(--rt-ink); }}
.rt-gauge-caption {{ font-family: var(--rt-mono); font-size: 0.74rem; color: var(--rt-muted); margin-top: 0.1rem; }}
.rt-icon {{ vertical-align: middle; }}

/* ---- responsive + motion ---------------------------------------------------- */
@media (max-width: 760px) {{
  .st-key-rt-nav [data-testid="stHorizontalBlock"] {{ flex-wrap: wrap; gap: 0.2rem 1rem; }}
  .st-key-rt-nav [data-testid="stColumn"] {{ min-width: 0 !important; flex: 0 0 auto !important; width: auto !important; }}
  [data-testid="stMainBlockContainer"] {{ padding: 1rem 1rem 3rem; }}
  .rt-kpi-row {{ grid-template-columns: repeat(2, 1fr); }}
}}
@media (prefers-reduced-motion: reduce) {{
  *, *::before, *::after {{ transition: none !important; animation: none !important; }}
}}
"""


def nav_active_css(index: int) -> str:
    """Underline + teal text for the active nav link (1-based nth-child of the keyed nav container)."""
    sel = f'.st-key-rt-nav [data-testid="stColumn"]:nth-child({index + 1}) [data-testid="stPageLink-NavLink"]'
    return f"<style>{sel} {{ border-bottom-color: var(--rt-teal) !important; }} {sel} p {{ color: var(--rt-teal-deep) !important; font-weight: 500; }}</style>"


def inject() -> None:
    """Call once per run, before any content."""
    st.html(f"<style>{css()}</style>")
