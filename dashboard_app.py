"""Streamlit UI for regime-trader.

    streamlit run dashboard_app.py        # run from inside regime-trader/

Pages (views/): Live (polls `dashboard_state.json`), Backtest (reads
`backtest_results/*.csv`), AI Research (Phase 10, read-only) and Methodology.
Styling lives in ui/ (tokens, CSS, escaped HTML builders, SVG gauges, charts)
and .streamlit/config.toml; nothing here recomputes financial math.
"""

from __future__ import annotations

import streamlit as st
from dotenv import load_dotenv

from ui import html as h
from ui import theme, tokens
from views import backtest, live, methodology, research

load_dotenv()  # API keys for the AI Research view

st.set_page_config(page_title="Regime Trader", page_icon=None, layout="wide", initial_sidebar_state="collapsed")
theme.inject()

PAGES = [
    st.Page(live.render, title="Live", default=True),
    st.Page(backtest.render, title="Backtest", url_path="backtest"),
    st.Page(research.render, title="AI Research", url_path="research"),
    st.Page(methodology.render, title="Methodology", url_path="methodology"),
]
current = st.navigation(PAGES, position="hidden")

brand, nav = st.columns([2, 3], vertical_alignment="bottom")
with brand:
    st.html(h.eyebrow(tokens.LAB) + f'<div class="rt-brand">{h.esc(tokens.BRAND)}<span class="rt-dot">.</span></div>')
with nav:
    with st.container(key="rt-nav"):
        links = st.columns([1, 1, 1.35, 1.35])
        for col, page in zip(links, PAGES):
            with col:
                st.page_link(page, label=page.title)
    # Streamlit gives the active page link no stable attribute, so mark it by position.
    active = next((i for i, page in enumerate(PAGES) if page.title == current.title), 0)
    st.html(theme.nav_active_css(active))
st.html('<hr class="rt-rule">')

current.run()
