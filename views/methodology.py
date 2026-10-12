"""Methodology page: how the system works, with every number read from config/settings.yaml."""

from __future__ import annotations

import streamlit as st

from data.feature_engineering import RAW_FEATURE_COLUMNS
from ui import html as h
from ui import svg
from views.common import load_settings


def _pct(value: float, digits: int = 0) -> str:
    return f"{value:.{digits}%}"


def render() -> None:
    s = load_settings()
    hmm, strat, risk = s.get("hmm", {}), s.get("strategy", {}), s.get("risk", {})

    st.html(h.eyebrow("04 / Methodology") + h.hero("How it works"))
    st.html(h.lead(
        "A volatility-regime classifier drives position sizing. It does not predict price direction, and a deterministic "
        "risk manager has the final say on every order. Numbers on this page are read live from config/settings.yaml."
    ))

    # ---- 01 data -------------------------------------------------------------
    st.html(h.eyebrow("01 / Data") + h.section_title("Bars + features."))
    st.html(h.tile_row([
        h.tile(svg.icon("chart", 44), "Daily bars", "OHLCV", "yfinance for research and backtests; the broker feed for paper trading"),
        h.tile(svg.icon("chip", 44), f"{len(RAW_FEATURE_COLUMNS)} features", "Volatility, trend, momentum, volume",
               "each standardized with a rolling 252-bar z-score"),
        h.tile(svg.icon("ledger", 44), "Saved model", "hmm_model.pkl", "retrained weekly; inference never trains"),
    ]))

    # ---- 02 regimes -------------------------------------------------------------
    st.html(h.eyebrow("02 / Regimes") + h.section_title("Filtered, never smoothed."))
    candidates = ", ".join(str(n) for n in hmm.get("n_candidates", []))
    st.html(h.tile_row([
        h.tile(svg.icon("gauge", 44), "Model selection", f"BIC over {candidates} regimes", f"{hmm.get('n_init', '?')} random restarts each"),
        h.tile(svg.icon("search", 44), "Forward filter", "state at t uses data up to t",
               "no look-ahead; a test proves earlier states never change"),
        h.tile(svg.icon("layers", 44), "Stability", f"{hmm.get('stability_bars', '?')} bars to confirm a change",
               f"flicker: more than {hmm.get('flicker_threshold', '?')} flips in {hmm.get('flicker_window', '?')} bars"),
    ]))
    st.html(h.notice(
        f"Labels such as BULL or BEAR only order regimes by historical return for readability. Strategy behavior follows each "
        f"regime's volatility. A call below {_pct(hmm.get('min_confidence', 0.55))} confidence is not acted on.",
        title="Read the labels carefully",
    ))

    # ---- 03 allocation ----------------------------------------------------------------
    st.html(h.eyebrow("03 / Strategy") + h.section_title("Allocation follows volatility."))
    st.html(h.tile_row([
        h.tile(svg.icon("layers", 44), _pct(strat.get("low_vol_allocation", 0)), "Low volatility",
               f"up to {strat.get('low_vol_leverage', 1):.2f}x leverage"),
        h.tile(svg.icon("layers", 44), f"{_pct(strat.get('mid_vol_allocation_trend', 0))} / {_pct(strat.get('mid_vol_allocation_no_trend', 0))}",
               "Mid volatility", "with trend / without trend"),
        h.tile(svg.icon("shield", 44), _pct(strat.get("high_vol_allocation", 0)), "High volatility", "defensive sizing"),
    ]))

    # ---- 04 limits (gauges) --------------------------------------------------------------
    st.html(h.eyebrow("04 / Risk, sizing, leverage") + h.section_title("Choose your limits."))
    per_trade, exposure, leverage = risk.get("max_risk_per_trade", 0.01), risk.get("max_exposure", 0.8), risk.get("max_leverage", 1.25)
    # Dial scales (fixed ceilings so the needle position is comparable): 5% per trade, 100% exposure, 2.0x leverage.
    st.html(svg.gauge_row([
        svg.gauge(svg.gauge_fraction(per_trade, 0.05), "One trade", _pct(per_trade, 1), "maximum loss", "shield", tone="neutral"),
        svg.gauge(svg.gauge_fraction(exposure, 1.0), "Total exposure", _pct(exposure), "maximum invested", "layers", tone="neutral"),
        svg.gauge(svg.gauge_fraction(leverage, 2.0), "Borrowing", f"{leverage:.2f}x", "leverage ceiling", "coin", tone="neutral"),
    ]))
    st.html(h.bullets([
        f"Circuit breakers: reduce at {_pct(risk.get('daily_dd_reduce', 0), 1)} daily drawdown, halt at {_pct(risk.get('daily_dd_halt', 0), 1)}; "
        f"weekly {_pct(risk.get('weekly_dd_reduce', 0), 1)} / {_pct(risk.get('weekly_dd_halt', 0), 1)}; full halt at {_pct(risk.get('max_dd_from_peak', 0), 1)} from peak.",
        f"At most {risk.get('max_concurrent', '?')} open positions, {_pct(risk.get('max_single_position', 0))} of equity in any one symbol, "
        f"{risk.get('max_daily_trades', '?')} trades a day.",
        "Every signal passes through the risk manager, which can shrink or refuse it. Nothing downstream can override it.",
    ]))

    # ---- 05 research loop ---------------------------------------------------------------------
    st.html(h.eyebrow("05 / Research") + h.section_title("The research loop."))
    research = s.get("research", {})
    provider = research.get("provider", "?")
    model = (research.get("models") or {}).get(provider) or research.get("model", "")
    st.html(h.flow([
        h.tile(svg.icon("chip", 44), "Quant snapshot", "read-only", "HMM regime, volatility, stability, saved backtest"),
        h.tile(svg.icon("search", 44), "Grounded research", f"{provider} {model}".strip(), "web sources kept with every claim"),
        h.tile(svg.icon("doc", 44), "Compare", "agree · conflict · orthogonal", "never a buy or sell signal"),
        h.tile(svg.icon("ledger", 44), "Cache + ledger", f"{research.get('cache_ttl_minutes', '?')} min", "usage and sources recorded"),
    ]))
    st.html(h.notice(
        "The research agent is informational. It cannot execute trades, change strategy or risk settings, or override the "
        "risk manager. Switch the model with research.provider (ollama, openai, anthropic, gemini) in config/settings.yaml.",
        title="Read-only by design",
    ))
