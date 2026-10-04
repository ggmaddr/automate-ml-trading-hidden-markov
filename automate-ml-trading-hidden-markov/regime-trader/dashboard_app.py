"""Streamlit UI for regime-trader.

    streamlit run dashboard_app.py        # run from inside regime-trader/

Two views:

- **Live**: polls `dashboard_state.json`, which a running
  `python main.py run --web-dashboard` instance writes every
  `monitoring.dashboard_refresh_seconds` via
  `TradingBot.start_json_dashboard_feed()`. There is no direct IPC into an
  already-running bot (same constraint the terminal `--live-dashboard`
  works around in-process, and `--dashboard` works around with a one-shot
  snapshot -- see the README's "How do I open the dashboard UI?" FAQ), so
  this tab is a separate process polling a file the bot keeps overwriting.
- **Backtest**: reads `backtest_results/*.csv`, as produced by
  `python main.py backtest ...`. Every number shown is computed with the
  same `backtest/performance.py` functions `main.py`'s own report uses --
  nothing here recomputes financial math independently.
"""

from __future__ import annotations

import json
import os
import time

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yaml
from plotly.subplots import make_subplots

from backtest.performance import (
    cagr,
    calmar_ratio,
    confidence_bucketed_performance,
    daily_returns,
    drawdown_series,
    max_drawdown,
    max_drawdown_duration,
    regime_performance_breakdown,
    sharpe_ratio,
    sortino_ratio,
    total_return,
    trade_pnls,
    trade_stats,
    worst_case_stats,
)

DASHBOARD_JSON_PATH = "dashboard_state.json"
STATE_SNAPSHOT_PATH = "state_snapshot.json"
BACKTEST_DIR = "backtest_results"

st.set_page_config(page_title="regime-trader", layout="wide")


@st.cache_data(ttl=60)
def load_settings() -> dict:
    with open("config/settings.yaml") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Live
# ---------------------------------------------------------------------------

def risk_badge(current: float, limit: float) -> str:
    pct_of_limit = (current / limit) if limit else 0.0
    if pct_of_limit < 0.5:
        return f":green[OK]  ({current:.1%} / {limit:.0%} limit)"
    if pct_of_limit < 0.9:
        return f":orange[WARNING]  ({current:.1%} / {limit:.0%} limit)"
    return f":red[HALT]  ({current:.1%} / {limit:.0%} limit)"


def render_live() -> None:
    if not os.path.exists(DASHBOARD_JSON_PATH):
        st.warning(
            f"No `{DASHBOARD_JSON_PATH}` found. Start the bot with "
            "`python main.py run --web-dashboard` (add `--dry-run` to watch "
            "without placing orders)."
        )
        if os.path.exists(STATE_SNAPSHOT_PATH):
            with open(STATE_SNAPSHOT_PATH) as f:
                snapshot = json.load(f)
            st.caption(f"Last saved state_snapshot.json (from a prior shutdown/error), saved at {snapshot.get('saved_at')}:")
            st.json(snapshot)
        return

    age_seconds = time.time() - os.path.getmtime(DASHBOARD_JSON_PATH)
    with open(DASHBOARD_JSON_PATH) as f:
        state = json.load(f)

    refresh_seconds = load_settings().get("monitoring", {}).get("dashboard_refresh_seconds", 5)
    if age_seconds > max(refresh_seconds * 3, 15):
        st.error(f"Snapshot is {age_seconds:.0f}s old -- is the bot still running?")
    else:
        st.caption(f"Last updated {age_seconds:.0f}s ago")

    regime, portfolio = state.get("regime", {}), state.get("portfolio", {})
    system = state.get("system", {})

    cols = st.columns(6)
    cols[0].metric("Regime", regime.get("label", "?"), f"{regime.get('probability', 0):.0%} confidence")
    cols[1].metric("Stability", f"{regime.get('stability_bars', 0)} bars")
    cols[2].metric("Flicker", f"{regime.get('flicker_rate', 0)}/{regime.get('flicker_window', 20)}")
    cols[3].metric("Equity", f"${portfolio.get('equity', 0):,.0f}", f"{portfolio.get('daily_pnl', 0):+,.0f} today")
    cols[4].metric("Allocation", f"{portfolio.get('allocation', 0):.0%}")
    cols[5].metric("Leverage", f"{portfolio.get('leverage', 1.0):.2f}x")

    st.subheader("Positions")
    positions = state.get("positions", [])
    if positions:
        df = pd.DataFrame(positions).rename(
            columns={"symbol": "Symbol", "direction": "Dir", "price": "Price", "pnl_pct": "P&L %", "stop": "Stop", "held": "Held"}
        )
        st.dataframe(df, hide_index=True, width="stretch")
    else:
        st.info("No open positions.")

    left, right = st.columns(2)
    with left:
        st.subheader("Risk status")
        risk = state.get("risk", {})
        st.markdown(f"**Daily drawdown**: {risk_badge(risk.get('daily_dd', 0), risk.get('daily_dd_limit', 0.03))}")
        st.markdown(f"**Drawdown from peak**: {risk_badge(risk.get('peak_dd', 0), risk.get('peak_dd_limit', 0.10))}")

        st.subheader("System")
        st.markdown(f"- Data feed: {':green[OK]' if system.get('data_ok') else ':red[DOWN]'}")
        st.markdown(f"- Broker API: {':green[OK]' if system.get('api_ok') else ':red[DOWN]'} ({system.get('api_latency_ms', 0)}ms)")
        st.markdown(f"- HMM last trained: {system.get('hmm_age', '?')}")
        st.markdown(f"- Mode: {':blue[PAPER]' if system.get('paper', True) else ':red[LIVE]'}")

    with right:
        st.subheader("Recent signals")
        signals = list(reversed(state.get("recent_signals", [])))
        if signals:
            st.dataframe(
                pd.DataFrame(signals).rename(columns={"time": "Time", "symbol": "Symbol", "action": "Action", "reason": "Reason"}),
                hide_index=True, width="stretch",
            )
        else:
            st.info("No signals yet.")


# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------

def _equity_drawdown_figure(equity: pd.Series) -> go.Figure:
    dd = drawdown_series(equity)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.05)
    fig.add_trace(go.Scatter(x=equity.index, y=equity.values, name="Equity", line=dict(color="#2563eb")), row=1, col=1)
    fig.add_trace(go.Scatter(x=dd.index, y=dd.values, name="Drawdown", fill="tozeroy", line=dict(color="#dc2626")), row=2, col=1)
    fig.update_yaxes(title_text="Equity ($)", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown", tickformat=".0%", row=2, col=1)
    fig.update_layout(height=500, margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
    return fig


def _regime_figure(regime_df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.5, 0.5], vertical_spacing=0.08)
    for label, group in regime_df.groupby("regime_label"):
        fig.add_trace(go.Scatter(x=group["date"], y=[label] * len(group), mode="markers", name=label, marker=dict(size=5)), row=1, col=1)
    fig.add_trace(
        go.Scatter(x=regime_df["date"], y=regime_df["target_allocation"], name="Target allocation", line=dict(color="#16a34a")),
        row=2, col=1,
    )
    fig.update_yaxes(title_text="Regime", row=1, col=1)
    fig.update_yaxes(title_text="Target allocation", tickformat=".0%", row=2, col=1)
    fig.update_layout(height=450, margin=dict(l=10, r=10, t=10, b=10))
    return fig


def render_backtest() -> None:
    equity_path = f"{BACKTEST_DIR}/equity_curve.csv"
    if not os.path.exists(equity_path):
        st.info(f"No backtest results in `{BACKTEST_DIR}/` yet. Run `python main.py backtest --symbols SPY --compare` first.")
        return

    settings = load_settings()
    risk_free_rate = settings["backtest"]["risk_free_rate"]

    equity = pd.read_csv(equity_path, index_col=0, parse_dates=True)["equity"]
    trades = pd.read_csv(f"{BACKTEST_DIR}/trade_log.csv", parse_dates=["decision_date", "fill_date"])
    regime_history = pd.read_csv(f"{BACKTEST_DIR}/regime_history.csv", parse_dates=["date"])
    returns = daily_returns(equity)
    stats = trade_stats(trades, equity)
    worst = worst_case_stats(equity)

    st.subheader("Portfolio performance")
    cols = st.columns(7)
    cols[0].metric("Total return", f"{total_return(equity):.2%}")
    cols[1].metric("CAGR", f"{cagr(equity):.2%}")
    cols[2].metric("Sharpe", f"{sharpe_ratio(returns, risk_free_rate):.2f}")
    cols[3].metric("Sortino", f"{sortino_ratio(returns, risk_free_rate):.2f}")
    cols[4].metric("Calmar", f"{calmar_ratio(equity):.2f}")
    cols[5].metric("Max drawdown", f"{max_drawdown(equity):.2%}", f"{max_drawdown_duration(equity)}d duration")
    cols[6].metric("Win rate", f"{stats['win_rate']:.2%}", f"{stats['total_trades']} trades")

    st.plotly_chart(_equity_drawdown_figure(equity), width="stretch")

    with st.expander("More performance detail"):
        c1, c2, c3 = st.columns(3)
        c1.metric("Avg win / avg loss", f"{stats['avg_win']:.2%} / {stats['avg_loss']:.2%}")
        c2.metric("Profit factor", f"{stats['profit_factor']:.2f}")
        c3.metric("Avg holding period", f"{stats['avg_holding_period_days']:.1f}d")
        c1.metric("Worst day / week / month", f"{worst['worst_day']:.2%} / {worst['worst_week']:.2%} / {worst['worst_month']:.2%}")
        c2.metric("Max consecutive loss days", worst["max_consecutive_loss_days"])
        c3.metric("Longest time underwater", f"{worst['longest_underwater_days']}d")

    comparison_path = f"{BACKTEST_DIR}/benchmark_comparison.csv"
    if os.path.exists(comparison_path):
        st.subheader("Benchmark comparison")
        st.dataframe(pd.read_csv(comparison_path), hide_index=True, width="stretch")

    symbols = sorted(regime_history["symbol"].unique())
    symbol = st.selectbox("Symbol", symbols)

    st.subheader(f"{symbol}: regime & allocation over time")
    st.plotly_chart(_regime_figure(regime_history[regime_history["symbol"] == symbol]), width="stretch")

    per_symbol_path = f"{BACKTEST_DIR}/per_symbol_equity.csv"
    symbol_trades = trades[trades["symbol"] == symbol]
    if os.path.exists(per_symbol_path):
        per_symbol_equity = pd.read_csv(per_symbol_path, index_col=0, parse_dates=True)[symbol].dropna()
        symbol_returns = daily_returns(per_symbol_equity)
        symbol_pnls = trade_pnls(symbol_trades, per_symbol_equity)
        regime_labels = regime_history[regime_history["symbol"] == symbol].set_index("date")["regime_label"]

        st.subheader(f"{symbol}: performance by regime")
        breakdown = regime_performance_breakdown(symbol_returns, regime_labels, symbol_trades, symbol_pnls, risk_free_rate)
        st.dataframe(breakdown, hide_index=True, width="stretch")

        st.subheader(f"{symbol}: performance by confidence bucket")
        conf = confidence_bucketed_performance(symbol_trades, symbol_pnls)
        st.dataframe(conf, hide_index=True, width="stretch")

    st.subheader("Trade log")
    st.dataframe(symbol_trades.sort_values("fill_date", ascending=False), hide_index=True, width="stretch")


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

st.title("regime-trader")

with st.sidebar:
    view = st.radio("View", ["Live", "Backtest"], key="view")
    auto_refresh = False
    if view == "Live":
        auto_refresh = st.toggle("Auto-refresh", value=True, key="auto_refresh")
        refresh_interval = st.number_input("Refresh interval (seconds)", min_value=1, value=5, key="refresh_interval")
        if st.button("Reload now"):
            st.rerun()

if view == "Live":
    render_live()
else:
    render_backtest()

if view == "Live" and auto_refresh:
    time.sleep(refresh_interval)
    st.rerun()
