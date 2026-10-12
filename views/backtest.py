"""Backtest view: reads backtest_results/*.csv produced by `python main.py backtest ...`.

Every number shown is computed with the same `backtest/performance.py`
functions `main.py`'s own report uses -- nothing here recomputes financial
math independently.
"""

from __future__ import annotations

import os

import pandas as pd
import streamlit as st

from backtest.performance import (
    cagr,
    calmar_ratio,
    confidence_bucketed_performance,
    daily_returns,
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
from ui import charts
from ui import html as h
from ui.widgets.portfolio_field import portfolio_field, strand_table
from views.common import BACKTEST_DIR, load_settings


def _field_series(per_symbol: pd.DataFrame, equity: pd.Series) -> dict[str, pd.Series]:
    """Portfolio + per-symbol equity paths (real data) used by the diversification card and the 3D field."""
    series = {"PORTFOLIO": equity}
    if len(per_symbol.columns) > 1:        # with a single symbol the portfolio IS that symbol; don't draw it twice
        for column in per_symbol.columns:
            series[str(column)] = per_symbol[column].dropna()
    elif len(per_symbol.columns) == 1:
        series = {str(per_symbol.columns[0]): equity}
    return series


def _render_results() -> None:
    equity_path = f"{BACKTEST_DIR}/equity_curve.csv"
    if not os.path.exists(equity_path):
        st.html(h.notice(f"No backtest results in {BACKTEST_DIR}/ yet. Run `python main.py backtest --symbols SPY --compare` first.",
                         title="Nothing to show"))
        return

    settings = load_settings()
    risk_free_rate = settings["backtest"]["risk_free_rate"]

    equity = pd.read_csv(equity_path, index_col=0, parse_dates=True)["equity"]
    trades = pd.read_csv(f"{BACKTEST_DIR}/trade_log.csv", parse_dates=["decision_date", "fill_date"])
    regime_history = pd.read_csv(f"{BACKTEST_DIR}/regime_history.csv", parse_dates=["date"])
    returns = daily_returns(equity)
    stats = trade_stats(trades, equity)
    worst = worst_case_stats(equity)

    per_symbol_path = f"{BACKTEST_DIR}/per_symbol_equity.csv"
    per_symbol = pd.read_csv(per_symbol_path, index_col=0, parse_dates=True) if os.path.exists(per_symbol_path) else pd.DataFrame()

    st.html(h.kpi_row([
        h.kpi("Total return", f"{total_return(equity):.2%}"),
        h.kpi("CAGR", f"{cagr(equity):.2%}"),
        h.kpi("Sharpe", f"{sharpe_ratio(returns, risk_free_rate):.2f}"),
        h.kpi("Sortino", f"{sortino_ratio(returns, risk_free_rate):.2f}"),
        h.kpi("Calmar", f"{calmar_ratio(equity):.2f}"),
        h.kpi("Max drawdown", f"{max_drawdown(equity):.2%}", f"{max_drawdown_duration(equity)}d duration"),
        h.kpi("Win rate", f"{stats['win_rate']:.2%}", f"{stats['total_trades']} trades"),
    ], min_width=140))

    # ---- portfolio field (3D) ------------------------------------------------------------
    field_series = _field_series(per_symbol, equity)
    left, right = st.columns([1, 2], vertical_alignment="center")
    with left:
        st.html(
            h.eyebrow("01 / Portfolio field")
            + f'<div class="rt-section" style="margin-top:.2rem">{h.esc(", ".join(k for k in field_series) or "Portfolio")}</div>'
            + h.lead("Normalized return paths from this backtest, drawn as one abstract shape. Drag to rotate; "
                     "the geometry is decorative, the numbers below are the result.")
            + f'<div class="rt-mono-muted">{len(field_series)} series · 96 strands · real equity paths</div>'
        )
    with right:
        portfolio_field(field_series)
    with st.expander("Data behind the field"):
        st.dataframe(strand_table(field_series), hide_index=True, width="stretch")

    # ---- three cards: net performance / return spectrum / diversification ----------------
    series = _field_series(per_symbol, equity)
    c1, c2, c3 = st.columns(3)
    with c1:
        with st.container(key="rtcard-net"):
            st.html(f'<div class="rt-label">Net performance · CAGR</div><div class="rt-bignum">{cagr(equity):.2%}</div>')
            st.plotly_chart(charts.growth_of_100(dict(list(series.items())[:5])),
                            width="stretch", config=charts.PLOT_CONFIG)
            st.html(f'<div class="rt-label">Growth of $100 · ${100 * equity.iloc[-1] / equity.iloc[0]:,.0f}</div>')
    with c2:
        with st.container(key="rtcard-spectrum"):
            st.html(f'<div class="rt-label">Return spectrum · Sharpe</div>'
                    f'<div class="rt-bignum">{sharpe_ratio(returns, risk_free_rate):.2f}<small>net</small></div>')
            st.plotly_chart(charts.monthly_returns(returns), width="stretch", config=charts.PLOT_CONFIG)
            st.html('<div class="rt-label">Monthly returns</div>')
    with c3:
        with st.container(key="rtcard-diversification"):
            if len(series) >= 2:
                frame = pd.concat({k: daily_returns(v) for k, v in series.items()}, axis=1).dropna()
                corr = frame.corr()
                n = len(corr)
                upper = [corr.iloc[i, j] for i in range(n) for j in range(i + 1, n)]   # off-diagonal pairs only
                avg = float(sum(upper) / len(upper)) if upper else 0.0
                st.html(f'<div class="rt-label">Diversification · mean off-diagonal r</div><div class="rt-bignum">{avg:.3f}</div>'
                        + charts.correlation_grid(corr.iloc[:5, :5]))
            else:
                st.html('<div class="rt-label">Diversification</div><div class="rt-bignum">–</div>'
                        + h.notice("Needs two or more symbols in the backtest to show correlation.", title="Single series"))
            st.html('<div class="rt-label">Daily return correlation</div>')

    st.html(h.section_title("Equity & drawdown"))
    st.plotly_chart(charts.equity_drawdown(equity), width="stretch", config=charts.PLOT_CONFIG)

    with st.expander("More performance detail"):
        st.html(h.kpi_row([
            h.kpi("Avg win / avg loss", f"{stats['avg_win']:.2%} / {stats['avg_loss']:.2%}"),
            h.kpi("Profit factor", f"{stats['profit_factor']:.2f}"),
            h.kpi("Avg holding period", f"{stats['avg_holding_period_days']:.1f}d"),
            h.kpi("Worst day / week / month", f"{worst['worst_day']:.2%} / {worst['worst_week']:.2%} / {worst['worst_month']:.2%}"),
            h.kpi("Max consecutive loss days", worst["max_consecutive_loss_days"]),
            h.kpi("Longest time underwater", f"{worst['longest_underwater_days']}d"),
        ]))

    comparison_path = f"{BACKTEST_DIR}/benchmark_comparison.csv"
    if os.path.exists(comparison_path):
        st.html(h.section_title("Benchmark comparison"))
        st.dataframe(pd.read_csv(comparison_path), hide_index=True, width="stretch")

    symbols = sorted(regime_history["symbol"].unique())
    symbol = st.selectbox("Symbol", symbols)

    st.html(h.section_title(f"{symbol}: regime & allocation"))
    st.plotly_chart(charts.regime_allocation(regime_history[regime_history["symbol"] == symbol]),
                    width="stretch", config=charts.PLOT_CONFIG)

    symbol_trades = trades[trades["symbol"] == symbol]
    if symbol in per_symbol.columns:
        per_symbol_equity = per_symbol[symbol].dropna()
        symbol_returns = daily_returns(per_symbol_equity)
        symbol_pnls = trade_pnls(symbol_trades, per_symbol_equity)
        regime_labels = regime_history[regime_history["symbol"] == symbol].set_index("date")["regime_label"]

        st.html(h.section_title(f"{symbol}: performance by regime"))
        st.dataframe(regime_performance_breakdown(symbol_returns, regime_labels, symbol_trades, symbol_pnls, risk_free_rate),
                     hide_index=True, width="stretch")

        st.html(h.section_title(f"{symbol}: performance by confidence bucket"))
        st.dataframe(confidence_bucketed_performance(symbol_trades, symbol_pnls), hide_index=True, width="stretch")

    st.html(h.section_title("Trade log"))
    st.dataframe(symbol_trades.sort_values("fill_date", ascending=False), hide_index=True, width="stretch")


def render() -> None:
    st.html(h.eyebrow("02 / Backtest") + h.hero("Walk-forward results"))
    _render_results()
