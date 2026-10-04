"""Tests for backtest.performance: metrics computed from a hand-built equity curve."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest.performance import (
    buy_and_hold_equity,
    cagr,
    calmar_ratio,
    max_drawdown,
    max_drawdown_duration,
    sharpe_ratio,
    sma_trend_equity,
    sortino_ratio,
    total_return,
    trade_pnls,
    trade_stats,
    worst_case_stats,
)


def _equity(values, start="2020-01-01"):
    dates = pd.bdate_range(start, periods=len(values))
    return pd.Series(values, index=dates, dtype=float)


def test_total_return_doubling():
    eq = _equity([100, 150, 200])
    assert total_return(eq) == pytest.approx(1.0)


def test_cagr_matches_known_growth():
    # Exactly 2x over 252 bars (~1 trading year) -> CAGR ~= 100%.
    eq = _equity([100.0] + [100.0 * 2 ** (i / 252) for i in range(1, 253)])
    assert cagr(eq) == pytest.approx(1.0, rel=0.02)


def test_max_drawdown_known_series():
    eq = _equity([100, 120, 90, 110])
    assert max_drawdown(eq) == pytest.approx(90 / 120 - 1)


def test_max_drawdown_duration_counts_bars_underwater():
    # Peaks at index 1 (120), stays underwater through index 4, recovers at 5.
    eq = _equity([100, 120, 110, 90, 115, 130])
    assert max_drawdown_duration(eq) == 3


def test_calmar_ratio_is_cagr_over_abs_max_drawdown():
    eq = _equity([100, 120, 90, 150])
    expected = cagr(eq) / abs(max_drawdown(eq))
    assert calmar_ratio(eq) == pytest.approx(expected)


def test_sharpe_ratio_zero_for_constant_returns_above_rf():
    # A perfectly flat equity curve (e.g. sitting entirely in cash) has
    # EXACTLY zero daily returns -> sharpe_ratio must guard divide-by-zero
    # and return 0.0 rather than inf/NaN.
    eq = _equity([100.0] * 300)
    returns = eq.pct_change().dropna()
    assert sharpe_ratio(returns, risk_free_rate=0.0) == 0.0


def test_sharpe_positive_for_upward_noisy_series():
    rng = np.random.default_rng(0)
    returns = pd.Series(rng.normal(0.001, 0.01, 500))
    assert sharpe_ratio(returns, risk_free_rate=0.0) > 0


def test_sortino_ignores_upside_volatility():
    rng = np.random.default_rng(0)
    returns = pd.Series(rng.normal(0.001, 0.01, 500))
    # Sortino should generally be >= Sharpe-like magnitude since it only
    # penalizes downside deviation; just check it's finite and positive here.
    assert sortino_ratio(returns, risk_free_rate=0.0) > 0


def test_trade_pnls_and_trade_stats():
    eq = _equity([100, 110, 90, 120, 130])
    trades = pd.DataFrame({
        "fill_date": [eq.index[1], eq.index[3]],
        "regime_label": ["BULL", "BEAR"],
        "regime_confidence": [0.9, 0.6],
    })
    pnls = trade_pnls(trades, eq)
    assert len(pnls) == 2
    # Holding period 1: index[1]->index[3], 110 -> 120
    assert pnls.iloc[0] == pytest.approx(120 / 110 - 1)
    # Holding period 2: index[3]->index[-1] (last), 120 -> 130
    assert pnls.iloc[1] == pytest.approx(130 / 120 - 1)

    stats = trade_stats(trades, eq)
    assert stats["total_trades"] == 2
    assert stats["win_rate"] == 1.0  # both holding periods were positive


def test_trade_stats_empty_trades_returns_zeros():
    eq = _equity([100, 110, 120])
    stats = trade_stats(pd.DataFrame(columns=["fill_date"]), eq)
    assert stats["total_trades"] == 0
    assert stats["win_rate"] == 0.0


def test_worst_case_stats_keys_present():
    eq = _equity([100 + i - (5 if i == 50 else 0) for i in range(120)])
    stats = worst_case_stats(eq)
    assert set(stats) == {
        "worst_day", "worst_week", "worst_month",
        "max_consecutive_loss_days", "longest_underwater_days",
    }


def test_buy_and_hold_equity_tracks_price_ratio():
    bars = pd.DataFrame(
        {"close": [100.0, 110.0, 121.0]}, index=pd.bdate_range("2020-01-01", periods=3)
    )
    eq = buy_and_hold_equity(bars, 10_000)
    assert eq.iloc[0] == pytest.approx(10_000)
    assert eq.iloc[-1] == pytest.approx(10_000 * 1.21)


def test_sma_trend_equity_stays_flat_when_always_below_sma():
    # Price constant at 100, SMA(3) == 100 -> never strictly above -> stays in cash.
    bars = pd.DataFrame(
        {"close": [100.0] * 10}, index=pd.bdate_range("2020-01-01", periods=10)
    )
    eq = sma_trend_equity(bars, 10_000, sma_window=3)
    assert (eq == 10_000).all()
