"""Performance metrics: Sharpe, drawdown, regime breakdown, benchmark comparison."""

from __future__ import annotations

import pandas as pd


def sharpe_ratio(returns: pd.Series, risk_free_rate: float) -> float:
    raise NotImplementedError


def sortino_ratio(returns: pd.Series, risk_free_rate: float) -> float:
    raise NotImplementedError


def max_drawdown(equity_curve: pd.Series) -> float:
    raise NotImplementedError


def regime_performance_breakdown(returns: pd.Series, regime_labels: pd.Series) -> pd.DataFrame:
    """Return per-regime return/vol/Sharpe/time-in-regime statistics."""
    raise NotImplementedError


def benchmark_comparison(returns: pd.Series, benchmark_returns: pd.Series) -> dict:
    """Compare strategy returns against a buy-and-hold benchmark."""
    raise NotImplementedError
