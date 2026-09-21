"""Walk-forward allocation backtester."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class BacktestResult:
    """Output of a single walk-forward backtest run."""

    equity_curve: pd.Series
    trades: pd.DataFrame
    regime_history: pd.DataFrame


class Backtester:
    """Runs expanding-window walk-forward backtests with no look-ahead bias."""

    def __init__(self, config: dict) -> None:
        raise NotImplementedError

    def run(self, data: dict[str, pd.DataFrame]) -> BacktestResult:
        """Run a full walk-forward backtest across train/test windows."""
        raise NotImplementedError

    def _run_window(self, train_data: pd.DataFrame, test_data: pd.DataFrame) -> BacktestResult:
        """Train on train_data, evaluate (filtered inference only) on test_data."""
        raise NotImplementedError
