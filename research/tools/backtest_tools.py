"""Read-only access to saved walk-forward backtest results.

Reads `backtest_results/*.csv` (written by `python main.py backtest ...`)
and computes metrics with the project's own `backtest.performance` functions.
Never runs a backtest and never writes anything.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from backtest.performance import (
    daily_returns,
    max_drawdown,
    regime_performance_breakdown,
    sharpe_ratio,
    total_return,
    trade_pnls,
    trade_stats,
)
from research.config import PROJECT_ROOT
from research.schemas import BacktestContext

DEFAULT_RESULTS_DIR = PROJECT_ROOT / "backtest_results"
DEFAULT_RISK_FREE_RATE = 0.045


def _finite(value: Any) -> Any:
    """JSON/Pydantic-safe: NaN/inf -> None, numpy scalars -> python."""
    if isinstance(value, float) or hasattr(value, "item"):
        try:
            f = float(value)
        except (TypeError, ValueError):
            return value
        return f if math.isfinite(f) else None
    return value


def _sanitize(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    return _finite(obj)


class BacktestReader:
    def __init__(self, results_dir: Path | str = DEFAULT_RESULTS_DIR, risk_free_rate: float = DEFAULT_RISK_FREE_RATE) -> None:
        self.results_dir = Path(results_dir)
        self.risk_free_rate = risk_free_rate

    def get_backtest_summary(self, symbol: str) -> BacktestContext:
        """Saved walk-forward metrics for `symbol`, or `available=False`."""
        symbol = symbol.upper()
        equity_path = self.results_dir / "per_symbol_equity.csv"
        trades_path = self.results_dir / "trade_log.csv"
        regimes_path = self.results_dir / "regime_history.csv"
        if not (equity_path.exists() and trades_path.exists() and regimes_path.exists()):
            return BacktestContext(symbol=symbol, available=False)

        per_symbol = pd.read_csv(equity_path, index_col=0, parse_dates=True)
        if symbol not in per_symbol.columns:
            return BacktestContext(symbol=symbol, available=False)
        equity = per_symbol[symbol].dropna()
        if len(equity) < 2:
            return BacktestContext(symbol=symbol, available=False)

        trades = pd.read_csv(trades_path, parse_dates=["decision_date", "fill_date"])
        trades = trades[trades["symbol"] == symbol]
        regimes = pd.read_csv(regimes_path, parse_dates=["date"])
        regimes = regimes[regimes["symbol"] == symbol]

        returns = daily_returns(equity)
        stats = trade_stats(trades, equity)
        pnls = trade_pnls(trades, equity)

        regime_metrics: dict[str, Any] = {}
        if not regimes.empty:
            labels = regimes.set_index("date")["regime_label"]
            breakdown = regime_performance_breakdown(returns, labels, trades, pnls, self.risk_free_rate)
            regime_metrics = {"by_regime": _sanitize(breakdown.to_dict(orient="records"))}

        return BacktestContext(
            symbol=symbol,
            available=True,
            period_start=equity.index[0].date(),
            period_end=equity.index[-1].date(),
            total_return=_finite(total_return(equity)),
            sharpe=_finite(sharpe_ratio(returns, self.risk_free_rate)),
            max_drawdown=_finite(max_drawdown(equity)),
            win_rate=_finite(stats["win_rate"]) if stats["total_trades"] else None,
            regime_metrics=regime_metrics,
        )
