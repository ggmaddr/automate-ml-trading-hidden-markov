"""Allocation-based walk-forward backtester.

This does NOT simulate individual trade entries/exits with stop-losses —
"no individual trade stops in backtest, stops are for live trading only."
Instead, every bar it asks one question: "given the detected volatility
regime, what fraction of the portfolio should be invested right now?" and
rebalances toward that target only when it has drifted far enough to be
worth the slippage. This is how real systematic allocation strategies work.

WALK-FORWARD VALIDATION, the whole point of this module: the HMM is
retrained periodically on a rolling in-sample (IS) window, then evaluated
bar-by-bar on the out-of-sample (OOS) window immediately following it. The
model never sees OOS data while training, and regime inference inside OOS
uses only data up to and including the current bar (see core.hmm_engine).
Portfolio state (cash, shares, pending orders) carries over continuously
across window boundaries — only the MODEL gets refreshed each window, the
money does not reset.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from core.hmm_engine import HMMRegimeEngine
from core.regime_strategies import StrategyOrchestrator
from data.feature_engineering import compute_features, log_returns


@dataclass
class TradeRecord:
    """One rebalance event: decided on `decision_date` (using that day's close),
    filled on `fill_date` (the next bar's open) — this is the 1-bar fill delay.
    """

    symbol: str
    decision_date: pd.Timestamp
    fill_date: pd.Timestamp
    old_allocation: float
    new_allocation: float
    fill_price: float
    shares_delta: int
    slippage_cost: float
    regime_label: str
    regime_confidence: float


@dataclass
class BacktestResult:
    """Output of a walk-forward backtest run, possibly combining several symbols."""

    equity_curve: pd.Series  # total portfolio equity, indexed by date
    trades: pd.DataFrame  # one row per TradeRecord, across all symbols
    regime_history: pd.DataFrame  # one row per (symbol, OOS bar)
    per_symbol_equity: dict[str, pd.Series] = field(default_factory=dict)


class _SymbolBook:
    """Mutable per-symbol bookkeeping, threaded continuously across walk-forward windows."""

    def __init__(self, symbol: str, initial_capital: float) -> None:
        self.symbol = symbol
        self.cash = initial_capital
        self.shares = 0
        self.current_allocation = 0.0

        self.pending_target: Optional[float] = None
        self.pending_decision_date: Optional[pd.Timestamp] = None
        self.pending_regime_label: str = ""
        self.pending_regime_confidence: float = 0.0

        self.equity_dates: list = []
        self.equity_values: list[float] = []
        self.trades: list[TradeRecord] = []
        self.regime_rows: list[dict] = []


class Backtester:
    """Runs walk-forward allocation backtests with strictly causal regime inference.

    `config` is the full settings dict (needs `hmm`, `strategy`, and
    `backtest` sections — see config/settings.yaml).
    """

    def __init__(self, config: dict) -> None:
        self.config = config
        self.hmm_config = config["hmm"]
        self.strategy_config = config["strategy"]
        self.backtest_config = config["backtest"]

    def run(self, data: dict[str, pd.DataFrame]) -> BacktestResult:
        """Run an independent walk-forward sleeve per symbol and combine them.

        Each symbol gets an equal share of `backtest.initial_capital` and
        trades on its OWN HMM-detected regime (trained from its own price
        history) — sleeves do not share a regime signal.
        """
        if not data:
            raise ValueError("data must contain at least one symbol")

        initial_capital = self.backtest_config["initial_capital"]
        capital_per_symbol = initial_capital / len(data)

        books: dict[str, _SymbolBook] = {}
        for symbol, bars in data.items():
            books[symbol] = self._run_symbol(symbol, bars, capital_per_symbol)

        return self._combine(books, capital_per_symbol)

    # ------------------------------------------------------------------
    # Per-symbol walk-forward loop
    # ------------------------------------------------------------------

    def _run_symbol(self, symbol: str, bars: pd.DataFrame, initial_capital: float) -> _SymbolBook:
        features = compute_features(bars)
        raw_returns = log_returns(bars["close"], 1)
        book = _SymbolBook(symbol, initial_capital)

        train_window = self.backtest_config["train_window"]
        test_window = self.backtest_config["test_window"]
        step_size = self.backtest_config["step_size"]

        n = len(features)
        train_start = 0
        ran_any_window = False
        while train_start + train_window + test_window <= n:
            train_end = train_start + train_window
            test_end = train_end + test_window
            self._run_window(book, bars, features, raw_returns, train_start, train_end, test_end)
            ran_any_window = True
            train_start += step_size

        if not ran_any_window:
            raise ValueError(
                f"{symbol}: only {n} valid feature rows, need at least "
                f"{train_window + test_window} for one walk-forward window"
            )

        return book

    def _run_window(
        self,
        book: _SymbolBook,
        bars: pd.DataFrame,
        features: pd.DataFrame,
        raw_returns: pd.Series,
        train_start: int,
        train_end: int,
        test_end: int,
    ) -> None:
        """Train fresh on [train_start:train_end), trade on [train_end:test_end)."""
        train_features = features.iloc[train_start:train_end]
        train_returns = raw_returns.reindex(train_features.index)

        engine = HMMRegimeEngine(self.hmm_config)
        engine.fit(train_features, train_returns)
        orchestrator = StrategyOrchestrator(self.strategy_config, engine.regime_info)

        # Warm up the stability/flicker filter over the IS period so OOS
        # doesn't start cold (consecutive_bars=1 on the very first OOS day).
        for i in range(train_start, train_end):
            engine.update(features.iloc[train_start : i + 1])

        slippage = self.backtest_config["slippage_pct"]
        rebalance_threshold = self.strategy_config["rebalance_threshold"]

        for i in range(train_end, test_end):
            date = features.index[i]

            # 1. Fill any rebalance DECIDED on the previous bar, at TODAY's open.
            if book.pending_target is not None:
                self._fill(book, bars, date, slippage)

            # 2. Mark to market at today's close.
            close_price = float(bars.loc[date, "close"])
            equity = book.cash + book.shares * close_price
            book.equity_dates.append(date)
            book.equity_values.append(equity)

            # 3. Decide (using data up to and including TODAY) what tomorrow's
            #    target allocation should be. Filtered inference is restarted
            #    at this window's train_start, not at the dawn of history —
            #    a freshly retrained model has no notion of "before its own
            #    training data" (see core.hmm_engine's forward algorithm).
            window_features = features.iloc[train_start : i + 1]
            regime_state = engine.update(window_features)
            is_flickering = engine.is_flickering()

            price_history = bars.loc[:date]
            signals = orchestrator.generate_signals(
                [book.symbol], {book.symbol: price_history}, regime_state, is_flickering
            )
            signal = signals[0] if signals else None
            target_allocation = (
                signal.position_size_pct * signal.leverage if signal is not None else book.current_allocation
            )

            book.regime_rows.append(
                {
                    "date": date,
                    "symbol": book.symbol,
                    "regime_id": regime_state.state_id,
                    "regime_label": regime_state.label,
                    "confidence": regime_state.probability,
                    "is_confirmed": regime_state.is_confirmed,
                    "is_flickering": is_flickering,
                    "consecutive_bars": regime_state.consecutive_bars,
                    "target_allocation": target_allocation,
                }
            )

            if signal is None:
                continue  # not enough price history yet for this symbol's own EMA/ATR

            if abs(target_allocation - book.current_allocation) > rebalance_threshold:
                book.pending_target = target_allocation
                book.pending_decision_date = date
                book.pending_regime_label = regime_state.label
                book.pending_regime_confidence = regime_state.probability

    def _fill(self, book: _SymbolBook, bars: pd.DataFrame, date: pd.Timestamp, slippage: float) -> None:
        """Execute a pending rebalance at `date`'s open, per the guide's allocation math:

            equity = cash + shares * price
            target_shares = int(equity * target_allocation / price)
            delta = target_shares - current_shares
            cash -= delta * price   (price here is the SLIPPED fill price)
            shares = target_shares

        Leverage > 1.0 can make `cash` go negative — that's margin, not a bug
        (equity = cash + shares*price is still correct since share value
        exceeds the margin debt).
        """
        open_price = float(bars.loc[date, "open"])
        target_allocation = book.pending_target
        assert target_allocation is not None

        equity = book.cash + book.shares * open_price
        target_shares = int(equity * target_allocation / open_price)
        delta = target_shares - book.shares

        if delta > 0:
            fill_price = open_price * (1 + slippage)
        elif delta < 0:
            fill_price = open_price * (1 - slippage)
        else:
            fill_price = open_price

        slippage_cost = abs(delta) * open_price * slippage
        book.cash -= delta * fill_price
        book.shares = target_shares

        book.trades.append(
            TradeRecord(
                symbol=book.symbol,
                decision_date=book.pending_decision_date,
                fill_date=date,
                old_allocation=book.current_allocation,
                new_allocation=target_allocation,
                fill_price=fill_price,
                shares_delta=delta,
                slippage_cost=slippage_cost,
                regime_label=book.pending_regime_label,
                regime_confidence=book.pending_regime_confidence,
            )
        )
        book.current_allocation = target_allocation
        book.pending_target = None

    # ------------------------------------------------------------------
    # Combining symbol sleeves into one portfolio result
    # ------------------------------------------------------------------

    def _combine(self, books: dict[str, _SymbolBook], capital_per_symbol: float) -> BacktestResult:
        per_symbol_equity: dict[str, pd.Series] = {}
        for symbol, book in books.items():
            per_symbol_equity[symbol] = pd.Series(
                book.equity_values, index=pd.DatetimeIndex(book.equity_dates), name=symbol
            )

        combined = pd.concat(per_symbol_equity.values(), axis=1).sort_index()
        for col in combined.columns:
            # Before a symbol's own walk-forward starts, treat it as sitting
            # in cash (not as a missing/garbage value) so the sum is correct.
            combined[col] = combined[col].ffill().fillna(capital_per_symbol)
        total_equity = combined.sum(axis=1)
        total_equity.name = "equity"

        all_trades = [t.__dict__ for book in books.values() for t in book.trades]
        all_regime_rows = [row for book in books.values() for row in book.regime_rows]

        trades_df = pd.DataFrame(all_trades)
        regime_df = pd.DataFrame(all_regime_rows)

        return BacktestResult(
            equity_curve=total_equity,
            trades=trades_df,
            regime_history=regime_df,
            per_symbol_equity=per_symbol_equity,
        )
