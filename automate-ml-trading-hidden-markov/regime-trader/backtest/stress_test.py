"""Stress testing: crash injection, gap injection, regime-shuffle sanity check.

NOTE on scope: core.risk_manager (Phase 5) doesn't exist yet. Where the
guide calls for a "circuit breaker," these tests read the drawdown limits
straight out of config["risk"] as a placeholder — Phase 5's RiskManager
will enforce these live; here we only check whether the backtest's own
equity curve would have breached them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from ta.volatility import AverageTrueRange

from backtest.backtester import Backtester, BacktestResult
from backtest.performance import max_drawdown

WARMUP_BUFFER = 50
TAIL_BUFFER = 5


def _eligible_dates(bars: pd.DataFrame) -> pd.DatetimeIndex:
    if len(bars) <= WARMUP_BUFFER + TAIL_BUFFER:
        return bars.index[0:0]
    return bars.index[WARMUP_BUFFER:-TAIL_BUFFER]


def inject_crash(bars: pd.DataFrame, rng: np.random.Generator, n_events: int = 10,
                  min_magnitude: float = 0.05, max_magnitude: float = 0.15) -> pd.DataFrame:
    """Copy of `bars` with `n_events` single-day crash gaps at random dates.

    Each crash multiplies that day's OHLC by (1 - magnitude), and every
    later bar is shifted by the same factor so the price series stays
    internally consistent (a permanent step down, not a one-day spike that
    snaps back).
    """
    out = bars.copy()
    eligible = _eligible_dates(out)
    n = min(n_events, len(eligible))
    if n == 0:
        return out

    event_dates = rng.choice(eligible, size=n, replace=False)
    for date in sorted(event_dates):
        magnitude = rng.uniform(min_magnitude, max_magnitude)
        idx = out.index.get_loc(date)
        out.loc[out.index[idx:], ["open", "high", "low", "close"]] *= (1 - magnitude)
    return out


def inject_gaps(bars: pd.DataFrame, rng: np.random.Generator, n_events: int = 10,
                 min_atr_mult: float = 2.0, max_atr_mult: float = 5.0) -> pd.DataFrame:
    """Copy of `bars` with overnight gaps of 2-5x ATR at random dates.

    Unlike a crash, a gap is a one-time shift applied to that day forward
    (not scaled by price level) — it models a surprise overnight move of a
    fixed dollar size relative to the stock's recent typical daily range.
    """
    out = bars.copy()
    atr = AverageTrueRange(high=out["high"], low=out["low"], close=out["close"], window=14).average_true_range()
    eligible = _eligible_dates(out)
    n = min(n_events, len(eligible))
    if n == 0:
        return out

    event_dates = rng.choice(eligible, size=n, replace=False)
    for date in sorted(event_dates):
        idx = out.index.get_loc(date)
        direction = rng.choice([-1.0, 1.0])
        gap_size = direction * rng.uniform(min_atr_mult, max_atr_mult) * atr.loc[date]
        out.loc[out.index[idx:], ["open", "high", "low", "close"]] += gap_size
    return out


def run_crash_stress_test(
    data: dict[str, pd.DataFrame], config: dict, n_simulations: int = 100, seed: int = 0,
) -> dict:
    """Re-run the FULL walk-forward backtest `n_simulations` times, each on
    price data with 10 random crash gaps injected, and report how badly it
    hurts. Note: this retrains the HMM on every simulation, so it is slow by
    design (production-grade validation is meant to run offline) — use a
    small n_simulations and a cheap hmm config for quick/interactive checks.
    """
    max_dd_limit = config.get("risk", {}).get("max_dd_from_peak", 0.10)
    rng = np.random.default_rng(seed)

    max_losses = []
    circuit_breaker_fired = 0
    for _ in range(n_simulations):
        sim_rng = np.random.default_rng(rng.integers(0, 2**32 - 1))
        perturbed = {symbol: inject_crash(bars, sim_rng) for symbol, bars in data.items()}
        result = Backtester(config).run(perturbed)
        dd = max_drawdown(result.equity_curve)
        max_losses.append(dd)
        if abs(dd) >= max_dd_limit:
            circuit_breaker_fired += 1

    return {
        "n_simulations": n_simulations,
        "mean_max_loss": float(np.mean(max_losses)),
        "worst_case_loss": float(np.min(max_losses)),
        "pct_circuit_breaker_fired": circuit_breaker_fired / n_simulations,
    }


def run_gap_stress_test(
    data: dict[str, pd.DataFrame], config: dict, n_simulations: int = 100, seed: int = 1,
) -> dict:
    """Compare the real (un-perturbed) backtest's drawdown to `n_simulations`
    runs with random overnight gaps injected, to isolate how much damage
    gap risk alone adds on top of ordinary market drawdown.
    """
    baseline_result = Backtester(config).run(data)
    baseline_dd = max_drawdown(baseline_result.equity_curve)

    rng = np.random.default_rng(seed)
    perturbed_dds = []
    for _ in range(n_simulations):
        sim_rng = np.random.default_rng(rng.integers(0, 2**32 - 1))
        perturbed = {symbol: inject_gaps(bars, sim_rng) for symbol, bars in data.items()}
        result = Backtester(config).run(perturbed)
        perturbed_dds.append(max_drawdown(result.equity_curve))

    return {
        "n_simulations": n_simulations,
        "baseline_max_drawdown": baseline_dd,
        "mean_perturbed_max_drawdown": float(np.mean(perturbed_dds)),
        "worst_perturbed_max_drawdown": float(np.min(perturbed_dds)),
        "mean_incremental_loss": float(np.mean(perturbed_dds) - baseline_dd),
    }


def run_regime_shuffle_test(
    data: dict[str, pd.DataFrame],
    result: BacktestResult,
    config: dict,
    n_shuffles: int = 50,
    seed: int = 2,
    blowup_threshold: float = -0.50,
) -> dict:
    """Shuffle the historical target-allocation sequence (simulating a regime
    detector that got every single call wrong) and re-simulate the SAME
    price path, to see how much damage pure allocation-sizing bounds alone
    can contain with no risk manager involved.

    If even the worst shuffle doesn't blow past `blowup_threshold`, the
    strategy's own position-size bounds (0.60-0.95, max 1.25x leverage) are
    providing some real containment by themselves. If it DOES blow up, that
    means the HMM being right is load-bearing — risk management (Phase 5)
    needs to be independent of regime detection, not just a backstop.
    """
    regime_df = result.regime_history
    if regime_df.empty:
        return {"real_max_drawdown": 0.0, "n_shuffles": 0, "mean_shuffled_max_drawdown": 0.0,
                "worst_shuffled_max_drawdown": 0.0, "contained": True}

    slippage = config["backtest"]["slippage_pct"]
    rebalance_threshold = config["strategy"]["rebalance_threshold"]
    initial_capital = config["backtest"]["initial_capital"]
    symbols = regime_df["symbol"].unique()
    capital_per_symbol = initial_capital / len(symbols)

    real_dd = max_drawdown(result.equity_curve)
    rng = np.random.default_rng(seed)
    shuffled_dds = []

    for _ in range(n_shuffles):
        total_equity: pd.Series | None = None
        for symbol in symbols:
            sub = regime_df[regime_df["symbol"] == symbol].sort_values("date").reset_index(drop=True)
            shuffled_alloc = sub["target_allocation"].sample(frac=1.0, random_state=int(rng.integers(0, 2**32 - 1))).reset_index(drop=True)
            closes = data[symbol]["close"].reindex(sub["date"]).ffill()

            cash, shares, current_alloc = capital_per_symbol, 0, 0.0
            equity_values = []
            for i, price in enumerate(closes.to_numpy()):
                target = float(shuffled_alloc.iloc[i])
                if abs(target - current_alloc) > rebalance_threshold:
                    equity = cash + shares * price
                    target_shares = int(equity * target / price)
                    delta = target_shares - shares
                    fill_price = price * (1 + slippage) if delta > 0 else price * (1 - slippage) if delta < 0 else price
                    cash -= delta * fill_price
                    shares = target_shares
                    current_alloc = target
                equity_values.append(cash + shares * price)

            symbol_equity = pd.Series(equity_values, index=sub["date"])
            total_equity = symbol_equity if total_equity is None else total_equity.add(symbol_equity, fill_value=0.0)

        shuffled_dds.append(max_drawdown(total_equity))

    worst = float(np.min(shuffled_dds))
    return {
        "real_max_drawdown": real_dd,
        "n_shuffles": n_shuffles,
        "mean_shuffled_max_drawdown": float(np.mean(shuffled_dds)),
        "worst_shuffled_max_drawdown": worst,
        "contained": worst > blowup_threshold,
    }
