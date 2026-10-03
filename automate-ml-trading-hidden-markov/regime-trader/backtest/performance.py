"""Performance metrics for a completed backtest.BacktestResult.

Everything here is a pure function of pandas Series/DataFrames so it can be
tested without running an actual backtest. `equity_curve` is always a
pd.Series of total portfolio value indexed by date; `trades` is always the
DataFrame shape produced by backtest.backtester.TradeRecord.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252

CONFIDENCE_BUCKETS = [
    (-np.inf, 0.50, "<50%"),
    (0.50, 0.60, "50-60%"),
    (0.60, 0.70, "60-70%"),
    (0.70, np.inf, "70%+"),
]


# ---------------------------------------------------------------------------
# Core return/risk metrics
# ---------------------------------------------------------------------------

def daily_returns(equity_curve: pd.Series) -> pd.Series:
    return equity_curve.pct_change().dropna()


def total_return(equity_curve: pd.Series) -> float:
    if len(equity_curve) < 2:
        return 0.0
    return float(equity_curve.iloc[-1] / equity_curve.iloc[0] - 1)


def cagr(equity_curve: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    n_periods = len(equity_curve) - 1
    if n_periods <= 0:
        return 0.0
    total = equity_curve.iloc[-1] / equity_curve.iloc[0]
    years = n_periods / periods_per_year
    if total <= 0:
        return -1.0
    return float(total ** (1 / years) - 1)


def sharpe_ratio(returns: pd.Series, risk_free_rate: float, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    if len(returns) < 2 or returns.std(ddof=0) == 0:
        return 0.0
    period_rf = risk_free_rate / periods_per_year
    excess = returns - period_rf
    return float(np.sqrt(periods_per_year) * excess.mean() / excess.std(ddof=0))


def sortino_ratio(returns: pd.Series, risk_free_rate: float, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    if len(returns) < 2:
        return 0.0
    period_rf = risk_free_rate / periods_per_year
    excess = returns - period_rf
    downside = excess[excess < 0]
    downside_std = downside.std(ddof=0)
    if not downside_std:
        return 0.0
    return float(np.sqrt(periods_per_year) * excess.mean() / downside_std)


def drawdown_series(equity_curve: pd.Series) -> pd.Series:
    running_max = equity_curve.cummax()
    return equity_curve / running_max - 1.0


def max_drawdown(equity_curve: pd.Series) -> float:
    """Most negative drawdown, e.g. -0.23 for a 23% peak-to-trough decline."""
    if len(equity_curve) == 0:
        return 0.0
    return float(drawdown_series(equity_curve).min())


def max_drawdown_duration(equity_curve: pd.Series) -> int:
    """Longest stretch (in bars) equity spent below a prior peak."""
    running_max = equity_curve.cummax()
    underwater = (equity_curve < running_max).to_numpy()
    longest = current = 0
    for flag in underwater:
        if flag:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return int(longest)


def calmar_ratio(equity_curve: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    mdd = max_drawdown(equity_curve)
    if mdd == 0:
        return 0.0
    return float(cagr(equity_curve, periods_per_year) / abs(mdd))


# ---------------------------------------------------------------------------
# Trade-level stats
#
# A "trade" here is a holding period between two consecutive rebalances
# (this is an allocation-based backtest, not discrete entry/exit trades).
# ---------------------------------------------------------------------------

def trade_pnls(trades: pd.DataFrame, equity_curve: pd.Series) -> pd.Series:
    """P&L (%) of each holding period, indexed by the fill_date that STARTS it.

    The period ends at the next trade's fill_date, or at the end of the
    backtest for the last trade.
    """
    if trades.empty:
        return pd.Series(dtype=float)

    fill_dates = sorted(trades["fill_date"].unique())
    bounds = list(fill_dates) + [equity_curve.index[-1]]
    pnls = []
    for start, end in zip(bounds[:-1], bounds[1:]):
        start_equity = equity_curve.loc[start]
        end_equity = equity_curve.loc[end]
        pnls.append(end_equity / start_equity - 1.0 if start_equity else 0.0)
    return pd.Series(pnls, index=fill_dates)


def trade_stats(trades: pd.DataFrame, equity_curve: pd.Series) -> dict:
    pnls = trade_pnls(trades, equity_curve)
    if pnls.empty:
        return {
            "total_trades": 0, "win_rate": 0.0, "avg_win": 0.0, "avg_loss": 0.0,
            "profit_factor": 0.0, "avg_holding_period_days": 0.0,
        }

    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]
    loss_sum = losses.sum()

    fill_dates = pd.Series(sorted(trades["fill_date"].unique()))
    holding_days = fill_dates.diff().dropna().dt.days

    return {
        "total_trades": int(len(trades)),
        "win_rate": float(len(wins) / len(pnls)),
        "avg_win": float(wins.mean()) if len(wins) else 0.0,
        "avg_loss": float(losses.mean()) if len(losses) else 0.0,
        "profit_factor": float(wins.sum() / abs(loss_sum)) if loss_sum != 0 else float("inf"),
        "avg_holding_period_days": float(holding_days.mean()) if len(holding_days) else 0.0,
    }


# ---------------------------------------------------------------------------
# Regime / confidence breakdowns — "does the HMM actually add value?"
# ---------------------------------------------------------------------------

def regime_performance_breakdown(
    returns: pd.Series,
    regime_labels: pd.Series,
    trades: pd.DataFrame,
    pnls: pd.Series,
    risk_free_rate: float,
) -> pd.DataFrame:
    """Per-regime: % time in, return contribution, avg trade P&L, win rate, Sharpe."""
    aligned_labels = regime_labels.reindex(returns.index).ffill()
    trades_by_date = trades.set_index("fill_date")["regime_label"] if not trades.empty else pd.Series(dtype=str)

    rows = []
    for label in sorted(aligned_labels.dropna().unique()):
        mask = aligned_labels == label
        sub_returns = returns[mask]

        label_trade_dates = trades_by_date[trades_by_date == label].index
        sub_pnls = pnls.reindex(label_trade_dates).dropna()

        rows.append(
            {
                "regime": label,
                "pct_time_in": float(mask.mean()),
                "return_contribution": float(sub_returns.sum()),
                "avg_trade_pnl": float(sub_pnls.mean()) if len(sub_pnls) else 0.0,
                "win_rate": float((sub_pnls > 0).mean()) if len(sub_pnls) else 0.0,
                "sharpe": sharpe_ratio(sub_returns, risk_free_rate),
                "n_trades": int(len(sub_pnls)),
            }
        )
    return pd.DataFrame(rows).sort_values("pct_time_in", ascending=False).reset_index(drop=True)


def confidence_bucketed_performance(trades: pd.DataFrame, pnls: pd.Series) -> pd.DataFrame:
    """Do higher-confidence regime calls actually perform better? If so, the HMM adds value."""
    if trades.empty:
        return pd.DataFrame(columns=["confidence_bucket", "trades", "win_rate", "avg_pnl", "sharpe"])

    trades_by_date = trades.set_index("fill_date")["regime_confidence"]

    rows = []
    for lo, hi, name in CONFIDENCE_BUCKETS:
        bucket_dates = trades_by_date.index[(trades_by_date > lo) & (trades_by_date <= hi)]
        bucket_pnls = pnls.reindex(bucket_dates).dropna()
        if bucket_pnls.empty:
            rows.append({"confidence_bucket": name, "trades": 0, "win_rate": 0.0, "avg_pnl": 0.0, "sharpe": 0.0})
            continue
        wins = bucket_pnls[bucket_pnls > 0]
        std = bucket_pnls.std(ddof=0)
        # Trade-level "Sharpe": mean/std of per-trade P&L, scaled by sqrt(n) —
        # a rough risk-adjusted score, not an annualized Sharpe (trades don't
        # occur at a fixed daily frequency).
        pseudo_sharpe = float(bucket_pnls.mean() / std * np.sqrt(len(bucket_pnls))) if std else 0.0
        rows.append(
            {
                "confidence_bucket": name,
                "trades": int(len(bucket_pnls)),
                "win_rate": float(len(wins) / len(bucket_pnls)),
                "avg_pnl": float(bucket_pnls.mean()),
                "sharpe": pseudo_sharpe,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Worst-case stats
# ---------------------------------------------------------------------------

def worst_case_stats(equity_curve: pd.Series) -> dict:
    daily = daily_returns(equity_curve)
    weekly = equity_curve.groupby(equity_curve.index.to_period("W")).last().pct_change().dropna()
    monthly = equity_curve.groupby(equity_curve.index.to_period("M")).last().pct_change().dropna()

    loss_streak = current = 0
    for r in daily:
        if r < 0:
            current += 1
            loss_streak = max(loss_streak, current)
        else:
            current = 0

    return {
        "worst_day": float(daily.min()) if len(daily) else 0.0,
        "worst_week": float(weekly.min()) if len(weekly) else 0.0,
        "worst_month": float(monthly.min()) if len(monthly) else 0.0,
        "max_consecutive_loss_days": int(loss_streak),
        "longest_underwater_days": max_drawdown_duration(equity_curve),
    }


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------

def buy_and_hold_equity(bars: pd.DataFrame, initial_capital: float) -> pd.Series:
    prices = bars["close"]
    shares = initial_capital / prices.iloc[0]
    return (shares * prices).rename("buy_and_hold")


def sma_trend_equity(
    bars: pd.DataFrame, initial_capital: float, sma_window: int = 200, slippage_pct: float = 0.0
) -> pd.Series:
    """Long whenever price is above its SMA(sma_window), flat (cash) otherwise."""
    close = bars["close"]
    sma = close.rolling(sma_window).mean()
    want_long = (close > sma).fillna(False)

    cash, shares, in_position = initial_capital, 0, False
    equity_values = []
    for date, price in close.items():
        if want_long.loc[date] and not in_position:
            fill_price = price * (1 + slippage_pct)
            shares = int(cash / fill_price)
            cash -= shares * fill_price
            in_position = True
        elif not want_long.loc[date] and in_position:
            fill_price = price * (1 - slippage_pct)
            cash += shares * fill_price
            shares = 0
            in_position = False
        equity_values.append(cash + shares * price)
    return pd.Series(equity_values, index=close.index, name="sma_200_trend")


def random_allocation_benchmark(
    bars: pd.DataFrame,
    initial_capital: float,
    rebalance_dates: list,
    allocation_choices: list,
    slippage_pct: float,
    n_seeds: int = 100,
) -> dict:
    """Random allocation changes at the SAME dates/frequency as the real
    strategy, drawn from the same observed allocation values. Isolates
    whether the HMM's TIMING adds value, holding sizing rules constant.
    """
    close = bars["close"]
    valid_dates = [d for d in rebalance_dates if d in close.index]

    final_equities, sharpes = [], []
    base_rng = np.random.default_rng(0)

    for _ in range(n_seeds):
        rng = np.random.default_rng(base_rng.integers(0, 2**32 - 1))
        cash, shares, next_idx = initial_capital, 0, 0
        equity_values = []
        for date, price in close.items():
            if next_idx < len(valid_dates) and date == valid_dates[next_idx]:
                target = float(rng.choice(allocation_choices))
                equity = cash + shares * price
                target_shares = int(equity * target / price)
                delta = target_shares - shares
                fill_price = price * (1 + slippage_pct) if delta > 0 else price * (1 - slippage_pct) if delta < 0 else price
                cash -= delta * fill_price
                shares = target_shares
                next_idx += 1
            equity_values.append(cash + shares * price)

        equity_curve = pd.Series(equity_values, index=close.index)
        final_equities.append(float(equity_curve.iloc[-1]))
        sharpes.append(sharpe_ratio(daily_returns(equity_curve), 0.0))

    return {
        "n_seeds": n_seeds,
        "mean_final_equity": float(np.mean(final_equities)),
        "std_final_equity": float(np.std(final_equities)),
        "mean_sharpe": float(np.mean(sharpes)),
        "std_sharpe": float(np.std(sharpes)),
    }


def benchmark_comparison(strategy_equity: pd.Series, benchmark_equity: pd.Series, risk_free_rate: float) -> dict:
    """Side-by-side headline stats for the strategy vs. one benchmark equity curve."""
    strat_returns = daily_returns(strategy_equity)
    bench_returns = daily_returns(benchmark_equity)
    return {
        "strategy_total_return": total_return(strategy_equity),
        "benchmark_total_return": total_return(benchmark_equity),
        "strategy_cagr": cagr(strategy_equity),
        "benchmark_cagr": cagr(benchmark_equity),
        "strategy_sharpe": sharpe_ratio(strat_returns, risk_free_rate),
        "benchmark_sharpe": sharpe_ratio(bench_returns, risk_free_rate),
        "strategy_max_drawdown": max_drawdown(strategy_equity),
        "benchmark_max_drawdown": max_drawdown(benchmark_equity),
    }
