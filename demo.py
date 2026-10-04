"""Interactive, end-to-end demo of everything built so far (Phases 1-3).

Fetches REAL historical daily bars (via yfinance — no API key needed),
trains the HMM volatility classifier, walks forward bar-by-bar exactly like
the live bot would (so you can see the stability filter and no-look-ahead
inference in action), and prints the resulting strategy signals.

Usage:
    python demo.py                      # defaults: regime symbol=SPY, 8y history
    python demo.py SPY 10y
    python demo.py QQQ 5y

Try editing HMM_CONFIG / STRATEGY_CONFIG below, or config/settings.yaml, and
re-running — that's the easiest way to build intuition for what each knob
does.
"""

from __future__ import annotations

import sys

import pandas as pd
import yaml
import yfinance as yf
from rich.console import Console
from rich.table import Table

from core.hmm_engine import HMMRegimeEngine
from core.regime_strategies import StrategyOrchestrator
from data.feature_engineering import compute_features, log_returns

console = Console(width=180)

# Extra symbols to run the SAME regime call through, to show that one
# regime read drives sizing across a whole basket of names (each symbol
# still uses its OWN price/ATR/EMA for the actual stop and trend check).
BASKET_SYMBOLS = ["SPY", "QQQ", "TSLA"]

# Number of most-recent valid feature rows to "replay" bar-by-bar at the end.
WALK_FORWARD_BARS = 180


def load_config(path: str = "config/settings.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def fetch_ohlcv(symbol: str, period: str) -> pd.DataFrame:
    df = yf.download(symbol, period=period, interval="1d", progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [str(c).lower() for c in df.columns]
    return df[["open", "high", "low", "close", "volume"]].dropna()


def print_bic_table(engine: HMMRegimeEngine) -> None:
    table = Table(title="Model selection: BIC per candidate n_regimes (lower = better)")
    table.add_column("n_regimes")
    table.add_column("BIC")
    table.add_column("selected?")
    for k, bic in sorted(engine.bic_scores.items()):
        selected = "← SELECTED" if k == engine.n_regimes else ""
        table.add_row(str(k), f"{bic:,.1f}", selected)
    console.print(table)


def print_regime_table(engine: HMMRegimeEngine, orchestrator: StrategyOrchestrator) -> None:
    table = Table(title=f"Trained regimes (n_regimes={engine.n_regimes})")
    table.add_column("state_id")
    table.add_column("label (by return)")
    table.add_column("vol_rank")
    table.add_column("expected_return")
    table.add_column("expected_volatility")
    table.add_column("strategy assigned")
    table.add_column("min_confidence_to_act")

    for state in sorted(engine.regime_info):
        info = engine.regime_info[state]
        strategy = orchestrator.strategy_by_state[state].__class__.__name__
        table.add_row(
            str(state),
            engine.regime_labels[state],
            str(orchestrator.vol_rank[state]),
            f"{info.expected_return:+.4%}",
            f"{info.expected_volatility:.4%}",
            strategy,
            f"{info.min_confidence_to_act:.2f}",
        )
    console.print(table)


def main() -> None:
    regime_symbol = sys.argv[1] if len(sys.argv) > 1 else "SPY"
    period = sys.argv[2] if len(sys.argv) > 2 else "8y"

    config = load_config()

    console.rule(f"[bold]Fetching data[/bold]")
    all_symbols = sorted(set([regime_symbol] + BASKET_SYMBOLS))
    bars_by_symbol: dict[str, pd.DataFrame] = {}
    for symbol in all_symbols:
        bars_by_symbol[symbol] = fetch_ohlcv(symbol, period)
        console.print(f"  {symbol}: {len(bars_by_symbol[symbol])} bars "
                      f"({bars_by_symbol[symbol].index[0].date()} → {bars_by_symbol[symbol].index[-1].date()})")

    console.rule("[bold]Feature engineering[/bold]")
    regime_bars = bars_by_symbol[regime_symbol]
    features = compute_features(regime_bars)
    raw_returns = log_returns(regime_bars["close"], 1)
    console.print(f"  {len(regime_bars)} raw bars → {len(features)} valid standardized feature rows "
                  f"(warmup consumed by rolling windows)")

    console.rule("[bold]Training the HMM (this does the BIC model-selection search)[/bold]")
    engine = HMMRegimeEngine(config["hmm"])
    engine.fit(features, raw_returns)
    print_bic_table(engine)

    orchestrator = StrategyOrchestrator(config["strategy"], engine.regime_info)
    print_regime_table(engine, orchestrator)

    console.rule(f"[bold]Walking forward the last {WALK_FORWARD_BARS} bars (no look-ahead)[/bold]")
    console.print(
        "  Each step below only sees data up to and including that day — "
        "exactly what predict_regime_filtered guarantees. Regime changes are only\n"
        "  logged once they've been CONFIRMED for hmm.stability_bars consecutive days."
    )

    walk_start = max(0, len(features) - WALK_FORWARD_BARS)
    change_log = []
    last_label = None
    for t in range(walk_start, len(features)):
        window = features.iloc[: t + 1]
        state = engine.update(window)
        if state.label != last_label and state.is_confirmed:
            change_log.append((window.index[-1].date(), state.label, state.probability))
            last_label = state.label

    changes_table = Table(title="Confirmed regime changes during the walk-forward window")
    changes_table.add_column("date")
    changes_table.add_column("new regime")
    changes_table.add_column("confidence")
    for date, label, prob in change_log:
        changes_table.add_row(str(date), label, f"{prob:.0%}")
    if not change_log:
        console.print("  (no confirmed regime change in this window — stayed in the same regime throughout)")
    else:
        console.print(changes_table)

    console.rule("[bold]Today's signals[/bold]")
    current_state = state  # last RegimeState from the walk-forward loop
    is_flickering = engine.is_flickering()
    console.print(
        f"  Current regime: [bold]{current_state.label}[/bold] "
        f"(state {current_state.state_id}), confidence={current_state.probability:.0%}, "
        f"confirmed={current_state.is_confirmed}, consecutive_bars={current_state.consecutive_bars}, "
        f"flickering={is_flickering}"
    )

    latest_date = features.index[-1]
    basket_bars_as_of = {
        symbol: bars.loc[:latest_date] for symbol, bars in bars_by_symbol.items() if symbol in BASKET_SYMBOLS
    }
    signals = orchestrator.generate_signals(BASKET_SYMBOLS, basket_bars_as_of, current_state, is_flickering)

    signal_table = Table(title="Signals for the basket, under today's regime")
    signal_table.add_column("symbol")
    signal_table.add_column("direction")
    signal_table.add_column("size %")
    signal_table.add_column("leverage")
    signal_table.add_column("entry")
    signal_table.add_column("stop")
    signal_table.add_column("strategy")
    for s in signals:
        signal_table.add_row(
            s.symbol, s.direction.value, f"{s.position_size_pct:.0%}", f"{s.leverage:.2f}x",
            f"${s.entry_price:,.2f}", f"${s.stop_loss:,.2f}", s.strategy_name,
        )
    console.print(signal_table)
    for s in signals:
        console.print(f"  [bold]{s.symbol}[/bold]: {s.reasoning}")

    console.rule("[bold]Try this next[/bold]")
    console.print(
        "  - Re-run with a different symbol/period: python demo.py QQQ 10y\n"
        "  - Edit config/settings.yaml (e.g. hmm.stability_bars, strategy.rebalance_threshold) and re-run\n"
        "  - Force uncertainty mode by temporarily raising hmm.min_confidence in settings.yaml\n"
        "  - Open a Python REPL and call engine.predict_regime_filtered(features.iloc[:N]) directly"
    )


if __name__ == "__main__":
    main()
