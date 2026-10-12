"""Entry point for regime-trader.

    python main.py backtest --symbols SPY --start 2019-01-01 --end 2024-12-31
    python main.py backtest --symbols SPY --start 2019-01-01 --end 2024-12-31 --compare
    python main.py backtest --symbols SPY QQQ --stress-test

    python main.py run                 # live/paper trading loop (needs real Alpaca keys in .env)
    python main.py run --dry-run       # full pipeline, no orders submitted
    python main.py run --train-only    # train the HMM and exit
    python main.py run --dashboard     # show the last saved state snapshot
    python main.py run --web-dashboard # feed dashboard_state.json for `streamlit run dashboard_app.py`

    python main.py research --symbol GOOGL [--question "..."] [--lookback-days 7] [--force-refresh]
                                    # Phase 10: grounded Gemini research (read-only, places no trades)

`run` requires a real Alpaca account (paper is fine, and is the default —
see config/settings.yaml's broker.paper_trading) and so cannot be
exercised without one; core.trading_bot.TradingBot, which it wires
together, is fully unit tested against mocked clients instead (see
tests/test_trading_bot.py).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import pandas as pd
import yaml
import yfinance as yf
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

load_dotenv()  # populate os.environ from .env (ALPACA_API_KEY etc.) before anything reads it

from backtest.backtester import Backtester, BacktestResult
from backtest.performance import (
    buy_and_hold_equity,
    cagr,
    calmar_ratio,
    confidence_bucketed_performance,
    daily_returns,
    max_drawdown,
    max_drawdown_duration,
    random_allocation_benchmark,
    regime_performance_breakdown,
    sharpe_ratio,
    sma_trend_equity,
    sortino_ratio,
    total_return,
    trade_pnls,
    trade_stats,
    worst_case_stats,
)
from backtest.stress_test import run_crash_stress_test, run_gap_stress_test, run_regime_shuffle_test
from data.feature_engineering import compute_features, log_returns

console = Console(width=180)


def load_config(path: str = "config/settings.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def fetch_bars(symbol: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(symbol, start=start, end=end, interval="1d", progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [str(c).lower() for c in df.columns]
    return df[["open", "high", "low", "close", "volume"]].dropna()


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_core_metrics(equity_curve: pd.Series, trades: pd.DataFrame, risk_free_rate: float, title: str) -> None:
    returns = daily_returns(equity_curve)
    stats = trade_stats(trades, equity_curve)
    worst = worst_case_stats(equity_curve)

    table = Table(title=title)
    table.add_column("metric")
    table.add_column("value")
    table.add_row("Total return", f"{total_return(equity_curve):.2%}")
    table.add_row("CAGR", f"{cagr(equity_curve):.2%}")
    table.add_row("Sharpe", f"{sharpe_ratio(returns, risk_free_rate):.2f}")
    table.add_row("Sortino", f"{sortino_ratio(returns, risk_free_rate):.2f}")
    table.add_row("Calmar", f"{calmar_ratio(equity_curve):.2f}")
    table.add_row("Max drawdown", f"{max_drawdown(equity_curve):.2%}")
    table.add_row("Max DD duration (days)", str(max_drawdown_duration(equity_curve)))
    table.add_row("Total trades", str(stats["total_trades"]))
    table.add_row("Win rate", f"{stats['win_rate']:.2%}")
    table.add_row("Avg win / avg loss", f"{stats['avg_win']:.2%} / {stats['avg_loss']:.2%}")
    table.add_row("Profit factor", f"{stats['profit_factor']:.2f}")
    table.add_row("Avg holding period (days)", f"{stats['avg_holding_period_days']:.1f}")
    table.add_row("Worst day / week / month", f"{worst['worst_day']:.2%} / {worst['worst_week']:.2%} / {worst['worst_month']:.2%}")
    table.add_row("Max consecutive loss days", str(worst["max_consecutive_loss_days"]))
    table.add_row("Longest time underwater (days)", str(worst["longest_underwater_days"]))
    console.print(table)


def print_symbol_breakdown(symbol: str, result: BacktestResult, risk_free_rate: float) -> None:
    equity = result.per_symbol_equity[symbol]
    returns = daily_returns(equity)
    regime_df = result.regime_history[result.regime_history["symbol"] == symbol]
    trades = result.trades[result.trades["symbol"] == symbol] if not result.trades.empty else result.trades

    if regime_df.empty:
        return

    pnls = trade_pnls(trades, equity)
    regime_labels = regime_df.set_index("date")["regime_label"]

    breakdown = regime_performance_breakdown(returns, regime_labels, trades, pnls, risk_free_rate)
    rtable = Table(title=f"{symbol}: performance by regime")
    for col in breakdown.columns:
        rtable.add_column(col)
    for _, row in breakdown.iterrows():
        rtable.add_row(*(f"{v:.2%}" if isinstance(v, float) and col != "sharpe" else f"{v:.2f}" if isinstance(v, float) else str(v)
                          for col, v in zip(breakdown.columns, row)))
    console.print(rtable)

    conf = confidence_bucketed_performance(trades, pnls)
    ctable = Table(title=f"{symbol}: performance by confidence bucket")
    for col in conf.columns:
        ctable.add_column(col)
    for _, row in conf.iterrows():
        ctable.add_row(*(str(v) for v in row))
    console.print(ctable)


def run_comparison(data: dict[str, pd.DataFrame], result: BacktestResult, config: dict) -> None:
    rf = config["backtest"]["risk_free_rate"]
    slippage = config["backtest"]["slippage_pct"]
    rows = []

    for symbol, bars in data.items():
        strategy_equity = result.per_symbol_equity[symbol]
        bars_aligned = bars.loc[strategy_equity.index[0]:]
        capital = strategy_equity.iloc[0]

        bh_equity = buy_and_hold_equity(bars_aligned, capital)
        sma_equity = sma_trend_equity(bars_aligned, capital, sma_window=200, slippage_pct=slippage)

        rebalance_dates = result.trades.loc[result.trades["symbol"] == symbol, "fill_date"].tolist()
        allocation_choices = sorted(result.regime_history.loc[result.regime_history["symbol"] == symbol, "target_allocation"].unique())
        random_stats = random_allocation_benchmark(
            bars_aligned, capital, rebalance_dates, allocation_choices or [0.0, 0.6, 0.95], slippage, n_seeds=100
        )

        rows.append({
            "symbol": symbol,
            "strategy_return": total_return(strategy_equity),
            "strategy_sharpe": sharpe_ratio(daily_returns(strategy_equity), rf),
            "buy_hold_return": total_return(bh_equity),
            "buy_hold_sharpe": sharpe_ratio(daily_returns(bh_equity), rf),
            "sma200_return": total_return(sma_equity),
            "sma200_sharpe": sharpe_ratio(daily_returns(sma_equity), rf),
            "random_mean_return": random_stats["mean_final_equity"] / capital - 1,
            "random_mean_sharpe": random_stats["mean_sharpe"],
        })

    comparison_df = pd.DataFrame(rows)
    table = Table(title="Benchmark comparison")
    for col in comparison_df.columns:
        table.add_column(col)
    for _, row in comparison_df.iterrows():
        table.add_row(*(f"{v:.2%}" if isinstance(v, float) and "return" in col else f"{v:.2f}" if isinstance(v, float) else str(v)
                         for col, v in zip(comparison_df.columns, row)))
    console.print(table)
    comparison_df.to_csv("backtest_results/benchmark_comparison.csv", index=False)


def save_outputs(result: BacktestResult, out_dir: str = "backtest_results") -> None:
    import os
    os.makedirs(out_dir, exist_ok=True)
    result.equity_curve.to_csv(f"{out_dir}/equity_curve.csv", header=["equity"])
    result.trades.to_csv(f"{out_dir}/trade_log.csv", index=False)
    result.regime_history.to_csv(f"{out_dir}/regime_history.csv", index=False)
    # One column per symbol, so dashboard_app.py can render the per-symbol
    # breakdown (print_symbol_breakdown's tables) without re-running the backtest.
    per_symbol_df = pd.concat(result.per_symbol_equity, axis=1)
    per_symbol_df.to_csv(f"{out_dir}/per_symbol_equity.csv")
    console.print(f"\n  Saved equity_curve.csv, trade_log.csv, regime_history.csv, per_symbol_equity.csv -> {out_dir}/")


# ---------------------------------------------------------------------------
# Subcommand: backtest
# ---------------------------------------------------------------------------

def cmd_backtest(args: argparse.Namespace) -> None:
    config = load_config()
    data = {symbol: fetch_bars(symbol, args.start, args.end) for symbol in args.symbols}

    if args.stress_test:
        console.rule("[bold]Stress testing[/bold] (this retrains the HMM on every simulation — slow by design)")
        crash = run_crash_stress_test(data, config, n_simulations=args.n_simulations)
        console.print(f"  Crash injection ({crash['n_simulations']} sims): "
                      f"mean max loss {crash['mean_max_loss']:.2%}, worst case {crash['worst_case_loss']:.2%}, "
                      f"circuit-breaker-level drawdown in {crash['pct_circuit_breaker_fired']:.0%} of sims")

        gap = run_gap_stress_test(data, config, n_simulations=args.n_simulations)
        console.print(f"  Gap risk ({gap['n_simulations']} sims): baseline max DD {gap['baseline_max_drawdown']:.2%}, "
                      f"mean perturbed max DD {gap['mean_perturbed_max_drawdown']:.2%}, "
                      f"mean incremental loss {gap['mean_incremental_loss']:.2%}")

        base_result = Backtester(config).run(data)
        shuffle = run_regime_shuffle_test(data, base_result, config, n_shuffles=max(args.n_simulations // 2, 5))
        console.print(f"  Regime shuffle ({shuffle['n_shuffles']} shuffles): real max DD {shuffle['real_max_drawdown']:.2%}, "
                      f"worst shuffled max DD {shuffle['worst_shuffled_max_drawdown']:.2%}, "
                      f"contained={shuffle['contained']}")
        return

    console.rule("[bold]Running walk-forward backtest[/bold]")
    result = Backtester(config).run(data)

    print_core_metrics(result.equity_curve, result.trades, config["backtest"]["risk_free_rate"], "Portfolio performance")
    for symbol in data:
        print_symbol_breakdown(symbol, result, config["backtest"]["risk_free_rate"])

    save_outputs(result)

    if args.compare:
        console.rule("[bold]Benchmark comparison[/bold]")
        run_comparison(data, result, config)


def cmd_run(args: argparse.Namespace) -> None:
    """Phase 7's live/paper trading entry point.

    Requires real ALPACA_API_KEY/ALPACA_SECRET_KEY in .env — there is no
    way to smoke-test this path without a real (paper) Alpaca account, so
    unlike every other command in this file it has not been exercised here.
    core.trading_bot.TradingBot, which this wires together, IS fully unit
    tested against mocked clients (see tests/test_trading_bot.py).
    """
    import pickle

    from broker.alpaca_client import AlpacaAuthError, AlpacaClient, AlpacaConnectionError
    from core.hmm_engine import HMMRegimeEngine
    from core.trading_bot import DEFAULT_MODEL_PATH, TradingBot
    from data.market_data import MarketDataFeed
    from monitoring.logger import get_alerts_logger, get_main_logger, get_regime_logger, get_trades_logger

    config = load_config()

    if args.dashboard:
        _show_last_snapshot()
        return

    # The live dashboard (rich.live.Live) and scrolling console log lines
    # fight over the same terminal region, so console echo is off whenever
    # --live-dashboard is active; file logging (logs/*.log) stays on either way.
    get_main_logger(console=not args.live_dashboard)
    get_trades_logger()
    get_alerts_logger()
    get_regime_logger()

    try:
        alpaca_client = AlpacaClient(paper=config["broker"]["paper_trading"])
        market_data = MarketDataFeed()
    except (ValueError, AlpacaAuthError, AlpacaConnectionError) as exc:
        console.print(f"[red]Could not connect to Alpaca:[/red] {exc}")
        console.print("Check ALPACA_API_KEY / ALPACA_SECRET_KEY in .env (see .env.example) and try again.")
        sys.exit(1)

    hmm_engine = None
    if os.path.exists(DEFAULT_MODEL_PATH):
        try:
            hmm_engine = HMMRegimeEngine.load(DEFAULT_MODEL_PATH)
        except (OSError, pickle.PickleError) as exc:
            console.print(f"[yellow]Could not load {DEFAULT_MODEL_PATH}: {exc} -- will train fresh[/yellow]")

    bot = TradingBot(config, alpaca_client, market_data, hmm_engine=hmm_engine, dry_run=args.dry_run)

    primary_symbol = config["broker"]["symbols"][0]
    bars = market_data.get_historical_bars(primary_symbol, timeframe="1Day", limit=2000)
    features = compute_features(bars)
    returns = log_returns(bars["close"], 1)

    if args.train_only:
        bot.train_hmm(features, returns)
        console.print(f"Trained HMM on {primary_symbol}: n_regimes={bot.hmm_engine.n_regimes}")
        return

    bot.startup(features_by_symbol={primary_symbol: features}, returns_by_symbol={primary_symbol: returns})

    if args.live_dashboard:
        bot.start_dashboard()
        console.print("[dim]Live dashboard active; console logging is suppressed -- tail logs/main.log for text logs.[/dim]")

    if args.web_dashboard:
        bot.start_json_dashboard_feed()
        console.print(
            "[dim]Writing dashboard_state.json for the Streamlit UI -- "
            "run `streamlit run dashboard_app.py` in another terminal.[/dim]"
        )

    bot.run(config["broker"]["symbols"], timeframe=config["broker"]["timeframe"])


def cmd_research(args: argparse.Namespace) -> None:
    """Phase 10: grounded Gemini research. Read-only — places no trades and
    touches no strategy/risk/HMM configuration."""
    from monitoring.logger import get_research_logger
    from research.config import load_research_config
    from research.orchestrator import ResearchOrchestrator
    from research.presentation import RELATIONSHIP_LABELS, STANCE_LABELS, age_label, quant_rows
    from research.schemas import ResearchRequest

    get_research_logger()
    env = dict(os.environ)
    if args.provider:
        env["RESEARCH_PROVIDER"] = args.provider
    if args.model:
        env["RESEARCH_MODEL"] = args.model
    if args.search:
        env["RESEARCH_SEARCH"] = args.search
    config = load_research_config(env=env)

    if args.check:
        from research.checks import run_checks

        failures = 0
        for result in run_checks(config):
            mark = "[green]ok[/green]  " if result.ok else "[red]FAIL[/red]"
            console.print(f"{mark} {result.name}: {result.detail}")
            failures += 0 if result.ok else 1
        sys.exit(1 if failures else 0)

    if args.list_models:
        from research.exceptions import ResearchError
        from research.llm import build_llm

        try:
            for name in build_llm(config).list_models():
                console.print(name)
        except ResearchError as exc:
            console.print(f"[red]{exc.user_message}[/red] {exc.detail}")
            sys.exit(1)
        return

    if args.usage:
        from research.usage import get_usage_summary, usage_lines

        for line in usage_lines(config, get_usage_summary(config)):
            console.print(f"- {line}", markup=False)
        return

    if not args.symbol:
        console.print("[red]--symbol is required[/red]")
        sys.exit(2)

    request = ResearchRequest(
        symbol=args.symbol, question=args.question,
        lookback_days=args.lookback_days or config.default_lookback_days,
        force_refresh=args.force_refresh, allow_without_quant=args.allow_without_quant,
    )
    outcome = ResearchOrchestrator(config).research(request)

    if outcome.quant_context is not None:
        qt = Table(title=f"{request.symbol}: quantitative snapshot (HMM)")
        qt.add_column("metric")
        qt.add_column("value")
        for label, value in quant_rows(outcome.quant_context):
            qt.add_row(label, value)
        console.print(qt)

    if not outcome.ok:
        console.print(f"[red]Research unavailable[/red] ({outcome.error.code}): {outcome.error.message}")
        console.print("[dim]The quantitative system is unaffected; no orders were placed.[/dim]")
        sys.exit(1)

    result = outcome.result
    brief, comparison = result.brief, result.comparison
    console.rule(f"[bold]Research: {brief.symbol}[/bold]")
    console.print(f"[bold]Question:[/bold] {brief.question}")
    console.print(f"[bold]Summary:[/bold] {brief.summary}")
    console.print(
        f"[bold]Evidence:[/bold] {STANCE_LABELS[brief.evidence_stance]} · confidence {brief.overall_confidence:.0%} · "
        f"{brief.source_count} sources"
    )
    for title, items in (("Catalysts", brief.catalysts), ("Risks", brief.risks), ("Recent events", brief.recent_events)):
        if items:
            console.print(f"\n[bold]{title}[/bold]")
            for e in items:
                console.print(f"  - [{e.category}/{e.impact}] {e.claim}\n      {e.source_title}: {e.source_url}")
    for title, items in (("Conflicting evidence", brief.conflicting_evidence), ("Unknowns", brief.unknowns)):
        if items:
            console.print(f"\n[bold]{title}[/bold]")
            for item in items:
                console.print(f"  - {item}")

    console.rule("[bold]Quant vs research[/bold]")
    console.print(f"[bold]Relationship:[/bold] {RELATIONSHIP_LABELS[comparison.relationship]}")
    console.print(comparison.explanation)
    console.print(f"[bold]Human review recommended:[/bold] {comparison.human_review_recommended}")

    console.rule("[bold]Sources[/bold]")
    seen: set[str] = set()
    for e in brief.catalysts + brief.risks + brief.recent_events:
        if e.source_url not in seen:
            seen.add(e.source_url)
            console.print(f"  {e.source_title}: {e.source_url}")

    status = "CACHE HIT" if result.cache_hit else "fresh"
    u = result.diagnostics.usage
    console.print(
        "[dim]Usage: free (cache hit)[/dim]" if result.cache_hit else
        f"[dim]Usage: {u.grounded_searches} grounded searches, {u.input_tokens:,} in / {u.output_tokens:,} out tokens, est. ≈${u.estimated_cost_usd:.3f}"
        + (f", {u.finance_searches} SerpApi Google Finance search" if u.finance_searches else "") + "[/dim]"
    )
    console.print(
        f"\n[dim]{status} · generated {age_label(brief.generated_at)} ({brief.generated_at:%Y-%m-%d %H:%M} UTC) · "
        f"{result.latency_ms} ms · {result.diagnostics.provider}/{result.diagnostics.model} · search {result.diagnostics.search_provider} · prompt v{result.diagnostics.prompt_version}. "
        "Informational only — not financial advice; no orders placed.[/dim]"
    )


def _quant_snapshot_or_none(symbol: str):
    """HMM snapshot via the read-only research tools; None (with a note) if unavailable."""
    from research.tools.hmm_tools import DEFAULT_MODEL_PATH, HMMRuntimeReader
    from research.exceptions import ToolError

    try:
        return HMMRuntimeReader(DEFAULT_MODEL_PATH).get_hmm_snapshot(symbol)
    except ToolError as exc:
        console.print(f"[yellow]HMM snapshot unavailable: {exc.detail}[/yellow]")
        return None


def cmd_research_prompt(args: argparse.Namespace) -> None:
    """No-API workflow, step 1: print (and copy) a prompt to paste into Google Finance Deep Search."""
    import subprocess

    from research.config import load_research_config
    from research.manual import build_deep_search_prompt
    from research.schemas import ResearchRequest

    request = ResearchRequest(
        symbol=args.symbol, question=args.question,
        lookback_days=args.lookback_days or load_research_config().default_lookback_days,
    )
    prompt = build_deep_search_prompt(request.symbol, request.question, request.lookback_days)
    console.print(prompt, markup=False, highlight=False)
    try:
        subprocess.run(["pbcopy"], input=prompt.encode(), check=True)
        console.print("\n[dim]Copied to clipboard. Paste it into Google Finance Deep Search, then run `research-import`.[/dim]")
    except (OSError, subprocess.CalledProcessError):
        console.print("\n[dim]Copy the text above into Google Finance Deep Search, then run `research-import`.[/dim]")


def cmd_research_import(args: argparse.Namespace) -> None:
    """No-API workflow, step 2: save the report you copied from Google Finance, with the HMM snapshot."""
    import subprocess

    from research.config import load_research_config
    from research.manual import extract_domains, save_manual_report
    from research.prompts import default_question
    from research.schemas import ResearchRequest

    request = ResearchRequest(symbol=args.symbol, question=args.question, lookback_days=args.lookback_days or load_research_config().default_lookback_days)
    if args.file:
        with open(args.file) as f:
            report = f.read()
    else:  # default: clipboard
        try:
            report = subprocess.run(["pbpaste"], capture_output=True, check=True).stdout.decode()
        except (OSError, subprocess.CalledProcessError):
            console.print("[red]Could not read the clipboard; pass --file report.txt instead.[/red]")
            sys.exit(1)
    if not report.strip():
        console.print("[red]Report is empty.[/red]")
        sys.exit(1)

    quant = _quant_snapshot_or_none(request.symbol)
    question = request.question or default_question(request.symbol, request.lookback_days)
    path = save_manual_report(request.symbol, question, report, quant)
    console.print(f"Saved briefing: {path}")
    if quant is not None:
        from research.prompts import summarize_quant_line

        console.print(f"[bold]HMM:[/bold] {summarize_quant_line(quant)}")
    domains = extract_domains(report)
    console.print(f"[bold]Source domains found in report:[/bold] {', '.join(domains) or '(none — ask Deep Search for sources)'}")
    if args.compare:
        _compare_pasted_report(request, report, path, quant)
    else:
        console.print("[dim]Informational only. Add --compare to compare it with this system's own research.[/dim]")


def _compare_pasted_report(request, report: str, saved_path, quant) -> None:
    """Run the agent's own research (cache-aware) and compare it with the pasted report."""
    from research.config import load_research_config
    from research.exceptions import ResearchError
    from research.llm.base import Deadline
    from research.orchestrator import ResearchOrchestrator
    from research.reference import compare_with_reference, format_markdown

    config = load_research_config()
    orchestrator = ResearchOrchestrator(config)
    outcome = orchestrator.research(request)
    if not outcome.ok:
        console.print(f"[red]Could not run the agent's research for the comparison[/red] ({outcome.error.code}): {outcome.error.message}")
        return
    try:
        comparison = compare_with_reference(
            orchestrator.get_llm(), outcome.result.quant_context, outcome.result.brief, report,
            Deadline(config.overall_timeout_seconds), outcome.request_id,
        )
    except ResearchError as exc:
        console.print(f"[red]Comparison failed[/red] ({exc.code}): {exc.message}")
        return
    markdown = format_markdown(comparison)
    with open(saved_path, "a") as f:
        f.write(markdown)
    console.print(markdown, markup=False, highlight=False)
    console.print(f"[dim]Appended to {saved_path}[/dim]")


def _show_last_snapshot() -> None:
    """Best-effort `--dashboard`: there is no live IPC between a running bot
    and this command, so this renders the LAST SAVED state_snapshot.json
    rather than a truly live-attached view.
    """
    from core.trading_bot import DEFAULT_STATE_PATH

    if not os.path.exists(DEFAULT_STATE_PATH):
        console.print(f"No {DEFAULT_STATE_PATH} found -- has an instance run and saved state yet?")
        return
    with open(DEFAULT_STATE_PATH) as f:
        snapshot = json.load(f)
    console.print(
        "[yellow]Note: this is the LAST SAVED snapshot (from shutdown or an error), "
        "not a live-attached view of a running instance.[/yellow]"
    )
    console.print(f"Saved at: {snapshot.get('saved_at')}")
    console.print_json(data=snapshot)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="main.py")
    subparsers = parser.add_subparsers(dest="command")

    backtest_parser = subparsers.add_parser("backtest", help="Run the walk-forward backtester")
    backtest_parser.add_argument("--symbols", nargs="+", default=["SPY"])
    backtest_parser.add_argument("--start", default="2019-01-01")
    backtest_parser.add_argument("--end", default="2024-12-31")
    backtest_parser.add_argument("--compare", action="store_true")
    backtest_parser.add_argument("--stress-test", action="store_true")
    backtest_parser.add_argument("--n-simulations", type=int, default=100)
    backtest_parser.set_defaults(func=cmd_backtest)

    run_parser = subparsers.add_parser("run", help="Run the live/paper trading loop (Phase 7)")
    run_parser.add_argument("--dry-run", action="store_true", help="Full pipeline, no orders submitted")
    run_parser.add_argument("--train-only", action="store_true", help="Train the HMM and exit")
    run_parser.add_argument("--dashboard", action="store_true", help="Show the last saved state snapshot and exit")
    run_parser.add_argument(
        "--live-dashboard", action="store_true",
        help="Show the live auto-refreshing dashboard UI in this terminal while running",
    )
    run_parser.add_argument(
        "--web-dashboard", action="store_true",
        help="Continuously write dashboard_state.json for the Streamlit UI (see `streamlit run dashboard_app.py`)",
    )
    run_parser.set_defaults(func=cmd_run)

    research_parser = subparsers.add_parser(
        "research", help="Grounded Gemini market research (Phase 10; read-only, places no trades)"
    )
    research_parser.add_argument("--symbol", help="Ticker to research, e.g. GOOGL")
    research_parser.add_argument("--question", help="Optional research question")
    research_parser.add_argument("--lookback-days", type=int, help="Research horizon in days (default: research.default_lookback_days)")
    research_parser.add_argument("--force-refresh", action="store_true", help="Bypass the result cache")
    research_parser.add_argument(
        "--allow-without-quant", action="store_true", help="Still research if no HMM snapshot can be loaded"
    )
    research_parser.add_argument("--provider", choices=["ollama", "openai", "anthropic", "gemini"], help="Override research.provider")
    research_parser.add_argument("--model", help="Override the model id for the chosen provider")
    research_parser.add_argument("--search", choices=["auto", "searxng", "gemini_grounding"], help="Override research.search_provider")
    research_parser.add_argument("--check", action="store_true", help="Probe the configured LLM + search backends and exit")
    research_parser.add_argument("--usage", action="store_true", help="Show this month's Gemini usage, free-tier limits and estimated spend, then exit")
    research_parser.add_argument("--list-models", action="store_true", help="List Gemini models your API key can use and exit")
    research_parser.set_defaults(func=cmd_research)

    prompt_parser = subparsers.add_parser(
        "research-prompt", help="No-API: print/copy a prompt to paste into Google Finance Deep Search by hand"
    )
    prompt_parser.add_argument("--symbol", required=True)
    prompt_parser.add_argument("--question")
    prompt_parser.add_argument("--lookback-days", type=int)
    prompt_parser.set_defaults(func=cmd_research_prompt)

    import_parser = subparsers.add_parser(
        "research-import", help="No-API: save a Google Finance report (clipboard or --file) with the HMM snapshot"
    )
    import_parser.add_argument("--symbol", required=True)
    import_parser.add_argument("--question")
    import_parser.add_argument("--lookback-days", type=int)
    import_parser.add_argument("--file", help="Read the report from a file instead of the clipboard")
    import_parser.add_argument(
        "--compare", action="store_true",
        help="Also run this system's own research and compare it with the pasted report (uses the configured provider)",
    )
    import_parser.set_defaults(func=cmd_research_import)

    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    if not getattr(args, "command", None):
        parser.print_help()
        sys.exit(1)
    args.func(args)


if __name__ == "__main__":
    main()
