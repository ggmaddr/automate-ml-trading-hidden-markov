"""Phase 9 integration tests — the guide's five required scenarios:

  a. End-to-end dry run: data -> HMM -> strategy -> risk -> simulated orders
  b. Look-ahead bias: backtest identical with different end dates
  c. Risk stress: extreme signals capped, rapid-fire blocked, no-stop rejected
  d. Alpaca paper: place bracket order, modify stop, cancel, verify clean state
  e. Recovery: kill process, restart, verify state recovery and no double-entry

(d) needs a REAL Alpaca paper account and so can't run here — see
test_alpaca_paper_trading_requires_real_credentials below, and the
paper-trading checklist in README.md.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import MagicMock

import pandas as pd
import pytest

from backtest.backtester import Backtester
from broker.alpaca_client import AccountSnapshot
from broker.position_tracker import PositionTracker
from core.risk_manager import RiskManager
from core.trading_bot import TradingBot


def _account(equity=100_000.0, cash=50_000.0, buying_power=150_000.0) -> AccountSnapshot:
    return AccountSnapshot(
        equity=equity, cash=cash, buying_power=buying_power, portfolio_value=equity,
        pattern_day_trader=False, trading_blocked=False, account_blocked=False, daytrade_count=0,
    )


def _build_bot(fresh_trained_hmm, full_config, tmp_path, dry_run, order_executor=None, equity=100_000.0):
    mock_alpaca = MagicMock()
    mock_alpaca.paper = True
    mock_alpaca.get_account.return_value = _account(equity=equity)
    mock_alpaca.is_market_open.return_value = True
    mock_alpaca.get_positions.return_value = []

    position_tracker = PositionTracker(mock_alpaca)
    # Explicit lock_file_path: a PEAK_HALT in any test must never write
    # trading_halted.lock into the real project directory.
    risk_manager = RiskManager(full_config["risk"], lock_file_path=str(tmp_path / "halt.lock"))
    bot = TradingBot(
        full_config, alpaca_client=mock_alpaca, market_data=MagicMock(), hmm_engine=fresh_trained_hmm,
        risk_manager=risk_manager, position_tracker=position_tracker,
        order_executor=order_executor or MagicMock(), alert_manager=MagicMock(),
        dry_run=dry_run, state_path=str(tmp_path / "state.json"), model_path=str(tmp_path / "model.pkl"),
    )
    return bot, mock_alpaca


# ---------------------------------------------------------------------------
# (a) End-to-end dry run: data -> HMM -> strategy -> risk -> simulated orders
# ---------------------------------------------------------------------------

def test_end_to_end_dry_run_walks_many_bars_without_orders_or_crashes(fresh_trained_hmm, full_config, synthetic_ohlcv, tmp_path):
    mock_executor = MagicMock()
    bot, _ = _build_bot(fresh_trained_hmm, full_config, tmp_path, dry_run=True, order_executor=mock_executor)

    now = datetime(2024, 3, 4, 10, 0)
    outcomes = []
    for i in range(30, 0, -1):  # walk the last 30 bars, expanding window each time
        bars = synthetic_ohlcv.iloc[: len(synthetic_ohlcv) - i]
        decision = bot.process_bar("SPY", bars, now=now)
        outcomes.append(decision.approved)
        now += timedelta(minutes=5)

    # The full pipeline ran on every bar without raising, and never touched the broker.
    assert len(outcomes) == 30
    mock_executor.submit_bracket_order.assert_not_called()
    mock_executor.close_position.assert_not_called()
    # It must have reached a real decision (approved or a concrete rejection reason)
    # on at least some bars, not silently done nothing the whole time.
    assert any(outcomes) or len(bot.recent_signals) > 0


# ---------------------------------------------------------------------------
# (b) Look-ahead bias at the BACKTEST level: identical results regardless of
#     how much future data follows the window under test.
# ---------------------------------------------------------------------------

def test_backtest_identical_with_different_end_dates(synthetic_ohlcv, backtest_config):
    short_result = Backtester(backtest_config).run({"TEST": synthetic_ohlcv.iloc[:700]})
    long_result = Backtester(backtest_config).run({"TEST": synthetic_ohlcv})  # full 1000 bars

    overlap_dates = short_result.equity_curve.index
    aligned_long = long_result.equity_curve.reindex(overlap_dates)

    pd.testing.assert_series_equal(short_result.equity_curve, aligned_long, check_names=False)


# ---------------------------------------------------------------------------
# (c) Risk stress, wired through the real pipeline rather than RiskManager in isolation
# ---------------------------------------------------------------------------

def test_extreme_signal_gets_capped_through_full_pipeline(fresh_trained_hmm, full_config, synthetic_ohlcv, tmp_path):
    mock_executor = MagicMock()
    mock_executor.submit_bracket_order.return_value = MagicMock(trade_id="t1", status="filled")
    bot, _ = _build_bot(fresh_trained_hmm, full_config, tmp_path, dry_run=False, order_executor=mock_executor)

    # Monkeypatch the orchestrator's signal generation to simulate a strategy
    # bug/extreme regime output: 5x the normal size, 10x leverage.
    real_generate_signals = bot.orchestrator.generate_signals

    def _inflated_generate_signals(*args, **kwargs):
        signals = real_generate_signals(*args, **kwargs)
        for s in signals:
            s.position_size_pct = 5.0
            s.leverage = 10.0
        return signals

    bot.orchestrator.generate_signals = _inflated_generate_signals

    decision = bot.process_bar("SPY", synthetic_ohlcv)

    if decision.approved:
        final = decision.signal
        assert final.position_size_pct * final.leverage <= full_config["risk"]["max_leverage"] + 1e-9
        assert decision.modifications  # risk manager had to intervene
    else:
        # also acceptable: the inflated notional got rejected outright (e.g. buying power)
        assert decision.rejection_reason is not None


def test_rapid_fire_duplicate_signal_blocked(fresh_trained_hmm, full_config, synthetic_ohlcv, tmp_path):
    mock_executor = MagicMock()
    mock_executor.submit_bracket_order.return_value = MagicMock(trade_id="t1", status="filled")
    bot, _ = _build_bot(fresh_trained_hmm, full_config, tmp_path, dry_run=False, order_executor=mock_executor)

    now = datetime(2024, 3, 4, 10, 0)
    first = bot.process_bar("SPY", synthetic_ohlcv, now=now)
    if not first.approved:
        pytest.skip("risk manager rejected the first bar for this synthetic series/seed")

    # Force a SECOND distinct entry attempt (different symbol-state so it's
    # not just "hold") within the same 60s window, simulating a rapid-fire
    # duplicate submission (e.g. a retry bug, or two bars processed too fast).
    bot.position_tracker.positions.clear()  # as if the first entry hadn't registered yet
    second = bot.process_bar("SPY", synthetic_ohlcv, now=now + timedelta(seconds=5))

    assert second.approved is False
    assert "duplicate" in second.rejection_reason


def test_signal_without_stop_loss_rejected(fresh_trained_hmm, full_config, synthetic_ohlcv, tmp_path):
    bot, _ = _build_bot(fresh_trained_hmm, full_config, tmp_path, dry_run=True)

    real_generate_signals = bot.orchestrator.generate_signals

    def _no_stop_signals(*args, **kwargs):
        signals = real_generate_signals(*args, **kwargs)
        for s in signals:
            s.stop_loss = None
        return signals

    bot.orchestrator.generate_signals = _no_stop_signals

    decision = bot.process_bar("SPY", synthetic_ohlcv)

    assert decision.approved is False
    assert "stop_loss" in decision.rejection_reason


# ---------------------------------------------------------------------------
# (d) Alpaca paper trading — cannot run without a real account
# ---------------------------------------------------------------------------

@pytest.mark.skip(
    reason=(
        "Requires a real Alpaca PAPER account (ALPACA_API_KEY/ALPACA_SECRET_KEY in .env). "
        "See README.md's 'Paper-trading verification' checklist for the manual steps this "
        "stands in for: submit_bracket_order -> modify_stop (tighten only) -> close_position, "
        "then confirm get_positions()/get_order_history() show a clean, fully-closed state."
    )
)
def test_alpaca_paper_trading_requires_real_credentials():
    pass


# ---------------------------------------------------------------------------
# (e) Recovery: kill process, restart, verify state recovery and no double-entry
# ---------------------------------------------------------------------------

def test_recovery_restarts_cleanly_with_no_double_entry(fresh_trained_hmm, full_config, synthetic_ohlcv, tmp_path):
    mock_executor_1 = MagicMock()
    mock_executor_1.submit_bracket_order.return_value = MagicMock(trade_id="t1", status="filled")
    bot1, mock_alpaca_1 = _build_bot(fresh_trained_hmm, full_config, tmp_path, dry_run=False, order_executor=mock_executor_1)

    now = datetime(2024, 3, 4, 10, 0)
    first = bot1.process_bar("SPY", synthetic_ohlcv, now=now)
    if not first.approved:
        pytest.skip("risk manager rejected the first bar for this synthetic series/seed")
    assert mock_executor_1.submit_bracket_order.call_count == 1

    opened = bot1.position_tracker.get_position("SPY")
    bot1.save_state_snapshot()  # "process killed" right after this

    # --- Simulate a restart: a brand-new TradingBot, same state_path, whose
    # broker now independently reports the position that was already opened. ---
    mock_executor_2 = MagicMock()
    bot2, mock_alpaca_2 = _build_bot(fresh_trained_hmm, full_config, tmp_path, dry_run=False, order_executor=mock_executor_2)
    mock_alpaca_2.get_positions.return_value = [
        {
            "symbol": "SPY", "qty": opened.qty, "avg_entry_price": opened.entry_price,
            "current_price": opened.entry_price, "market_value": opened.qty * opened.entry_price,
            "unrealized_pl": 0.0, "side": "long",
        }
    ]
    mock_alpaca_2.get_account.return_value = _account(equity=100_000.0)

    bot2.startup()  # sync_with_broker() adopts the position, then the snapshot restores its metadata

    recovered = bot2.position_tracker.get_position("SPY")
    assert recovered is not None
    assert recovered.regime_at_entry == opened.regime_at_entry  # metadata survived the restart
    assert recovered.stop_level == opened.stop_level

    # The next bar should see the position already at (roughly) target and NOT re-enter.
    recovered.current_price = first.signal.entry_price
    second = bot2.process_bar("SPY", synthetic_ohlcv, now=now + timedelta(minutes=10))

    assert second.approved is True
    mock_executor_2.submit_bracket_order.assert_not_called()  # <-- the no-double-entry guarantee
