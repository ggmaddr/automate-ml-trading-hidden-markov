"""Tests for core.trading_bot. Uses a REAL trained HMMRegimeEngine/RiskManager/
PositionTracker (cheap, in-memory) but a MOCKED AlpacaClient/MarketDataFeed/
OrderExecutor/AlertManager — no real network access or live feed required.

`TradingBot.run()` itself (the actual "listen to the live WebSocket feed
forever" loop) is NOT covered here — see the module docstring in
core/trading_bot.py for why.
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timedelta
from unittest.mock import MagicMock

import pytest

from broker.alpaca_client import AccountSnapshot
from broker.position_tracker import PositionTracker
from core.risk_manager import CircuitBreakerStatus, RiskManager
from core.trading_bot import TradingBot


def _account(equity=100_000.0, cash=50_000.0, buying_power=150_000.0) -> AccountSnapshot:
    return AccountSnapshot(
        equity=equity, cash=cash, buying_power=buying_power, portfolio_value=equity,
        pattern_day_trader=False, trading_blocked=False, account_blocked=False, daytrade_count=0,
    )


def _build_bot(fresh_trained_hmm, full_config, tmp_path, dry_run=True, order_executor=None):
    mock_alpaca = MagicMock()
    mock_alpaca.paper = True
    mock_alpaca.get_account.return_value = _account()
    mock_alpaca.is_market_open.return_value = True
    mock_alpaca.get_positions.return_value = []

    mock_market_data = MagicMock()
    mock_alerts = MagicMock()
    position_tracker = PositionTracker(mock_alpaca)
    # Explicit lock_file_path: a PEAK_HALT in any test must never write
    # trading_halted.lock into the real project directory.
    risk_manager = RiskManager(full_config["risk"], lock_file_path=str(tmp_path / "halt.lock"))

    bot = TradingBot(
        full_config,
        alpaca_client=mock_alpaca,
        market_data=mock_market_data,
        hmm_engine=fresh_trained_hmm,
        risk_manager=risk_manager,
        position_tracker=position_tracker,
        order_executor=order_executor or MagicMock(),
        alert_manager=mock_alerts,
        dry_run=dry_run,
        state_path=str(tmp_path / "state_snapshot.json"),
        model_path=str(tmp_path / "hmm_model.pkl"),
    )
    return bot, mock_alpaca, mock_market_data, mock_alerts


# ---------------------------------------------------------------------------
# startup
# ---------------------------------------------------------------------------

def test_startup_connects_and_syncs_positions(fresh_trained_hmm, full_config, tmp_path):
    bot, mock_alpaca, _, _ = _build_bot(fresh_trained_hmm, full_config, tmp_path)
    bot.startup()
    mock_alpaca.get_account.assert_called()
    mock_alpaca.get_positions.assert_called()  # via position_tracker.sync_with_broker()


def test_startup_without_model_or_training_data_does_not_crash(full_config, tmp_path):
    mock_alpaca = MagicMock()
    mock_alpaca.paper = True
    mock_alpaca.get_account.return_value = _account()
    mock_alpaca.get_positions.return_value = []

    bot = TradingBot(
        full_config, alpaca_client=mock_alpaca, market_data=MagicMock(),
        position_tracker=PositionTracker(mock_alpaca), order_executor=MagicMock(), alert_manager=MagicMock(),
        state_path=str(tmp_path / "state.json"), model_path=str(tmp_path / "model.pkl"),
    )
    bot.startup()  # should log a warning, not raise
    assert bot.hmm_engine is None


# ---------------------------------------------------------------------------
# process_bar
# ---------------------------------------------------------------------------

def test_process_bar_approves_and_submits_order(fresh_trained_hmm, full_config, synthetic_ohlcv, tmp_path):
    mock_executor = MagicMock()
    mock_executor.submit_bracket_order.return_value = MagicMock(trade_id="t1", status="filled")
    bot, _, _, _ = _build_bot(fresh_trained_hmm, full_config, tmp_path, dry_run=False, order_executor=mock_executor)

    decision = bot.process_bar("SPY", synthetic_ohlcv)

    assert decision.regime_state is not None
    if decision.approved:
        assert decision.signal is not None
        mock_executor.submit_bracket_order.assert_called_once()
        assert bot.position_tracker.get_position("SPY") is not None
        assert bot.daily_trades_count == 1
    else:
        # The risk manager is free to reject (e.g. a wide stop capping size to $0) --
        # either outcome is legitimate, but it must never silently place an order
        # while reporting rejected.
        mock_executor.submit_bracket_order.assert_not_called()


def test_process_bar_does_not_double_enter_when_already_at_target(fresh_trained_hmm, full_config, synthetic_ohlcv, tmp_path):
    """The core double-entry guard: an approved signal every bar must NOT
    mean a brand-new order every bar once a position is already at target.
    """
    mock_executor = MagicMock()
    mock_executor.submit_bracket_order.return_value = MagicMock(trade_id="t1", status="filled")
    bot, _, _, _ = _build_bot(fresh_trained_hmm, full_config, tmp_path, dry_run=False, order_executor=mock_executor)

    now = datetime(2024, 3, 4, 10, 0)
    first = bot.process_bar("SPY", synthetic_ohlcv, now=now)
    if not first.approved:
        pytest.skip("risk manager rejected the first bar for this synthetic series/seed")
    assert mock_executor.submit_bracket_order.call_count == 1

    position = bot.position_tracker.get_position("SPY")
    position.current_price = first.signal.entry_price  # mark-to-market at the same price -> same allocation

    # Next bar, 5 minutes later -- well past the 60s duplicate-order window,
    # so only the target-allocation check (not the duplicate guard) is in play.
    second = bot.process_bar("SPY", synthetic_ohlcv, now=now + timedelta(minutes=5))

    assert second.approved is True
    assert mock_executor.submit_bracket_order.call_count == 1  # still just the one order -- no re-entry
    mock_executor.close_position.assert_not_called()


def test_process_bar_rebalances_via_close_then_reopen(fresh_trained_hmm, full_config, synthetic_ohlcv, tmp_path):
    mock_executor = MagicMock()
    mock_executor.submit_bracket_order.return_value = MagicMock(trade_id="t1", status="filled")
    bot, _, _, _ = _build_bot(fresh_trained_hmm, full_config, tmp_path, dry_run=False, order_executor=mock_executor)

    now = datetime(2024, 3, 4, 10, 0)
    first = bot.process_bar("SPY", synthetic_ohlcv, now=now)
    if not first.approved:
        pytest.skip("risk manager rejected the first bar for this synthetic series/seed")

    # Force a position far from any plausible target so the rebalance_threshold trips.
    bot.position_tracker.get_position("SPY").qty = 1

    second = bot.process_bar("SPY", synthetic_ohlcv, now=now + timedelta(minutes=5))

    assert second.approved is True
    mock_executor.close_position.assert_called_once_with("SPY")
    assert mock_executor.submit_bracket_order.call_count == 2


def test_process_bar_dry_run_never_submits_order(fresh_trained_hmm, full_config, synthetic_ohlcv, tmp_path):
    mock_executor = MagicMock()
    bot, _, _, _ = _build_bot(fresh_trained_hmm, full_config, tmp_path, dry_run=True, order_executor=mock_executor)

    bot.process_bar("SPY", synthetic_ohlcv)

    mock_executor.submit_bracket_order.assert_not_called()


def test_process_bar_rejects_when_circuit_breaker_halted(fresh_trained_hmm, full_config, synthetic_ohlcv, tmp_path):
    bot, mock_alpaca, _, _ = _build_bot(fresh_trained_hmm, full_config, tmp_path, dry_run=False)
    # The bot re-reads equity from get_account() internally when validating a
    # signal — it must match what's fed into the breaker below, or the
    # breaker "heals" back to NORMAL the moment process_bar re-checks it.
    mock_alpaca.get_account.return_value = _account(equity=96_000.0)

    now = datetime(2024, 3, 4, 10, 0)
    bot.risk_manager.circuit_breaker.update(100_000, now, [])
    bot.risk_manager.circuit_breaker.update(96_000, now + timedelta(hours=1), [])  # -4% -> DAILY_HALT

    decision = bot.process_bar("SPY", synthetic_ohlcv, now=now + timedelta(hours=2))

    assert decision.approved is False
    assert "circuit breaker" in decision.rejection_reason


def test_process_bar_holds_last_regime_on_hmm_error(fresh_trained_hmm, full_config, synthetic_ohlcv, tmp_path):
    bot, _, _, _ = _build_bot(fresh_trained_hmm, full_config, tmp_path)

    first = bot.process_bar("SPY", synthetic_ohlcv)
    assert first.regime_state is not None

    broken_engine = MagicMock()
    broken_engine.update.side_effect = RuntimeError("boom")
    broken_engine.is_flickering.return_value = False
    broken_engine.regime_info = bot.hmm_engine.regime_info
    bot.hmm_engine = broken_engine

    second = bot.process_bar("SPY", synthetic_ohlcv)
    assert second.regime_state == first.regime_state  # held, not crashed


def test_process_bar_no_signal_with_too_few_bars(fresh_trained_hmm, full_config, synthetic_ohlcv, tmp_path):
    bot, _, _, _ = _build_bot(fresh_trained_hmm, full_config, tmp_path)
    tiny_bars = synthetic_ohlcv.iloc[-10:]  # nowhere near enough for compute_features' warmup
    decision = bot.process_bar("SPY", tiny_bars)
    assert decision.approved is False


def test_process_bar_without_trained_model_returns_rejected(full_config, tmp_path):
    mock_alpaca = MagicMock()
    mock_alpaca.paper = True
    mock_alpaca.get_account.return_value = _account()
    bot = TradingBot(
        full_config, alpaca_client=mock_alpaca, market_data=MagicMock(),
        position_tracker=PositionTracker(mock_alpaca), order_executor=MagicMock(), alert_manager=MagicMock(),
        state_path=str(tmp_path / "s.json"), model_path=str(tmp_path / "m.pkl"),
    )
    decision = bot.process_bar("SPY", None)
    assert decision.approved is False
    assert "no HMM model" in decision.rejection_reason


# ---------------------------------------------------------------------------
# HMM retraining
# ---------------------------------------------------------------------------

def test_maybe_retrain_weekly_skips_when_recent(fresh_trained_hmm, full_config, synthetic_ohlcv, tmp_path):
    from data.feature_engineering import compute_features, log_returns

    bot, _, _, _ = _build_bot(fresh_trained_hmm, full_config, tmp_path)
    bot.last_train_date = datetime.now() - timedelta(days=1)
    retrained = bot.maybe_retrain_weekly(compute_features(synthetic_ohlcv), log_returns(synthetic_ohlcv["close"], 1))
    assert retrained is False


def test_maybe_retrain_weekly_triggers_after_7_days(fresh_trained_hmm, full_config, synthetic_ohlcv, tmp_path):
    from data.feature_engineering import compute_features, log_returns

    bot, _, _, mock_alerts = _build_bot(fresh_trained_hmm, full_config, tmp_path)
    bot.last_train_date = datetime.now() - timedelta(days=8)
    retrained = bot.maybe_retrain_weekly(compute_features(synthetic_ohlcv), log_returns(synthetic_ohlcv["close"], 1))
    assert retrained is True
    mock_alerts.send_alert.assert_called()
    assert os.path.exists(bot.model_path)


# ---------------------------------------------------------------------------
# State snapshot
# ---------------------------------------------------------------------------

def test_state_snapshot_round_trip(fresh_trained_hmm, full_config, tmp_path):
    bot, mock_alpaca, _, _ = _build_bot(fresh_trained_hmm, full_config, tmp_path)
    bot.last_train_date = datetime(2024, 1, 1, 9, 0)
    bot.daily_trades_count = 3
    bot.risk_manager.circuit_breaker.update(100_000, datetime(2024, 1, 2, 9, 30), [])
    bot.position_tracker.open_position(
        "SPY", qty=10, entry_price=500.0, stop_level=490.0, regime_at_entry="BULL", trade_id="abc",
        entry_time=datetime(2024, 1, 2, 9, 31),
    )

    bot.save_state_snapshot()
    assert os.path.exists(bot.state_path)

    bot2, _, _, _ = _build_bot(fresh_trained_hmm, full_config, tmp_path)
    bot2.position_tracker.open_position("SPY", qty=10, entry_price=500.0)  # as if sync_with_broker re-adopted it
    loaded = bot2._load_state_snapshot()

    assert loaded is True
    assert bot2.last_train_date == bot.last_train_date
    assert bot2.daily_trades_count == 3
    assert bot2.risk_manager.circuit_breaker.day_start_equity == 100_000
    assert bot2.position_tracker.get_position("SPY").stop_level == 490.0
    assert bot2.position_tracker.get_position("SPY").regime_at_entry == "BULL"


def test_load_state_snapshot_returns_false_when_missing(fresh_trained_hmm, full_config, tmp_path):
    bot, _, _, _ = _build_bot(fresh_trained_hmm, full_config, tmp_path)
    assert bot._load_state_snapshot() is False


# ---------------------------------------------------------------------------
# Trailing stops / forced liquidation
# ---------------------------------------------------------------------------

def test_update_trailing_stops_calls_modify_stop(fresh_trained_hmm, full_config, synthetic_ohlcv, tmp_path):
    mock_executor = MagicMock()
    mock_executor.modify_stop.return_value = True
    bot, _, _, _ = _build_bot(fresh_trained_hmm, full_config, tmp_path, dry_run=False, order_executor=mock_executor)
    bot.position_tracker.open_position("SPY", qty=10, entry_price=500.0, stop_level=480.0)

    regime_state = bot.hmm_engine.update(__import__("data.feature_engineering", fromlist=["compute_features"]).compute_features(synthetic_ohlcv))
    bot.update_trailing_stops("SPY", synthetic_ohlcv, regime_state)

    mock_executor.modify_stop.assert_called_once()


def test_update_trailing_stops_noop_in_dry_run(fresh_trained_hmm, full_config, synthetic_ohlcv, tmp_path):
    mock_executor = MagicMock()
    bot, _, _, _ = _build_bot(fresh_trained_hmm, full_config, tmp_path, dry_run=True, order_executor=mock_executor)
    bot.position_tracker.open_position("SPY", qty=10, entry_price=500.0, stop_level=480.0)

    bot.update_trailing_stops("SPY", synthetic_ohlcv, bot.last_regime_state.get("SPY"))

    mock_executor.modify_stop.assert_not_called()


def test_force_liquidate_if_halted_closes_positions(fresh_trained_hmm, full_config, tmp_path):
    mock_executor = MagicMock()
    bot, mock_alpaca, _, mock_alerts = _build_bot(
        fresh_trained_hmm, full_config, tmp_path, dry_run=False, order_executor=mock_executor
    )
    mock_alpaca.get_account.return_value = _account(equity=89_000.0)  # must match what the breaker sees below

    now = datetime(2024, 3, 4, 10, 0)
    bot.risk_manager.circuit_breaker.update(100_000, now, [])
    bot.risk_manager.circuit_breaker.update(89_000, now + timedelta(days=1), [])  # peak halt

    halted = bot.force_liquidate_if_halted(now=now + timedelta(days=1))

    assert halted is True
    mock_executor.close_all_positions.assert_called_once()
    mock_alerts.send_alert.assert_called()


def test_force_liquidate_if_halted_false_when_normal(fresh_trained_hmm, full_config, tmp_path):
    mock_executor = MagicMock()
    bot, _, _, _ = _build_bot(fresh_trained_hmm, full_config, tmp_path, dry_run=False, order_executor=mock_executor)
    assert bot.force_liquidate_if_halted() is False
    mock_executor.close_all_positions.assert_not_called()


# ---------------------------------------------------------------------------
# Shutdown / dashboard state
# ---------------------------------------------------------------------------

def test_shutdown_saves_state_and_stops_streams(fresh_trained_hmm, full_config, tmp_path):
    bot, _, mock_market_data, _ = _build_bot(fresh_trained_hmm, full_config, tmp_path)
    bot.shutdown()
    mock_market_data.stop_stream.assert_called_once()
    assert os.path.exists(bot.state_path)


def test_get_dashboard_state_shape(fresh_trained_hmm, full_config, synthetic_ohlcv, tmp_path):
    bot, _, _, _ = _build_bot(fresh_trained_hmm, full_config, tmp_path)
    bot.process_bar("SPY", synthetic_ohlcv)

    state = bot.get_dashboard_state()
    assert set(state) == {"regime", "portfolio", "positions", "recent_signals", "risk", "system"}
    assert state["system"]["paper"] is True


def test_dashboard_state_renders_through_real_dashboard(fresh_trained_hmm, full_config, synthetic_ohlcv, tmp_path):
    """The actual integration point start_dashboard() relies on: whatever
    get_dashboard_state() produces must be renderable by the real Dashboard,
    not just shaped right in isolation.
    """
    from monitoring.dashboard import Dashboard

    bot, _, _, _ = _build_bot(fresh_trained_hmm, full_config, tmp_path)
    bot.process_bar("SPY", synthetic_ohlcv)
    bot.position_tracker.open_position("SPY", qty=10, entry_price=500.0, stop_level=490.0)

    text = Dashboard(full_config["monitoring"]).render_to_text(bot.get_dashboard_state())
    assert "SPY" in text
    assert "PAPER" in text


def test_start_dashboard_runs_in_background_without_crashing(fresh_trained_hmm, full_config, synthetic_ohlcv, tmp_path):
    bot, _, _, _ = _build_bot(fresh_trained_hmm, full_config, tmp_path)
    bot.process_bar("SPY", synthetic_ohlcv)

    thread = bot.start_dashboard(refresh_seconds=0.05)
    try:
        time.sleep(0.3)  # let it render a few frames
        assert thread.is_alive()
    finally:
        pass  # daemon thread -- dies with the test process, nothing to join/stop


def test_write_dashboard_snapshot_round_trips_get_dashboard_state(fresh_trained_hmm, full_config, synthetic_ohlcv, tmp_path):
    import json

    bot, _, _, _ = _build_bot(fresh_trained_hmm, full_config, tmp_path)
    bot.process_bar("SPY", synthetic_ohlcv)

    out_path = str(tmp_path / "dashboard_state.json")
    bot.write_dashboard_snapshot(out_path)

    with open(out_path) as f:
        written = json.load(f)
    assert written == bot.get_dashboard_state()
    assert not os.path.exists(out_path + ".tmp")  # atomic rename leaves no temp file behind


def test_start_json_dashboard_feed_runs_in_background_without_crashing(fresh_trained_hmm, full_config, synthetic_ohlcv, tmp_path):
    bot, _, _, _ = _build_bot(fresh_trained_hmm, full_config, tmp_path)
    bot.process_bar("SPY", synthetic_ohlcv)

    out_path = str(tmp_path / "dashboard_state.json")
    thread = bot.start_json_dashboard_feed(out_path, refresh_seconds=0.05)
    try:
        time.sleep(0.3)  # let it write a few frames
        assert thread.is_alive()
        assert os.path.exists(out_path)
    finally:
        pass  # daemon thread -- dies with the test process, nothing to join/stop
