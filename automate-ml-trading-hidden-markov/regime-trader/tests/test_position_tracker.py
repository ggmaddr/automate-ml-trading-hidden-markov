"""Tests for broker.position_tracker. Uses a mocked AlpacaClient — no real
network access required.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import MagicMock

import pytest

from broker.position_tracker import PositionTracker


def _broker_position(symbol="AAPL", qty=10.0, avg_entry_price=100.0, current_price=105.0) -> dict:
    return {
        "symbol": symbol, "qty": qty, "avg_entry_price": avg_entry_price, "current_price": current_price,
        "market_value": qty * current_price, "unrealized_pl": (current_price - avg_entry_price) * qty,
        "side": "long",
    }


# ---------------------------------------------------------------------------
# sync_with_broker
# ---------------------------------------------------------------------------

def test_sync_adopts_untracked_broker_position():
    mock_client = MagicMock()
    mock_client.get_positions.return_value = [_broker_position()]
    tracker = PositionTracker(mock_client)

    tracker.sync_with_broker()

    assert "AAPL" in tracker.positions
    assert tracker.positions["AAPL"].regime_at_entry is None  # adopted, regime unknown


def test_sync_drops_position_no_longer_at_broker():
    mock_client = MagicMock()
    mock_client.get_positions.return_value = []
    tracker = PositionTracker(mock_client)
    tracker.open_position("AAPL", qty=10, entry_price=100.0)

    tracker.sync_with_broker()

    assert "AAPL" not in tracker.positions


def test_sync_updates_existing_tracked_position_price():
    mock_client = MagicMock()
    mock_client.get_positions.return_value = [_broker_position(current_price=110.0, qty=12.0)]
    tracker = PositionTracker(mock_client)
    tracker.open_position("AAPL", qty=10, entry_price=100.0, regime_at_entry="BULL")

    tracker.sync_with_broker()

    assert tracker.positions["AAPL"].current_price == 110.0
    assert tracker.positions["AAPL"].qty == 12.0
    assert tracker.positions["AAPL"].regime_at_entry == "BULL"  # preserved, not overwritten


# ---------------------------------------------------------------------------
# on_fill
# ---------------------------------------------------------------------------

def test_on_fill_buy_opens_new_position():
    tracker = PositionTracker(MagicMock())
    tracker.on_fill({"event": "fill", "symbol": "AAPL", "qty": 10, "price": 100.0, "side": "buy"})

    position = tracker.get_position("AAPL")
    assert position is not None
    assert position.qty == 10
    assert position.entry_price == 100.0


def test_on_fill_buy_blends_average_entry_price():
    tracker = PositionTracker(MagicMock())
    tracker.open_position("AAPL", qty=10, entry_price=100.0)
    tracker.on_fill({"event": "fill", "symbol": "AAPL", "qty": 10, "price": 120.0, "side": "buy"})

    position = tracker.get_position("AAPL")
    assert position.qty == 20
    assert position.entry_price == pytest.approx(110.0)  # (100*10 + 120*10) / 20


def test_on_fill_sell_reduces_qty():
    tracker = PositionTracker(MagicMock())
    tracker.open_position("AAPL", qty=10, entry_price=100.0)
    tracker.on_fill({"event": "fill", "symbol": "AAPL", "qty": 4, "price": 105.0, "side": "sell"})

    assert tracker.get_position("AAPL").qty == 6


def test_on_fill_sell_closes_position_at_zero():
    tracker = PositionTracker(MagicMock())
    tracker.open_position("AAPL", qty=10, entry_price=100.0)
    tracker.on_fill({"event": "fill", "symbol": "AAPL", "qty": 10, "price": 105.0, "side": "sell"})

    assert tracker.get_position("AAPL") is None


def test_on_fill_ignores_non_fill_events():
    tracker = PositionTracker(MagicMock())
    tracker.on_fill({"event": "canceled", "symbol": "AAPL"})
    assert tracker.get_position("AAPL") is None


# ---------------------------------------------------------------------------
# Updates / queries
# ---------------------------------------------------------------------------

def test_update_price_and_pnl():
    tracker = PositionTracker(MagicMock())
    tracker.open_position("AAPL", qty=10, entry_price=100.0)
    tracker.update_price("AAPL", 110.0)

    position = tracker.get_position("AAPL")
    assert position.unrealized_pnl == pytest.approx(100.0)
    assert position.unrealized_pnl_pct == pytest.approx(0.10)


def test_holding_period():
    entry_time = datetime(2024, 1, 1, 9, 30)
    tracker = PositionTracker(MagicMock())
    tracker.open_position("AAPL", qty=10, entry_price=100.0, entry_time=entry_time)

    elapsed = tracker.get_position("AAPL").holding_period(as_of=entry_time + timedelta(days=3))
    assert elapsed == timedelta(days=3)


def test_total_exposure_and_unrealized_pnl():
    tracker = PositionTracker(MagicMock())
    tracker.open_position("AAPL", qty=10, entry_price=100.0)
    tracker.open_position("MSFT", qty=5, entry_price=200.0)
    tracker.update_price("AAPL", 110.0)
    tracker.update_price("MSFT", 190.0)

    assert tracker.total_exposure() == pytest.approx(10 * 110.0 + 5 * 190.0)
    assert tracker.total_unrealized_pnl() == pytest.approx(10 * 10.0 + 5 * (-10.0))


def test_to_portfolio_state_builds_position_info():
    tracker = PositionTracker(MagicMock())
    tracker.open_position("AAPL", qty=10, entry_price=100.0, sector="tech")

    state = tracker.to_portfolio_state(equity=100_000, cash=50_000, buying_power=150_000)

    assert state.equity == 100_000
    assert "AAPL" in state.positions
    assert state.positions["AAPL"].sector == "tech"
    assert state.positions["AAPL"].qty == 10
