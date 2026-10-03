"""Tests for broker.order_executor. Uses a mocked AlpacaClient (dict-shaped
responses matching broker.alpaca_client's contract) — no real network access.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from broker.order_executor import OrderExecutor, OrderStatus
from core.regime_strategies import Direction, Signal


def _signal(symbol="AAPL", entry=100.0, stop=98.0, take_profit=None) -> Signal:
    return Signal(
        symbol=symbol, direction=Direction.LONG, confidence=0.9, entry_price=entry, stop_loss=stop,
        take_profit=take_profit, position_size_pct=0.15, leverage=1.0, regime_id=0, regime_name="BULL",
        regime_probability=0.9, timestamp=None, reasoning="test", strategy_name="TestStrategy", metadata={},
    )


def _order_dict(order_id="o1", status="filled", symbol="AAPL", filled_qty=10.0,
                 filled_avg_price=100.0, legs=None, **overrides) -> dict:
    base = {
        "id": order_id, "client_order_id": None, "symbol": symbol, "qty": filled_qty, "side": "buy",
        "order_type": "limit", "status": status, "filled_qty": filled_qty, "filled_avg_price": filled_avg_price,
        "limit_price": None, "stop_price": None, "submitted_at": None, "filled_at": None, "canceled_at": None,
        "legs": legs or [],
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# submit_order
# ---------------------------------------------------------------------------

def test_submit_order_fills_immediately():
    mock_client = MagicMock()
    mock_client.submit_order.return_value = _order_dict(status="filled")
    executor = OrderExecutor(mock_client)

    result = executor.submit_order(_signal(), qty=10)

    assert result.status == OrderStatus.FILLED
    assert result.filled_avg_price == 100.0
    assert mock_client.submit_order.call_count == 1  # no market retry needed


def test_submit_order_limit_price_offset_for_buy():
    mock_client = MagicMock()
    mock_client.submit_order.return_value = _order_dict(status="filled")
    executor = OrderExecutor(mock_client)

    executor.submit_order(_signal(entry=100.0), qty=10)

    request = mock_client.submit_order.call_args[0][0]
    assert request.limit_price == pytest.approx(100.10, abs=1e-6)  # +0.1% for a BUY


def test_submit_order_retries_at_market_after_timeout(monkeypatch):
    monkeypatch.setattr("broker.order_executor.time.sleep", lambda _: None)
    fake_time = iter([0, 5, 35, 35, 35, 35])
    monkeypatch.setattr("broker.order_executor.time.monotonic", lambda: next(fake_time))

    mock_client = MagicMock()
    mock_client.submit_order.side_effect = [
        _order_dict(order_id="limit1", status="new", filled_qty=0.0),
        _order_dict(order_id="market1", status="filled", filled_avg_price=101.0),
    ]
    mock_client.get_order.return_value = _order_dict(order_id="limit1", status="new", filled_qty=0.0)
    mock_client.cancel_order.return_value = True

    executor = OrderExecutor(mock_client)
    result = executor.submit_order(_signal(), qty=10)

    assert result.status == OrderStatus.FILLED
    assert result.filled_avg_price == 101.0
    mock_client.cancel_order.assert_called_once_with("limit1")
    assert mock_client.submit_order.call_count == 2


def test_submit_order_no_retry_when_disabled(monkeypatch):
    monkeypatch.setattr("broker.order_executor.time.sleep", lambda _: None)
    fake_time = iter([0, 35, 35])
    monkeypatch.setattr("broker.order_executor.time.monotonic", lambda: next(fake_time))

    mock_client = MagicMock()
    mock_client.submit_order.return_value = _order_dict(order_id="limit1", status="new", filled_qty=0.0)
    mock_client.get_order.return_value = _order_dict(order_id="limit1", status="new", filled_qty=0.0)

    executor = OrderExecutor(mock_client)
    result = executor.submit_order(_signal(), qty=10, retry_at_market=False)

    assert result.status == OrderStatus.PENDING
    assert mock_client.submit_order.call_count == 1
    mock_client.cancel_order.assert_not_called()


# ---------------------------------------------------------------------------
# submit_bracket_order
# ---------------------------------------------------------------------------

def test_submit_bracket_order_requires_stop_loss():
    executor = OrderExecutor(MagicMock())
    with pytest.raises(ValueError):
        executor.submit_bracket_order(_signal(stop=None), qty=10)


def test_submit_bracket_order_tracks_stop_leg_id():
    mock_client = MagicMock()
    stop_leg = _order_dict(order_id="stop1", status="held", stop_price=98.0, filled_avg_price=None)
    tp_leg = _order_dict(order_id="tp1", status="held", filled_avg_price=None)
    mock_client.submit_order.return_value = _order_dict(
        order_id="entry1", status="filled", legs=[stop_leg, tp_leg],
    )

    executor = OrderExecutor(mock_client)
    result = executor.submit_bracket_order(_signal(stop=98.0, take_profit=110.0), qty=10)

    assert result.status == OrderStatus.FILLED
    assert executor._stop_order_ids["AAPL"] == "stop1"
    assert executor._stops["AAPL"] == 98.0

    request = mock_client.submit_order.call_args[0][0]
    assert request.stop_loss.stop_price == 98.0
    assert request.take_profit.limit_price == 110.0


# ---------------------------------------------------------------------------
# modify_stop
# ---------------------------------------------------------------------------

def test_modify_stop_tightens_successfully():
    mock_client = MagicMock()
    executor = OrderExecutor(mock_client)
    executor._stops["AAPL"] = 98.0
    executor._stop_order_ids["AAPL"] = "stop1"

    assert executor.modify_stop("AAPL", 99.0) is True
    mock_client.replace_order.assert_called_once()
    assert executor._stops["AAPL"] == 99.0


def test_modify_stop_refuses_to_widen():
    mock_client = MagicMock()
    executor = OrderExecutor(mock_client)
    executor._stops["AAPL"] = 98.0
    executor._stop_order_ids["AAPL"] = "stop1"

    assert executor.modify_stop("AAPL", 97.0) is False
    mock_client.replace_order.assert_not_called()
    assert executor._stops["AAPL"] == 98.0  # unchanged


def test_modify_stop_without_tracked_order_fails():
    executor = OrderExecutor(MagicMock())
    assert executor.modify_stop("AAPL", 99.0) is False


# ---------------------------------------------------------------------------
# Cancel / close
# ---------------------------------------------------------------------------

def test_close_position_clears_tracked_stop():
    mock_client = MagicMock()
    mock_client.close_position.return_value = {"id": "x", "status": "pending_cancel"}
    executor = OrderExecutor(mock_client)
    executor._stops["AAPL"] = 98.0
    executor._stop_order_ids["AAPL"] = "stop1"

    executor.close_position("AAPL")

    assert "AAPL" not in executor._stops
    assert "AAPL" not in executor._stop_order_ids


def test_close_all_positions_clears_all_tracked_stops():
    mock_client = MagicMock()
    mock_client.close_all_positions.return_value = [{"symbol": "AAPL", "order_id": "x", "status": "200"}]
    executor = OrderExecutor(mock_client)
    executor._stops = {"AAPL": 98.0, "MSFT": 300.0}
    executor._stop_order_ids = {"AAPL": "s1", "MSFT": "s2"}

    executor.close_all_positions()

    assert executor._stops == {}
    assert executor._stop_order_ids == {}
