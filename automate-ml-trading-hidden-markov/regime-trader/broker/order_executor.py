"""Order placement, modification, and cancellation against Alpaca.

Entry orders default to LIMIT (+/- 0.1% of the signal's reference price) to
avoid paying the full bid-ask spread on a market order; if unfilled after
~30s they're canceled and, optionally, retried at market. Exit legs
(stop-loss / take-profit) are submitted together with the entry as an
Alpaca bracket order, so the stop is live the instant the entry fills.

Every submission carries a `trade_id` (defaults to a fresh UUID) that the
caller can use to correlate signal -> risk_decision -> order -> fill across
logs, by passing the same id that was attached to the RiskDecision upstream.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from alpaca.trading.enums import OrderClass, OrderSide, TimeInForce
from alpaca.trading.requests import (
    LimitOrderRequest,
    MarketOrderRequest,
    ReplaceOrderRequest,
    StopLossRequest,
    TakeProfitRequest,
)

from broker.alpaca_client import AlpacaClient
from core.regime_strategies import Direction, Signal

logger = logging.getLogger(__name__)

LIMIT_OFFSET_PCT = 0.001  # +/- 0.1% of reference price for the entry limit
FILL_TIMEOUT_SECONDS = 30
FILL_POLL_INTERVAL_SECONDS = 2

_TERMINAL_STATUSES = {"filled", "canceled", "expired", "rejected", "done_for_day"}


class OrderStatus(Enum):
    """Lifecycle status of a submitted order."""

    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELED = "canceled"
    REJECTED = "rejected"


@dataclass
class OrderResult:
    """Outcome of an order submission."""

    trade_id: str
    order_id: Optional[str]
    symbol: str
    status: OrderStatus
    filled_qty: float
    filled_avg_price: Optional[float]


class OrderExecutor:
    """Translates approved signals into Alpaca orders."""

    def __init__(self, client: AlpacaClient) -> None:
        self.client = client
        self._stops: dict[str, float] = {}  # symbol -> current live stop price
        self._stop_order_ids: dict[str, str] = {}  # symbol -> open stop-leg order id

    # ------------------------------------------------------------------
    # Submission
    # ------------------------------------------------------------------

    def submit_order(
        self, signal: Signal, qty: int, trade_id: Optional[str] = None, retry_at_market: bool = True,
    ) -> OrderResult:
        """Submit a plain entry order: LIMIT first, optionally MARKET if unfilled after 30s."""
        trade_id = trade_id or str(uuid.uuid4())
        side = OrderSide.BUY if signal.direction == Direction.LONG else OrderSide.SELL
        limit_price = self._limit_price(signal.entry_price, side)

        order_request = LimitOrderRequest(
            symbol=signal.symbol, qty=qty, side=side, time_in_force=TimeInForce.DAY,
            limit_price=limit_price, client_order_id=trade_id,
        )
        order = self.client.submit_order(order_request)
        result = self._await_fill(order, trade_id)

        if result.status not in (OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED) and retry_at_market:
            logger.info("Order %s unfilled after %ds, retrying at market", trade_id, FILL_TIMEOUT_SECONDS)
            self.cancel_order(order["id"])
            market_request = MarketOrderRequest(
                symbol=signal.symbol, qty=qty, side=side, time_in_force=TimeInForce.DAY,
                client_order_id=f"{trade_id}-market",
            )
            order = self.client.submit_order(market_request)
            result = self._await_fill(order, trade_id)

        return result

    def submit_bracket_order(self, signal: Signal, qty: int, trade_id: Optional[str] = None) -> OrderResult:
        """Entry + stop-loss + take-profit in one Alpaca bracket order.

        The stop and take-profit legs form an Alpaca One-Cancels-Other pair
        that only goes live once the entry fills.
        """
        if signal.stop_loss is None:
            raise ValueError("submit_bracket_order requires signal.stop_loss")

        trade_id = trade_id or str(uuid.uuid4())
        side = OrderSide.BUY if signal.direction == Direction.LONG else OrderSide.SELL
        limit_price = self._limit_price(signal.entry_price, side)

        order_request = LimitOrderRequest(
            symbol=signal.symbol, qty=qty, side=side, time_in_force=TimeInForce.DAY,
            limit_price=limit_price, client_order_id=trade_id, order_class=OrderClass.BRACKET,
            stop_loss=StopLossRequest(stop_price=round(signal.stop_loss, 2)),
            take_profit=TakeProfitRequest(limit_price=round(signal.take_profit, 2)) if signal.take_profit else None,
        )
        order = self.client.submit_order(order_request)

        self._stops[signal.symbol] = signal.stop_loss
        stop_leg = next((leg for leg in order.get("legs", []) if leg.get("stop_price") is not None), None)
        if stop_leg:
            self._stop_order_ids[signal.symbol] = stop_leg["id"]

        return self._await_fill(order, trade_id)

    # ------------------------------------------------------------------
    # Stop management
    # ------------------------------------------------------------------

    def modify_stop(self, symbol: str, new_stop: float, stop_order_id: Optional[str] = None) -> bool:
        """Tighten an existing stop. Refuses to widen it.

        Long-only here: "tighten" means RAISE the stop (closer to price,
        less room to lose), never lower it.
        """
        current = self._stops.get(symbol)
        if current is not None and new_stop < current:
            logger.warning("Refusing to widen stop for %s: %.2f -> %.2f", symbol, current, new_stop)
            return False

        order_id = stop_order_id or self._stop_order_ids.get(symbol)
        if order_id is None:
            logger.warning("No tracked stop order for %s -- cannot modify_stop", symbol)
            return False

        self.client.replace_order(order_id, ReplaceOrderRequest(stop_price=round(new_stop, 2)))
        self._stops[symbol] = new_stop
        return True

    # ------------------------------------------------------------------
    # Cancellation / closing
    # ------------------------------------------------------------------

    def cancel_order(self, order_id: str) -> bool:
        return self.client.cancel_order(order_id)

    def close_position(self, symbol: str) -> dict:
        result = self.client.close_position(symbol)
        self._stops.pop(symbol, None)
        self._stop_order_ids.pop(symbol, None)
        return result

    def close_all_positions(self) -> list[dict]:
        results = self.client.close_all_positions(cancel_orders=True)
        self._stops.clear()
        self._stop_order_ids.clear()
        return results

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _limit_price(reference_price: float, side: OrderSide) -> float:
        offset = (1 + LIMIT_OFFSET_PCT) if side == OrderSide.BUY else (1 - LIMIT_OFFSET_PCT)
        return round(reference_price * offset, 2)

    def _await_fill(self, order: dict, trade_id: str, timeout: int = FILL_TIMEOUT_SECONDS) -> OrderResult:
        """Poll until the order reaches a terminal status or `timeout` elapses."""
        deadline = time.monotonic() + timeout
        while order["status"] not in _TERMINAL_STATUSES and time.monotonic() < deadline:
            time.sleep(FILL_POLL_INTERVAL_SECONDS)
            order = self.client.get_order(order["id"])
        return self._to_order_result(order, trade_id)

    @staticmethod
    def _to_order_result(order: dict, trade_id: str) -> OrderResult:
        raw_status = order["status"]
        filled_qty = order.get("filled_qty") or 0.0
        if raw_status == "filled":
            status = OrderStatus.FILLED
        elif raw_status == "partially_filled" or (filled_qty and raw_status not in _TERMINAL_STATUSES):
            status = OrderStatus.PARTIALLY_FILLED
        elif raw_status in ("canceled", "expired", "done_for_day"):
            status = OrderStatus.CANCELED
        elif raw_status == "rejected":
            status = OrderStatus.REJECTED
        else:
            status = OrderStatus.PENDING

        return OrderResult(
            trade_id=trade_id,
            order_id=order.get("id"),
            symbol=order["symbol"],
            status=status,
            filled_qty=filled_qty,
            filled_avg_price=order.get("filled_avg_price"),
        )
