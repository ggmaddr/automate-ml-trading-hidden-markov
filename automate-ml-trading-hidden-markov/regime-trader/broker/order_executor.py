"""Order placement, modification, and cancellation against the broker."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from broker.alpaca_client import AlpacaClient
from core.signal_generator import TradingSignal


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

    order_id: str
    symbol: str
    status: OrderStatus
    filled_qty: float
    filled_avg_price: Optional[float]


class OrderExecutor:
    """Translates trading signals into broker order calls."""

    def __init__(self, client: AlpacaClient, config: dict) -> None:
        raise NotImplementedError

    def execute_signal(self, signal: TradingSignal) -> OrderResult:
        """Submit an order (or no-op) for a single trading signal."""
        raise NotImplementedError

    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order by id."""
        raise NotImplementedError

    def get_order_status(self, order_id: str) -> OrderStatus:
        """Poll current status of a submitted order."""
        raise NotImplementedError
