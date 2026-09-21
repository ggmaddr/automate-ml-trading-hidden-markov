"""Tracks open positions and realized/unrealized P&L."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass
class Position:
    """A single open position."""

    symbol: str
    qty: float
    avg_entry_price: float
    current_price: float
    opened_at: datetime

    @property
    def market_value(self) -> float:
        raise NotImplementedError

    @property
    def unrealized_pnl(self) -> float:
        raise NotImplementedError


class PositionTracker:
    """Maintains the book of open positions and P&L history."""

    def __init__(self) -> None:
        raise NotImplementedError

    def update_position(self, symbol: str, qty_delta: float, fill_price: float) -> None:
        """Apply a fill to the tracked position for a symbol."""
        raise NotImplementedError

    def get_position(self, symbol: str) -> Position | None:
        """Return the current position for a symbol, if any."""
        raise NotImplementedError

    def get_all_positions(self) -> list[Position]:
        """Return all currently open positions."""
        raise NotImplementedError

    def total_exposure(self) -> float:
        """Sum of absolute market value across all open positions."""
        raise NotImplementedError

    def realized_pnl_today(self) -> float:
        """Sum of realized P&L from trades closed today."""
        raise NotImplementedError
