"""Position sizing, leverage limits, and drawdown-based trading halts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import Enum
from typing import Optional


class RiskState(Enum):
    """Current risk posture, driven by realized drawdown."""

    NORMAL = "normal"
    REDUCED = "reduced"
    HALTED = "halted"


@dataclass
class PositionSizeResult:
    """Result of a position-sizing calculation for a single symbol."""

    symbol: str
    shares: int
    notional: float
    risk_amount: float
    capped_by: Optional[str]  # which limit constrained sizing, if any


class RiskManager:
    """Enforces per-trade risk, exposure, leverage, and drawdown limits."""

    def __init__(self, config: dict) -> None:
        raise NotImplementedError

    def size_position(
        self,
        symbol: str,
        entry_price: float,
        stop_price: float,
        equity: float,
        size_multiplier: float = 1.0,
    ) -> PositionSizeResult:
        """Size a position under max_risk_per_trade, capped by exposure/leverage limits."""
        raise NotImplementedError

    def check_exposure_limits(self, proposed_notional: float, current_exposure: float, equity: float) -> bool:
        """True if adding proposed_notional keeps gross exposure within max_exposure/leverage."""
        raise NotImplementedError

    def update_drawdown(self, equity: float, as_of: date) -> RiskState:
        """Update daily/weekly/peak drawdown tracking and return the resulting risk state."""
        raise NotImplementedError

    def get_risk_state(self) -> RiskState:
        """Return the current risk state without updating it."""
        raise NotImplementedError

    def can_open_new_position(self, open_position_count: int, trades_today: int) -> bool:
        """True if max_concurrent and max_daily_trades limits allow a new position."""
        raise NotImplementedError
