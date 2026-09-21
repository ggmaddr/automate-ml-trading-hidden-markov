"""Volatility-regime-based portfolio allocation strategies.

Translates a regime classification (from core.hmm_engine) plus trend context
into a target allocation, leverage, and position-sizing multiplier. Does not
predict price direction — it decides how much exposure to carry given the
current volatility regime.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from core.hmm_engine import RegimeState


@dataclass
class AllocationDecision:
    """Target portfolio allocation produced by a regime strategy."""

    target_allocation: float
    leverage: float
    size_multiplier: float
    reason: str


class RegimeStrategy:
    """Maps regime state + trend context to an allocation decision."""

    def __init__(self, config: dict) -> None:
        raise NotImplementedError

    def decide_allocation(
        self,
        regime_state: RegimeState,
        is_trending: bool,
        current_allocation: Optional[float] = None,
    ) -> AllocationDecision:
        """Compute the target allocation for the current regime/trend context."""
        raise NotImplementedError

    def needs_rebalance(self, current_allocation: float, target_allocation: float) -> bool:
        """True if drift between current and target allocation exceeds threshold."""
        raise NotImplementedError

    def classify_volatility_tier(self, regime_state: RegimeState) -> str:
        """Bucket a regime into 'low', 'mid', or 'high' volatility for sizing."""
        raise NotImplementedError
