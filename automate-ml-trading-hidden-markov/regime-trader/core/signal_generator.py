"""Combines HMM regime detection + strategy allocation + risk limits into
concrete trading signals ready for order execution.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import pandas as pd

from core.hmm_engine import HMMRegimeEngine, RegimeState
from core.regime_strategies import AllocationDecision, RegimeStrategy
from core.risk_manager import RiskManager


class SignalAction(Enum):
    """Directive produced for a given symbol at a given bar."""

    BUY = "buy"
    SELL = "sell"
    REBALANCE = "rebalance"
    HOLD = "hold"
    NO_ACTION = "no_action"


@dataclass
class TradingSignal:
    """A single actionable output of the signal pipeline for one symbol."""

    symbol: str
    action: SignalAction
    target_allocation: float
    regime_state: RegimeState
    allocation_decision: AllocationDecision
    confidence: float


class SignalGenerator:
    """Orchestrates HMM engine, strategy, and risk manager into signals."""

    def __init__(
        self,
        hmm_engine: HMMRegimeEngine,
        strategy: RegimeStrategy,
        risk_manager: RiskManager,
        config: dict,
    ) -> None:
        raise NotImplementedError

    def generate_signal(self, symbol: str, features: pd.DataFrame, current_allocation: float) -> TradingSignal:
        """Produce a trading signal for one symbol from its latest feature row(s)."""
        raise NotImplementedError

    def generate_signals(self, features_by_symbol: dict[str, pd.DataFrame], current_allocations: dict[str, float]) -> list[TradingSignal]:
        """Batch-generate signals across all tracked symbols."""
        raise NotImplementedError
