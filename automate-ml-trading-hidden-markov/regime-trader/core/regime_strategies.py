"""Volatility-regime-based allocation strategies.

CORE IDEA: the HMM (core.hmm_engine) tells us how CALM or TURBULENT the
market is right now. It does not tell us which direction price will move.
This module turns that volatility read into "how much of the portfolio
should be invested, and with how much leverage" — nothing more.

We are always LONG or FLAT, never short. Shorting was tried in backtesting
and consistently lost money: markets drift up over time, and V-shaped
recoveries happen faster than the HMM can react, so short positions get run
over during the rebound that follows a crash. The response to high
volatility here is to hold LESS, not to bet on a drop.

Three strategies, chosen by how volatile a regime is relative to the
OTHER regimes the HMM found (not by its return-based label — a "BULL"
label just means "highest average return among the regimes we found," it
says nothing about how bumpy the ride is):

- LowVolBullStrategy    — calmest third of regimes: fully invested, small leverage.
- MidVolCautiousStrategy — middle third: invested if the trend (price vs 50 EMA)
  is intact, cut back if it's not.
- HighVolDefensiveStrategy — most turbulent third: reduced size, no leverage,
  but still partially invested so we don't miss the rebound.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import pandas as pd
from ta.volatility import AverageTrueRange

from core.hmm_engine import RegimeInfo, RegimeState

MIN_BARS_REQUIRED = 50  # need this many bars for a meaningful 50 EMA / ATR read


class Direction(Enum):
    """Position direction. FLAT means "no position" — LONG is the only side we take."""

    LONG = "long"
    FLAT = "flat"


@dataclass
class Signal:
    """One strategy's recommendation for one symbol at one point in time."""

    symbol: str
    direction: Direction
    confidence: float
    entry_price: float
    stop_loss: float
    take_profit: Optional[float]
    position_size_pct: float  # target fraction of portfolio equity, e.g. 0.60-0.95
    leverage: float  # 1.0 or 1.25
    regime_id: int
    regime_name: str
    regime_probability: float
    timestamp: Any
    reasoning: str
    strategy_name: str
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Shared price helpers (raw price-space, not the standardized HMM features)
# ---------------------------------------------------------------------------

def _ema(close: pd.Series, span: int) -> pd.Series:
    """Exponential moving average."""
    return close.ewm(span=span, adjust=False).mean()


def _atr(bars: pd.DataFrame, window: int = 14) -> pd.Series:
    """Average True Range in raw price units (not normalized by price)."""
    return AverageTrueRange(
        high=bars["high"], low=bars["low"], close=bars["close"], window=window, fillna=False
    ).average_true_range()


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

class BaseStrategy(ABC):
    """Common interface every volatility-tier strategy implements.

    `config` is the `strategy:` section of settings.yaml.
    """

    def __init__(self, config: dict) -> None:
        self.config = config

    @abstractmethod
    def generate_signal(
        self, symbol: str, bars: pd.DataFrame, regime_state: RegimeState
    ) -> Optional[Signal]:
        """Produce a Signal for `symbol` given its OHLCV bars and current regime.

        Returns None if there isn't enough price history yet to compute a
        stop level (the caller should simply skip the symbol this bar).
        """

    def _price_and_indicators(self, bars: pd.DataFrame) -> Optional[tuple[float, float, float]]:
        """Return (latest_close, atr_14, ema_50), or None if not enough history."""
        if len(bars) < MIN_BARS_REQUIRED:
            return None
        latest_close = float(bars["close"].iloc[-1])
        atr_val = float(_atr(bars).iloc[-1])
        ema50_val = float(_ema(bars["close"], 50).iloc[-1])
        if pd.isna(atr_val) or pd.isna(ema50_val):
            return None
        return latest_close, atr_val, ema50_val

    def _build_signal(
        self,
        symbol: str,
        regime_state: RegimeState,
        entry_price: float,
        stop_loss: float,
        position_size_pct: float,
        leverage: float,
        reasoning: str,
        metadata: Optional[dict] = None,
    ) -> Signal:
        return Signal(
            symbol=symbol,
            direction=Direction.LONG,
            confidence=regime_state.probability,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=None,
            position_size_pct=position_size_pct,
            leverage=leverage,
            regime_id=regime_state.state_id,
            regime_name=regime_state.label,
            regime_probability=regime_state.probability,
            timestamp=regime_state.timestamp,
            reasoning=reasoning,
            strategy_name=self.__class__.__name__,
            metadata=metadata or {},
        )


class LowVolBullStrategy(BaseStrategy):
    """Calmest regimes: this is where most of the strategy's returns come from.

    Calm markets trend up more often than not, so we go in heavy (95% of the
    portfolio) and add a little leverage (1.25x) to amplify a favorable
    environment, compounding over time.
    """

    def generate_signal(self, symbol: str, bars: pd.DataFrame, regime_state: RegimeState) -> Optional[Signal]:
        indicators = self._price_and_indicators(bars)
        if indicators is None:
            return None
        latest_close, atr_val, ema50_val = indicators

        stop_loss = max(latest_close - 3 * atr_val, ema50_val - 0.5 * atr_val)
        allocation = self.config["low_vol_allocation"]
        leverage = self.config["low_vol_leverage"]

        reasoning = (
            f"Low-volatility regime '{regime_state.label}': calm conditions historically "
            f"trend up, so we go {allocation:.0%} invested with {leverage:.2f}x leverage."
        )
        return self._build_signal(symbol, regime_state, latest_close, stop_loss, allocation, leverage, reasoning)


class MidVolCautiousStrategy(BaseStrategy):
    """Middle-of-the-road regimes: stay in only while the trend holds.

    Uses price vs. the 50 EMA as the trend filter: above it, the uptrend is
    still intact and we stay close to fully invested; below it, we cut back
    to reduce exposure until the trend re-establishes.
    """

    def generate_signal(self, symbol: str, bars: pd.DataFrame, regime_state: RegimeState) -> Optional[Signal]:
        indicators = self._price_and_indicators(bars)
        if indicators is None:
            return None
        latest_close, atr_val, ema50_val = indicators

        trend_intact = latest_close > ema50_val
        if trend_intact:
            allocation = self.config["mid_vol_allocation_trend"]
            trend_desc = "trend intact (price above 50 EMA)"
        else:
            allocation = self.config["mid_vol_allocation_no_trend"]
            trend_desc = "trend broken (price below 50 EMA)"

        leverage = 1.0
        stop_loss = ema50_val - 0.5 * atr_val

        reasoning = (
            f"Mid-volatility regime '{regime_state.label}': {trend_desc}, "
            f"so allocation is {allocation:.0%} with no leverage."
        )
        return self._build_signal(
            symbol, regime_state, latest_close, stop_loss, allocation, leverage, reasoning,
            metadata={"trend_intact": trend_intact},
        )


class HighVolDefensiveStrategy(BaseStrategy):
    """Most turbulent regimes: cut back hard, but don't go to zero.

    Going fully to cash would miss the sharp V-shaped rebounds that often
    follow a selloff, so we stay 60% invested with no leverage and a wider
    stop (volatile conditions need more room before a stop is "real").
    """

    def generate_signal(self, symbol: str, bars: pd.DataFrame, regime_state: RegimeState) -> Optional[Signal]:
        indicators = self._price_and_indicators(bars)
        if indicators is None:
            return None
        latest_close, atr_val, ema50_val = indicators

        allocation = self.config["high_vol_allocation"]
        leverage = 1.0
        stop_loss = ema50_val - 1.0 * atr_val

        reasoning = (
            f"High-volatility regime '{regime_state.label}': turbulent conditions call for "
            f"reduced ({allocation:.0%}) exposure, kept partial to catch a rebound."
        )
        return self._build_signal(symbol, regime_state, latest_close, stop_loss, allocation, leverage, reasoning)


# Backward-compatible aliases — earlier drafts of this bot named strategies
# after the return-based regime label instead of the volatility tier. Kept
# so old configs/scripts that reference these names keep working.
CrashDefensiveStrategy = HighVolDefensiveStrategy
BearTrendStrategy = HighVolDefensiveStrategy
StrongBearStrategy = HighVolDefensiveStrategy
WeakBearStrategy = MidVolCautiousStrategy
MeanReversionStrategy = MidVolCautiousStrategy
NeutralStrategy = MidVolCautiousStrategy
WeakBullStrategy = MidVolCautiousStrategy
BullTrendStrategy = LowVolBullStrategy
StrongBullStrategy = LowVolBullStrategy
EuphoriaCautiousStrategy = LowVolBullStrategy

# Naive label -> strategy fallback. The orchestrator below does NOT use this
# (it maps by volatility rank instead, per the module docstring) — this
# exists only for callers that have a regime label but no RegimeInfo yet,
# e.g. tooling/docs that want a rough strategy guess from a label alone.
LABEL_TO_STRATEGY: dict[str, type[BaseStrategy]] = {
    "CRASH": HighVolDefensiveStrategy,
    "STRONG_BEAR": HighVolDefensiveStrategy,
    "BEAR": HighVolDefensiveStrategy,
    "WEAK_BEAR": MidVolCautiousStrategy,
    "NEUTRAL": MidVolCautiousStrategy,
    "WEAK_BULL": MidVolCautiousStrategy,
    "BULL": LowVolBullStrategy,
    "STRONG_BULL": LowVolBullStrategy,
    "EUPHORIA": LowVolBullStrategy,
}


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class StrategyOrchestrator:
    """Maps each HMM regime to a strategy by volatility rank, and runs it.

    Also applies the cross-cutting "uncertainty mode" overlay: when the
    HMM's confidence in the current regime is low, or the regime has been
    flickering between states, every signal this bar gets its size halved
    and leverage forced to 1.0x, regardless of which strategy produced it.
    """

    def __init__(self, config: dict, regime_infos: dict[int, RegimeInfo]) -> None:
        self.config = config
        self.update_regime_infos(regime_infos)

    def update_regime_infos(self, regime_infos: dict[int, RegimeInfo]) -> None:
        """Rebuild the regime -> strategy mapping (call this after each HMM retrain)."""
        self.regime_infos = regime_infos
        n = len(regime_infos)

        # Sort by volatility ascending. Deliberately independent of the
        # HMM's return-based labels — see module docstring.
        states_by_vol = sorted(regime_infos, key=lambda s: regime_infos[s].expected_volatility)
        self.vol_rank: dict[int, int] = {state: rank for rank, state in enumerate(states_by_vol)}

        self.strategy_by_state: dict[int, BaseStrategy] = {}
        for state, rank in self.vol_rank.items():
            position = rank / (n - 1) if n > 1 else 0.0
            if position <= 0.33:
                strategy_cls: type[BaseStrategy] = LowVolBullStrategy
            elif position >= 0.67:
                strategy_cls = HighVolDefensiveStrategy
            else:
                strategy_cls = MidVolCautiousStrategy
            self.strategy_by_state[state] = strategy_cls(self.config)

    def _is_uncertain(self, regime_state: RegimeState, is_flickering: bool) -> bool:
        info = self.regime_infos.get(regime_state.state_id)
        threshold = info.min_confidence_to_act if info is not None else self.config.get("min_confidence", 0.55)
        return is_flickering or regime_state.probability < threshold

    def generate_signals(
        self,
        symbols: list[str],
        bars_by_symbol: dict[str, pd.DataFrame],
        regime_state: RegimeState,
        is_flickering: bool,
    ) -> list[Signal]:
        """Generate one Signal per symbol that has enough history, under the current regime."""
        strategy = self.strategy_by_state.get(regime_state.state_id)
        if strategy is None:
            return []

        uncertain = self._is_uncertain(regime_state, is_flickering)
        uncertainty_mult = self.config.get("uncertainty_size_mult", 0.5)

        signals: list[Signal] = []
        for symbol in symbols:
            bars = bars_by_symbol.get(symbol)
            if bars is None or bars.empty:
                continue

            signal = strategy.generate_signal(symbol, bars, regime_state)
            if signal is None:
                continue

            if uncertain:
                signal.position_size_pct *= uncertainty_mult
                signal.leverage = 1.0
                signal.reasoning += " [UNCERTAINTY → size halved]"

            signals.append(signal)

        return signals

    def needs_rebalance(self, current_allocation: float, target_allocation: float) -> bool:
        """True if drift between current and target allocation exceeds rebalance_threshold.

        Keeps us from re-trading on every tiny probability wobble — fewer
        trades means less slippage and better real-world performance.
        """
        threshold = self.config.get("rebalance_threshold", 0.10)
        return abs(target_allocation - current_allocation) > threshold
