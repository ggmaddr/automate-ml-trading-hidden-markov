"""Tests for core.regime_strategies: per-tier strategies and the orchestrator."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.hmm_engine import RegimeInfo, RegimeState
from core.regime_strategies import (
    Direction,
    HighVolDefensiveStrategy,
    LowVolBullStrategy,
    MidVolCautiousStrategy,
    StrategyOrchestrator,
)

STRATEGY_CONFIG = {
    "low_vol_allocation": 0.95,
    "mid_vol_allocation_trend": 0.95,
    "mid_vol_allocation_no_trend": 0.60,
    "high_vol_allocation": 0.60,
    "low_vol_leverage": 1.25,
    "rebalance_threshold": 0.10,
    "uncertainty_size_mult": 0.50,
    "min_confidence": 0.55,
}


def _make_bars(n=80, trend="up", start=100.0, daily_move=0.5, noise=0.05, seed=1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    direction = 1 if trend == "up" else -1
    steps = direction * daily_move + rng.normal(0, noise, n)
    close = start + np.cumsum(steps)
    close = np.maximum(close, 1.0)
    high = close + rng.uniform(0.1, 0.5, n)
    low = close - rng.uniform(0.1, 0.5, n)
    open_ = close - direction * daily_move / 2
    volume = rng.uniform(1e6, 2e6, n)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume})


def _regime_state(state_id=0, label="BULL", probability=0.9) -> RegimeState:
    return RegimeState(
        label=label,
        state_id=state_id,
        probability=probability,
        state_probabilities=np.array([probability]),
        timestamp=pd.Timestamp("2024-01-01"),
        is_confirmed=True,
        consecutive_bars=5,
    )


# ---------------------------------------------------------------------------
# Individual strategies
# ---------------------------------------------------------------------------

def test_low_vol_bull_strategy_allocation_and_leverage():
    bars = _make_bars(trend="up")
    strategy = LowVolBullStrategy(STRATEGY_CONFIG)
    signal = strategy.generate_signal("SPY", bars, _regime_state())

    assert signal is not None
    assert signal.direction == Direction.LONG
    assert signal.position_size_pct == pytest.approx(0.95)
    assert signal.leverage == pytest.approx(1.25)
    assert signal.stop_loss < signal.entry_price


def test_low_vol_bull_returns_none_with_insufficient_history():
    bars = _make_bars(n=10, trend="up")
    strategy = LowVolBullStrategy(STRATEGY_CONFIG)
    assert strategy.generate_signal("SPY", bars, _regime_state()) is None


def test_mid_vol_strategy_reduces_allocation_when_trend_broken():
    up_bars = _make_bars(trend="up")
    down_bars = _make_bars(trend="down")
    strategy = MidVolCautiousStrategy(STRATEGY_CONFIG)

    trending_signal = strategy.generate_signal("SPY", up_bars, _regime_state())
    broken_signal = strategy.generate_signal("SPY", down_bars, _regime_state())

    assert trending_signal.position_size_pct == pytest.approx(0.95)
    assert broken_signal.position_size_pct == pytest.approx(0.60)
    assert trending_signal.leverage == 1.0
    assert broken_signal.leverage == 1.0


def test_high_vol_defensive_strategy_stays_partially_invested():
    bars = _make_bars(trend="down")
    strategy = HighVolDefensiveStrategy(STRATEGY_CONFIG)
    signal = strategy.generate_signal("SPY", bars, _regime_state())

    assert signal.direction == Direction.LONG  # never short
    assert signal.position_size_pct == pytest.approx(0.60)
    assert signal.leverage == 1.0


# ---------------------------------------------------------------------------
# Orchestrator: volatility-rank mapping is independent of the return label
# ---------------------------------------------------------------------------

def test_orchestrator_maps_by_volatility_not_by_label():
    # Deliberately mislabel state 0 as "BULL" (highest return) while giving
    # it the HIGHEST volatility, to prove the orchestrator ignores labels.
    regime_infos = {
        0: RegimeInfo(0, "BULL", expected_return=0.01, expected_volatility=0.05,
                      recommended_strategy_type="x", max_leverage_allowed=1.0,
                      max_position_size_pct=0.5, min_confidence_to_act=0.55),
        1: RegimeInfo(1, "NEUTRAL", expected_return=0.0, expected_volatility=0.02,
                      recommended_strategy_type="x", max_leverage_allowed=1.0,
                      max_position_size_pct=0.5, min_confidence_to_act=0.55),
        2: RegimeInfo(2, "BEAR", expected_return=-0.01, expected_volatility=0.005,
                      recommended_strategy_type="x", max_leverage_allowed=1.0,
                      max_position_size_pct=0.5, min_confidence_to_act=0.55),
    }
    orchestrator = StrategyOrchestrator(STRATEGY_CONFIG, regime_infos)

    # state 2 has the LOWEST volatility -> LowVolBullStrategy, even though its
    # label is "BEAR" (lowest return).
    assert isinstance(orchestrator.strategy_by_state[2], LowVolBullStrategy)
    # state 0 has the HIGHEST volatility -> HighVolDefensiveStrategy, even
    # though its label is "BULL" (highest return).
    assert isinstance(orchestrator.strategy_by_state[0], HighVolDefensiveStrategy)
    assert isinstance(orchestrator.strategy_by_state[1], MidVolCautiousStrategy)


def _default_regime_infos() -> dict[int, RegimeInfo]:
    return {
        0: RegimeInfo(0, "BEAR", -0.01, 0.03, "high_vol", 1.0, 0.5, 0.55),
        1: RegimeInfo(1, "NEUTRAL", 0.0, 0.015, "mid_vol", 1.0, 0.5, 0.55),
        2: RegimeInfo(2, "BULL", 0.01, 0.005, "low_vol", 1.25, 0.5, 0.55),
    }


def test_orchestrator_generates_signals_for_each_symbol():
    regime_infos = _default_regime_infos()
    orchestrator = StrategyOrchestrator(STRATEGY_CONFIG, regime_infos)
    bars_by_symbol = {"SPY": _make_bars(trend="up"), "QQQ": _make_bars(trend="up", seed=2)}

    signals = orchestrator.generate_signals(
        ["SPY", "QQQ"], bars_by_symbol, _regime_state(state_id=2, probability=0.9), is_flickering=False
    )

    assert {s.symbol for s in signals} == {"SPY", "QQQ"}
    assert all(s.position_size_pct == pytest.approx(0.95) for s in signals)


def test_orchestrator_skips_symbols_missing_bars():
    regime_infos = _default_regime_infos()
    orchestrator = StrategyOrchestrator(STRATEGY_CONFIG, regime_infos)
    bars_by_symbol = {"SPY": _make_bars(trend="up")}

    signals = orchestrator.generate_signals(
        ["SPY", "MISSING"], bars_by_symbol, _regime_state(state_id=2), is_flickering=False
    )

    assert {s.symbol for s in signals} == {"SPY"}


def test_uncertainty_mode_halves_size_on_low_confidence():
    regime_infos = _default_regime_infos()
    orchestrator = StrategyOrchestrator(STRATEGY_CONFIG, regime_infos)
    bars_by_symbol = {"SPY": _make_bars(trend="up")}

    low_conf_state = _regime_state(state_id=2, probability=0.30)  # below min_confidence_to_act
    signals = orchestrator.generate_signals(["SPY"], bars_by_symbol, low_conf_state, is_flickering=False)

    assert len(signals) == 1
    signal = signals[0]
    assert signal.position_size_pct == pytest.approx(0.95 * 0.50)
    assert signal.leverage == 1.0
    assert "UNCERTAINTY" in signal.reasoning


def test_uncertainty_mode_triggers_on_flickering_even_with_high_confidence():
    regime_infos = _default_regime_infos()
    orchestrator = StrategyOrchestrator(STRATEGY_CONFIG, regime_infos)
    bars_by_symbol = {"SPY": _make_bars(trend="up")}

    high_conf_state = _regime_state(state_id=2, probability=0.95)
    signals = orchestrator.generate_signals(["SPY"], bars_by_symbol, high_conf_state, is_flickering=True)

    assert signals[0].position_size_pct == pytest.approx(0.95 * 0.50)
    assert signals[0].leverage == 1.0


def test_orchestrator_unknown_regime_id_returns_no_signals():
    regime_infos = _default_regime_infos()
    orchestrator = StrategyOrchestrator(STRATEGY_CONFIG, regime_infos)
    bars_by_symbol = {"SPY": _make_bars(trend="up")}

    unseen_state = _regime_state(state_id=99)
    signals = orchestrator.generate_signals(["SPY"], bars_by_symbol, unseen_state, is_flickering=False)

    assert signals == []


def test_update_regime_infos_rebuilds_mapping():
    orchestrator = StrategyOrchestrator(STRATEGY_CONFIG, _default_regime_infos())
    assert isinstance(orchestrator.strategy_by_state[2], LowVolBullStrategy)

    # After a retrain, state 2 is now the MOST volatile.
    new_infos = {
        0: RegimeInfo(0, "BEAR", -0.01, 0.005, "x", 1.0, 0.5, 0.55),
        1: RegimeInfo(1, "NEUTRAL", 0.0, 0.015, "x", 1.0, 0.5, 0.55),
        2: RegimeInfo(2, "BULL", 0.01, 0.05, "x", 1.0, 0.5, 0.55),
    }
    orchestrator.update_regime_infos(new_infos)
    assert isinstance(orchestrator.strategy_by_state[2], HighVolDefensiveStrategy)
    assert isinstance(orchestrator.strategy_by_state[0], LowVolBullStrategy)


# ---------------------------------------------------------------------------
# Rebalancing
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "current,target,expected",
    [
        (0.90, 0.95, False),   # 5% drift, under 10% threshold
        (0.80, 0.95, True),    # 15% drift, over threshold
        (0.60, 0.60, False),   # no drift
    ],
)
def test_needs_rebalance_threshold(current, target, expected):
    orchestrator = StrategyOrchestrator(STRATEGY_CONFIG, _default_regime_infos())
    assert orchestrator.needs_rebalance(current, target) is expected
