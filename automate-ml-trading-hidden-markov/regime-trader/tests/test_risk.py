"""Tests for core.risk_manager: circuit breakers and signal validation."""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from core.regime_strategies import Direction, Signal
from core.risk_manager import (
    CircuitBreaker,
    CircuitBreakerStatus,
    PortfolioState,
    PositionInfo,
    RiskManager,
)

RISK_CONFIG = {
    "max_risk_per_trade": 0.01,
    "max_exposure": 0.80,
    "max_leverage": 1.25,
    "max_single_position": 0.15,
    "max_concurrent": 5,
    "max_daily_trades": 20,
    "daily_dd_reduce": 0.02,
    "daily_dd_halt": 0.03,
    "weekly_dd_reduce": 0.05,
    "weekly_dd_halt": 0.07,
    "max_dd_from_peak": 0.10,
    "max_correlated_exposure": 0.30,
    "min_position_dollars": 100,
    "gap_risk_atr_mult": 3,
    "overnight_max_pct": 0.02,
    "correlation_reduce_threshold": 0.70,
    "correlation_reject_threshold": 0.85,
    "max_bid_ask_spread_pct": 0.005,
    "duplicate_order_window_seconds": 60,
}

NOW = datetime(2024, 3, 4, 16, 0)


def _signal(symbol="AAPL", entry=100.0, stop=98.0, position_size_pct=0.95, leverage=1.25,
            confidence=0.9, metadata=None, timestamp=NOW) -> Signal:
    return Signal(
        symbol=symbol, direction=Direction.LONG, confidence=confidence, entry_price=entry,
        stop_loss=stop, take_profit=None, position_size_pct=position_size_pct, leverage=leverage,
        regime_id=0, regime_name="BULL", regime_probability=confidence, timestamp=timestamp,
        reasoning="test", strategy_name="TestStrategy", metadata=metadata or {},
    )


def _portfolio(equity=100_000.0, cash=100_000.0, buying_power=100_000.0, **kwargs) -> PortfolioState:
    return PortfolioState(equity=equity, cash=cash, buying_power=buying_power, **kwargs)


# ---------------------------------------------------------------------------
# CircuitBreaker
# ---------------------------------------------------------------------------

def test_circuit_breaker_normal_with_no_drawdown(tmp_path):
    cb = CircuitBreaker(RISK_CONFIG, lock_file_path=str(tmp_path / "halt.lock"))
    status = cb.update(100_000, NOW, [])
    assert status == CircuitBreakerStatus.NORMAL


def test_circuit_breaker_daily_reduce_then_halt(tmp_path):
    cb = CircuitBreaker(RISK_CONFIG, lock_file_path=str(tmp_path / "halt.lock"))
    cb.update(100_000, NOW, [])  # establishes day_start_equity
    assert cb.update(97_500, NOW + timedelta(hours=1), []) == CircuitBreakerStatus.DAILY_REDUCE  # -2.5%
    assert cb.update(96_000, NOW + timedelta(hours=2), []) == CircuitBreakerStatus.DAILY_HALT  # -4%


def test_circuit_breaker_weekly_thresholds(tmp_path):
    cb = CircuitBreaker(RISK_CONFIG, lock_file_path=str(tmp_path / "halt.lock"))
    cb.update(100_000, NOW, [])
    assert cb.update(94_000, NOW + timedelta(days=1), []) == CircuitBreakerStatus.WEEKLY_REDUCE  # -6%
    assert cb.update(92_000, NOW + timedelta(days=2), []) == CircuitBreakerStatus.WEEKLY_HALT  # -8%


def test_circuit_breaker_peak_halt_writes_lock_file(tmp_path):
    lock_path = tmp_path / "halt.lock"
    cb = CircuitBreaker(RISK_CONFIG, lock_file_path=str(lock_path))
    cb.update(100_000, NOW, [])
    cb.update(120_000, NOW + timedelta(days=1), [])  # new peak
    assert not lock_path.exists()
    status = cb.update(105_000, NOW + timedelta(days=2), ["AAPL", "SPY"])  # -12.5% from peak
    assert status == CircuitBreakerStatus.PEAK_HALT
    assert lock_path.exists()


def test_circuit_breaker_check_respects_lock_file_after_recovery(tmp_path):
    lock_path = tmp_path / "halt.lock"
    cb = CircuitBreaker(RISK_CONFIG, lock_file_path=str(lock_path))
    cb.update(100_000, NOW, [])
    cb.update(89_000, NOW + timedelta(days=1), [])  # -11%, triggers peak halt
    assert lock_path.exists()
    # Equity "recovers" but the lock file (manual-reset-only) still blocks trading.
    assert cb.check() == CircuitBreakerStatus.PEAK_HALT
    lock_path.unlink()
    cb._status = CircuitBreakerStatus.NORMAL
    assert cb.check() == CircuitBreakerStatus.NORMAL


def test_circuit_breaker_history_logs_once_per_escalation(tmp_path):
    cb = CircuitBreaker(RISK_CONFIG, lock_file_path=str(tmp_path / "halt.lock"))
    cb.update(100_000, NOW, [])
    cb.update(97_500, NOW + timedelta(hours=1), [])
    cb.update(97_400, NOW + timedelta(hours=2), [])  # still DAILY_REDUCE, no new event
    cb.update(97_300, NOW + timedelta(hours=3), [])
    assert len(cb.get_history()) == 1


# ---------------------------------------------------------------------------
# RiskManager.validate_signal — rejections
# ---------------------------------------------------------------------------

def test_rejects_signal_without_stop_loss():
    rm = RiskManager(RISK_CONFIG)
    signal = _signal(stop=100.0, entry=100.0)  # stop == entry -> invalid
    decision = rm.validate_signal(signal, _portfolio(), now=NOW)
    assert not decision.approved
    assert "stop_loss" in decision.rejection_reason


def test_rejects_duplicate_order_within_window():
    rm = RiskManager(RISK_CONFIG)
    portfolio = _portfolio(recent_orders=[("AAPL", Direction.LONG, NOW - timedelta(seconds=30))])
    decision = rm.validate_signal(_signal(), portfolio, now=NOW)
    assert not decision.approved
    assert "duplicate" in decision.rejection_reason


def test_allows_same_symbol_after_duplicate_window_expires():
    rm = RiskManager(RISK_CONFIG)
    portfolio = _portfolio(recent_orders=[("AAPL", Direction.LONG, NOW - timedelta(seconds=120))])
    decision = rm.validate_signal(_signal(), portfolio, now=NOW)
    assert decision.approved


def test_rejects_when_not_tradeable():
    rm = RiskManager(RISK_CONFIG)
    portfolio = _portfolio(tradeable={"AAPL": False})
    decision = rm.validate_signal(_signal(), portfolio, now=NOW)
    assert not decision.approved
    assert "not tradeable" in decision.rejection_reason


def test_rejects_when_spread_too_wide():
    rm = RiskManager(RISK_CONFIG)
    portfolio = _portfolio(bid_ask_spread_pct={"AAPL": 0.01})
    decision = rm.validate_signal(_signal(), portfolio, now=NOW)
    assert not decision.approved
    assert "spread" in decision.rejection_reason


def test_rejects_at_max_daily_trades():
    rm = RiskManager(RISK_CONFIG)
    portfolio = _portfolio(daily_trades_count=RISK_CONFIG["max_daily_trades"])
    decision = rm.validate_signal(_signal(), portfolio, now=NOW)
    assert not decision.approved
    assert "max_daily_trades" in decision.rejection_reason


def test_rejects_at_max_concurrent_positions():
    rm = RiskManager(RISK_CONFIG)
    positions = {f"SYM{i}": PositionInfo(f"SYM{i}", 10, 50.0, 50.0) for i in range(5)}
    portfolio = _portfolio(positions=positions)
    decision = rm.validate_signal(_signal(), portfolio, now=NOW)
    assert not decision.approved
    assert "max_concurrent" in decision.rejection_reason


def test_halted_breaker_rejects_everything():
    rm = RiskManager(RISK_CONFIG)
    rm.circuit_breaker.update(100_000, NOW, [])
    rm.circuit_breaker.update(96_000, NOW + timedelta(hours=1), [])  # -4% daily halt
    decision = rm.validate_signal(_signal(), _portfolio(equity=96_000), now=NOW + timedelta(hours=2))
    assert not decision.approved
    assert "circuit breaker" in decision.rejection_reason


# ---------------------------------------------------------------------------
# RiskManager.validate_signal — sizing caps
# ---------------------------------------------------------------------------

def test_max_single_position_caps_a_full_strength_request():
    rm = RiskManager(RISK_CONFIG)
    # Deliberately unrealistic ultra-tight stop so the 1%-risk/gap caps are
    # miles above 1.0 and don't bind — isolates max_single_position (15%).
    signal = _signal(entry=100.0, stop=99.9, position_size_pct=0.95, leverage=1.25)
    decision = rm.validate_signal(signal, _portfolio(), now=NOW)
    assert decision.approved
    assert decision.modified_signal.position_size_pct * decision.modified_signal.leverage == pytest.approx(0.15, abs=1e-6)
    assert any("capped" in m for m in decision.modifications)


def test_regime_max_tighter_than_portfolio_max_binds():
    rm = RiskManager(RISK_CONFIG)
    signal = _signal(entry=100.0, stop=99.9, position_size_pct=0.95, leverage=1.0)
    decision = rm.validate_signal(signal, _portfolio(), regime_max_position_size_pct=0.05, now=NOW)
    assert decision.approved
    assert decision.modified_signal.position_size_pct == pytest.approx(0.05, abs=1e-6)


def test_risk_based_cap_binds_when_gap_cap_relaxed():
    # Relax overnight_max_pct so gap risk never binds, isolating the 1%-risk formula.
    config = {**RISK_CONFIG, "overnight_max_pct": 10.0}
    rm = RiskManager(config)
    # stop_distance_pct = 10% -> risk_based_pct = 0.01/0.10 = 0.10, under max_single_position (0.15).
    signal = _signal(entry=100.0, stop=90.0, position_size_pct=0.95, leverage=1.0)
    decision = rm.validate_signal(signal, _portfolio(), now=NOW)
    assert decision.approved
    assert decision.modified_signal.position_size_pct == pytest.approx(0.10, abs=1e-6)


def test_never_sizes_above_what_strategy_requested():
    rm = RiskManager(RISK_CONFIG)
    # A conservative, already-small request: no cap should ever INCREASE it.
    signal = _signal(entry=100.0, stop=90.0, position_size_pct=0.05, leverage=1.0)
    decision = rm.validate_signal(signal, _portfolio(), now=NOW)
    assert decision.approved
    assert decision.modified_signal.position_size_pct <= 0.05 + 1e-9
    assert decision.modifications == []


def test_min_position_dollars_rejects_tiny_size():
    rm = RiskManager(RISK_CONFIG)
    signal = _signal(entry=100.0, stop=99.9, position_size_pct=0.0005, leverage=1.0)
    decision = rm.validate_signal(signal, _portfolio(equity=100_000), now=NOW)
    assert not decision.approved
    assert "minimum" in decision.rejection_reason


def test_insufficient_buying_power_rejects():
    rm = RiskManager(RISK_CONFIG)
    signal = _signal(entry=100.0, stop=99.9, position_size_pct=0.95, leverage=1.25)
    portfolio = _portfolio(equity=100_000, buying_power=1_000)  # tiny buying power
    decision = rm.validate_signal(signal, portfolio, now=NOW)
    assert not decision.approved
    assert "buying power" in decision.rejection_reason


# ---------------------------------------------------------------------------
# Leverage rules
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kwargs", [
    {"regime_uncertain": True},
    {"is_flickering": True},
])
def test_leverage_forced_to_one_on_uncertainty_or_flicker(kwargs):
    rm = RiskManager(RISK_CONFIG)
    signal = _signal(entry=100.0, stop=99.9, leverage=1.25)
    decision = rm.validate_signal(signal, _portfolio(), now=NOW, **kwargs)
    assert decision.approved
    assert decision.modified_signal.leverage == 1.0
    assert any("leverage forced" in m for m in decision.modifications)


def test_leverage_forced_to_one_with_three_or_more_positions():
    rm = RiskManager(RISK_CONFIG)
    positions = {f"SYM{i}": PositionInfo(f"SYM{i}", 10, 50.0, 50.0) for i in range(3)}
    signal = _signal(entry=100.0, stop=99.9, leverage=1.25)
    decision = rm.validate_signal(signal, _portfolio(positions=positions), now=NOW)
    assert decision.approved
    assert decision.modified_signal.leverage == 1.0


def test_leverage_unforced_in_calm_normal_conditions():
    rm = RiskManager(RISK_CONFIG)
    signal = _signal(entry=100.0, stop=99.9, leverage=1.25)
    decision = rm.validate_signal(signal, _portfolio(), now=NOW)
    assert decision.approved
    assert decision.modified_signal.leverage == 1.25


# ---------------------------------------------------------------------------
# Correlation check
# ---------------------------------------------------------------------------

def _correlated_price_series(target_corr: float, n: int = 80, seed: int = 0):
    rng = np.random.default_rng(seed)
    base = rng.normal(size=n)
    other = rng.normal(size=n)
    other = other - (np.dot(other, base) / np.dot(base, base)) * base  # orthogonalize
    combined = target_corr * (base / base.std()) + np.sqrt(1 - target_corr**2) * (other / other.std())

    dates = pd.bdate_range("2024-01-01", periods=n)
    prices_a = pd.Series(100 * np.cumprod(1 + base * 0.01), index=dates)
    prices_b = pd.Series(100 * np.cumprod(1 + combined * 0.01), index=dates)
    return prices_a, prices_b


def test_correlation_above_reduce_threshold_halves_size():
    prices_existing, prices_new = _correlated_price_series(target_corr=0.78)
    rm = RiskManager(RISK_CONFIG)
    positions = {"SPY": PositionInfo("SPY", 10, 100.0, 100.0)}
    portfolio = _portfolio(positions=positions, price_history={"SPY": prices_existing, "AAPL": prices_new})

    # Tight stop so sizing caps don't also bind — isolates the correlation halving.
    signal = _signal(entry=100.0, stop=99.9, position_size_pct=0.10, leverage=1.0)
    decision = rm.validate_signal(signal, portfolio, now=NOW)
    assert decision.approved
    assert decision.modified_signal.position_size_pct == pytest.approx(0.05, abs=1e-6)
    assert any("correlation" in m for m in decision.modifications)


def test_correlation_above_reject_threshold_rejects():
    prices_existing, prices_new = _correlated_price_series(target_corr=0.97)
    rm = RiskManager(RISK_CONFIG)
    positions = {"SPY": PositionInfo("SPY", 10, 100.0, 100.0)}
    portfolio = _portfolio(positions=positions, price_history={"SPY": prices_existing, "AAPL": prices_new})

    signal = _signal(entry=100.0, stop=99.9, position_size_pct=0.10, leverage=1.0)
    decision = rm.validate_signal(signal, portfolio, now=NOW)
    assert not decision.approved
    assert "correlation" in decision.rejection_reason


# ---------------------------------------------------------------------------
# Sector concentration
# ---------------------------------------------------------------------------

def test_sector_concentration_rejects_over_limit():
    rm = RiskManager(RISK_CONFIG)
    # 290 shares @ $100 = $29,000 = 29% of $100,000 equity.
    positions = {"MSFT": PositionInfo("MSFT", 290, 100.0, 100.0, sector="tech")}
    portfolio = _portfolio(positions=positions)
    signal = _signal(symbol="GOOGL", entry=100.0, stop=99.9, position_size_pct=0.10, leverage=1.0,
                      metadata={"sector": "tech"})
    decision = rm.validate_signal(signal, portfolio, now=NOW)
    assert not decision.approved
    assert "sector" in decision.rejection_reason


def test_sector_concentration_skipped_when_sector_unknown():
    rm = RiskManager(RISK_CONFIG)
    positions = {"MSFT": PositionInfo("MSFT", 290, 100.0, 100.0, sector="tech")}
    portfolio = _portfolio(positions=positions)
    signal = _signal(symbol="GOOGL", entry=100.0, stop=99.9, position_size_pct=0.05, leverage=1.0)  # no sector metadata
    decision = rm.validate_signal(signal, portfolio, now=NOW)
    assert decision.approved


# ---------------------------------------------------------------------------
# Exposure caps
# ---------------------------------------------------------------------------

def test_exposure_cap_reduces_size_near_max_exposure():
    rm = RiskManager(RISK_CONFIG)
    # Existing exposure already at 75% of equity; max_exposure is 80% ->
    # only 5% of room left for a new (non-leveraged) position.
    positions = {"MSFT": PositionInfo("MSFT", 750, 100.0, 100.0)}  # 75,000 / 100,000 = 75%
    portfolio = _portfolio(positions=positions)
    signal = _signal(symbol="GOOGL", entry=100.0, stop=99.9, position_size_pct=0.50, leverage=1.0)
    decision = rm.validate_signal(signal, portfolio, now=NOW)
    assert decision.approved
    assert decision.modified_signal.position_size_pct <= 0.05 + 1e-6
    assert any("exposure cap" in m for m in decision.modifications)
