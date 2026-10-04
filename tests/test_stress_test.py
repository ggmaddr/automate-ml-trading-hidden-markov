"""Tests for backtest.stress_test. Uses tiny n_simulations + the cheap
small_hmm_config fixture since these re-run the full walk-forward backtest
per simulation and are slow by design.
"""

from __future__ import annotations

import numpy as np
import pytest

from backtest.backtester import Backtester
from backtest.stress_test import (
    inject_crash,
    inject_gaps,
    run_crash_stress_test,
    run_gap_stress_test,
    run_regime_shuffle_test,
)


def test_inject_crash_applies_exact_magnitude_from_event_onward(synthetic_ohlcv):
    rng = np.random.default_rng(42)
    # min == max magnitude removes randomness from the magnitude draw, only
    # the event date is random — lets us assert an EXACT before/after ratio.
    crashed = inject_crash(synthetic_ohlcv, rng, n_events=1, min_magnitude=0.05, max_magnitude=0.05)

    ratio = (crashed["close"] / synthetic_ohlcv["close"]).round(6)
    assert set(ratio.unique()) <= {1.0, 0.95}
    assert 0.95 in ratio.unique()


def test_inject_crash_zero_events_is_a_noop(synthetic_ohlcv):
    rng = np.random.default_rng(0)
    out = inject_crash(synthetic_ohlcv.iloc[:40], rng, n_events=10)  # too short for eligible dates
    pd_equal = (out == synthetic_ohlcv.iloc[:40]).all().all()
    assert pd_equal


def test_inject_gaps_changes_prices_after_event(synthetic_ohlcv):
    rng = np.random.default_rng(1)
    gapped = inject_gaps(synthetic_ohlcv, rng, n_events=3)
    assert not gapped["close"].equals(synthetic_ohlcv["close"])
    assert gapped.shape == synthetic_ohlcv.shape


def test_crash_stress_test_smoke(synthetic_ohlcv, backtest_config):
    result = run_crash_stress_test({"TEST": synthetic_ohlcv}, backtest_config, n_simulations=2, seed=0)
    assert result["n_simulations"] == 2
    assert result["mean_max_loss"] <= 0.0
    assert result["worst_case_loss"] <= result["mean_max_loss"]
    assert 0.0 <= result["pct_circuit_breaker_fired"] <= 1.0


def test_gap_stress_test_smoke(synthetic_ohlcv, backtest_config):
    result = run_gap_stress_test({"TEST": synthetic_ohlcv}, backtest_config, n_simulations=2, seed=1)
    assert result["baseline_max_drawdown"] <= 0.0
    assert result["mean_perturbed_max_drawdown"] <= 0.0


def test_regime_shuffle_test_smoke(synthetic_ohlcv, backtest_config):
    base_result = Backtester(backtest_config).run({"TEST": synthetic_ohlcv})
    result = run_regime_shuffle_test({"TEST": synthetic_ohlcv}, base_result, backtest_config, n_shuffles=3)
    assert result["n_shuffles"] == 3
    assert result["worst_shuffled_max_drawdown"] <= result["mean_shuffled_max_drawdown"]
    assert isinstance(result["contained"], (bool, np.bool_))
