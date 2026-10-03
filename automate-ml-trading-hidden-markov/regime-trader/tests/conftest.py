"""Shared fixtures for regime-trader tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _make_synthetic_ohlcv(n_bars: int = 1000, seed: int = 7) -> pd.DataFrame:
    """Synthetic daily OHLCV with alternating calm/turbulent volatility regimes.

    Deterministic (fixed seed) so tests are reproducible. Regimes alternate
    in blocks so the HMM has genuinely distinguishable states to learn.
    """
    rng = np.random.default_rng(seed)

    block_size = 60
    n_blocks = n_bars // block_size + 1
    log_returns = np.empty(0)
    volumes = np.empty(0)

    for i in range(n_blocks):
        calm = i % 2 == 0
        if calm:
            drift, vol, vol_base = 0.0006, 0.006, 1_000_000
        else:
            drift, vol, vol_base = -0.0010, 0.028, 2_500_000

        block_returns = rng.normal(drift, vol, block_size)
        block_volume = rng.lognormal(mean=np.log(vol_base), sigma=0.25, size=block_size)
        log_returns = np.concatenate([log_returns, block_returns])
        volumes = np.concatenate([volumes, block_volume])

    log_returns = log_returns[:n_bars]
    volumes = volumes[:n_bars]

    close = 100.0 * np.exp(np.cumsum(log_returns))
    open_ = np.empty(n_bars)
    open_[0] = close[0]
    open_[1:] = close[:-1]

    intraday_noise = rng.uniform(0.001, 0.01, n_bars)
    high = np.maximum(open_, close) * (1 + intraday_noise)
    low = np.minimum(open_, close) * (1 - intraday_noise)

    dates = pd.bdate_range("2018-01-02", periods=n_bars)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volumes},
        index=dates,
    )


@pytest.fixture(scope="session")
def synthetic_ohlcv() -> pd.DataFrame:
    return _make_synthetic_ohlcv()


@pytest.fixture(scope="session")
def small_hmm_config() -> dict:
    """A cheap `hmm:` config section for fast tests (few candidates/inits)."""
    return {
        "n_candidates": [2, 3],
        "n_init": 2,
        "covariance_type": "diag",
        "min_train_bars": 100,
        "stability_bars": 3,
        "flicker_window": 20,
        "flicker_threshold": 4,
        "min_confidence": 0.55,
    }


@pytest.fixture(scope="session")
def trained_hmm(synthetic_ohlcv, small_hmm_config):
    """A real, already-fitted HMMRegimeEngine — expensive to build, reused
    across tests. Mutate its stability-filter state? Use `fresh_trained_hmm`
    instead so tests don't leak state into each other.
    """
    from core.hmm_engine import HMMRegimeEngine
    from data.feature_engineering import compute_features, log_returns

    features = compute_features(synthetic_ohlcv)
    returns = log_returns(synthetic_ohlcv["close"], 1)
    engine = HMMRegimeEngine(small_hmm_config)
    engine.fit(features, returns)
    return engine


@pytest.fixture
def fresh_trained_hmm(trained_hmm):
    """The shared trained_hmm, with its stability/flicker filter reset so
    each test starts from a clean state without paying to refit the model.
    """
    trained_hmm._reset_stability_state()
    return trained_hmm


@pytest.fixture(scope="session")
def backtest_config(small_hmm_config) -> dict:
    """A full settings dict (hmm/strategy/backtest/risk) sized for fast tests.

    train_window + test_window (180) fits several times inside the ~550
    valid feature rows synthetic_ohlcv produces, so walk-forward continuity
    across window boundaries is actually exercised.
    """
    return {
        "hmm": small_hmm_config,
        "strategy": {
            "low_vol_allocation": 0.95,
            "mid_vol_allocation_trend": 0.95,
            "mid_vol_allocation_no_trend": 0.60,
            "high_vol_allocation": 0.60,
            "low_vol_leverage": 1.25,
            "rebalance_threshold": 0.10,
            "uncertainty_size_mult": 0.50,
            "min_confidence": 0.55,
        },
        "backtest": {
            "slippage_pct": 0.0005,
            "initial_capital": 100_000,
            "train_window": 120,
            "test_window": 60,
            "step_size": 60,
            "risk_free_rate": 0.045,
        },
        "risk": {
            "max_dd_from_peak": 0.10,
        },
    }


@pytest.fixture(scope="session")
def full_config(backtest_config) -> dict:
    """backtest_config plus the complete `risk:`/`monitoring:` sections a
    real RiskManager/AlertManager/TradingBot needs (backtest_config's risk
    section only carries the one key backtest.stress_test's placeholder
    circuit-breaker check uses).
    """
    return {
        **backtest_config,
        "risk": {
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
        },
        "monitoring": {
            "alert_rate_limit_minutes": 15,
            "dashboard_refresh_seconds": 5,
        },
    }
