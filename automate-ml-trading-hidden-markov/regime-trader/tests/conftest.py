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
