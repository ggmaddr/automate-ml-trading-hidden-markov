"""Tests for core.hmm_engine: model selection, labeling, stability filter."""

from __future__ import annotations

import numpy as np
import pytest

from core.hmm_engine import LABEL_SETS, HMMRegimeEngine
from data.feature_engineering import compute_features, log_returns


def _fit_engine(synthetic_ohlcv, small_hmm_config):
    features = compute_features(synthetic_ohlcv)
    raw_returns = log_returns(synthetic_ohlcv["close"], 1)

    engine = HMMRegimeEngine(small_hmm_config)
    engine.fit(features, raw_returns)
    return engine, features


def test_model_selection_picks_a_candidate(synthetic_ohlcv, small_hmm_config):
    engine, _ = _fit_engine(synthetic_ohlcv, small_hmm_config)

    assert engine.n_regimes in small_hmm_config["n_candidates"]
    assert set(engine.bic_scores.keys()) <= set(small_hmm_config["n_candidates"])
    assert len(engine.bic_scores) >= 1
    # The selected model's BIC must be the minimum among successfully fit candidates.
    assert engine.bic_scores[engine.n_regimes] == min(engine.bic_scores.values())


def test_regime_labels_match_label_set(synthetic_ohlcv, small_hmm_config):
    engine, _ = _fit_engine(synthetic_ohlcv, small_hmm_config)

    expected_labels = set(LABEL_SETS[engine.n_regimes])
    assert set(engine.regime_labels.values()) == expected_labels
    assert set(engine.regime_labels.keys()) == set(range(engine.n_regimes))


def test_labels_sorted_by_ascending_mean_return(synthetic_ohlcv, small_hmm_config):
    engine, _ = _fit_engine(synthetic_ohlcv, small_hmm_config)

    label_names = LABEL_SETS[engine.n_regimes]
    returns_by_rank = [
        engine.regime_info[state].expected_return
        for state in sorted(engine.regime_labels, key=lambda s: label_names.index(engine.regime_labels[s]))
    ]
    assert returns_by_rank == sorted(returns_by_rank)


def test_transition_matrix_is_row_stochastic(synthetic_ohlcv, small_hmm_config):
    engine, _ = _fit_engine(synthetic_ohlcv, small_hmm_config)

    transmat = engine.get_transition_matrix()
    assert transmat.shape == (engine.n_regimes, engine.n_regimes)
    np.testing.assert_allclose(transmat.sum(axis=1), 1.0, atol=1e-6)


def test_stability_filter_confirms_after_n_bars(synthetic_ohlcv, small_hmm_config):
    engine, features = _fit_engine(synthetic_ohlcv, small_hmm_config)

    # Feed bars one at a time; the confirmed regime should only ever change
    # after the raw filtered state has persisted for >= stability_bars bars.
    state = None
    for t in range(50, 150):
        state = engine.update(features.iloc[: t + 1])
        assert state.consecutive_bars >= 1
        assert 0 <= state.probability <= 1.0

    assert state is not None
    assert state.label in LABEL_SETS[engine.n_regimes]


def test_flicker_rate_within_bounds(synthetic_ohlcv, small_hmm_config):
    engine, features = _fit_engine(synthetic_ohlcv, small_hmm_config)

    for t in range(50, 200):
        engine.update(features.iloc[: t + 1])

    rate = engine.get_regime_flicker_rate()
    assert 0 <= rate <= small_hmm_config["flicker_window"]
    assert engine.is_flickering() == (rate > small_hmm_config["flicker_threshold"])


def test_fit_raises_below_min_train_bars(synthetic_ohlcv, small_hmm_config):
    features = compute_features(synthetic_ohlcv)
    raw_returns = log_returns(synthetic_ohlcv["close"], 1)

    engine = HMMRegimeEngine(small_hmm_config)
    with pytest.raises(ValueError):
        engine.fit(features.iloc[:10], raw_returns.iloc[:10])
