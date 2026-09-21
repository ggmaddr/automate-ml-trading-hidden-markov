"""MANDATORY: verifies core.hmm_engine has no look-ahead bias.

predict_regime_filtered() must use the forward algorithm (filtered
inference), never model.predict() (Viterbi, which revises past states using
future observations). Regime at time T must be identical whether it is
computed from data[0:T] or from data[0:T+k] for any k > 0.
"""

from __future__ import annotations

from core.hmm_engine import HMMRegimeEngine
from data.feature_engineering import compute_features, log_returns


def test_no_look_ahead_bias(synthetic_ohlcv, small_hmm_config):
    features = compute_features(synthetic_ohlcv)
    raw_returns = log_returns(synthetic_ohlcv["close"], 1)

    engine = HMMRegimeEngine(small_hmm_config)
    engine.fit(features, raw_returns)

    regime_short = engine.predict_regime_filtered(features.iloc[0:400])[-1]
    regime_long = engine.predict_regime_filtered(features.iloc[0:500])[399]

    assert regime_short == regime_long, "LOOK-AHEAD BIAS DETECTED"


def test_filtered_path_agrees_on_every_shared_index(synthetic_ohlcv, small_hmm_config):
    """Stronger version: the full filtered path must agree on the entire overlap,
    not just the last index, since the recursion must be strictly causal.
    """
    features = compute_features(synthetic_ohlcv)
    raw_returns = log_returns(synthetic_ohlcv["close"], 1)

    engine = HMMRegimeEngine(small_hmm_config)
    engine.fit(features, raw_returns)

    short_path = engine.predict_regime_filtered(features.iloc[0:400])
    long_path = engine.predict_regime_filtered(features.iloc[0:500])

    assert list(short_path) == list(long_path[:400]), "LOOK-AHEAD BIAS DETECTED"
