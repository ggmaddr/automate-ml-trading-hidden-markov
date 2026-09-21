"""Gaussian HMM volatility-regime classifier.

The HMM classifies market VOLATILITY regime (calm / moderate / turbulent).
It does not predict price direction — regime labels are assigned post-hoc by
sorting states by mean return purely for human readability. The strategy
layer decides allocation from the volatility classification, independent of
the return-based label.

The single most important correctness property of this module: regime
inference for bar t must use only observations up to and including t. See
`predict_regime_filtered` and tests/test_look_ahead.py.
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Union

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

logger = logging.getLogger(__name__)

# Regime name sets, ordered by ascending mean return, keyed by n_regimes.
LABEL_SETS: dict[int, list[str]] = {
    3: ["BEAR", "NEUTRAL", "BULL"],
    4: ["CRASH", "BEAR", "BULL", "EUPHORIA"],
    5: ["CRASH", "BEAR", "NEUTRAL", "BULL", "EUPHORIA"],
    6: ["CRASH", "STRONG_BEAR", "WEAK_BEAR", "WEAK_BULL", "STRONG_BULL", "EUPHORIA"],
    7: ["CRASH", "STRONG_BEAR", "WEAK_BEAR", "NEUTRAL", "WEAK_BULL", "STRONG_BULL", "EUPHORIA"],
}


@dataclass
class RegimeInfo:
    """Static metadata describing one trained regime/state."""

    regime_id: int
    regime_name: str
    expected_return: float
    expected_volatility: float
    recommended_strategy_type: str  # "low_vol" | "mid_vol" | "high_vol"
    max_leverage_allowed: float
    max_position_size_pct: float
    min_confidence_to_act: float


@dataclass
class RegimeState:
    """A single point-in-time regime observation, after the stability filter."""

    label: str
    state_id: int
    probability: float
    state_probabilities: np.ndarray
    timestamp: object
    is_confirmed: bool
    consecutive_bars: int


class HMMRegimeEngine:
    """Trains a GaussianHMM volatility classifier and runs filtered inference."""

    def __init__(self, config: dict) -> None:
        self.config = config

        self.model: Optional[GaussianHMM] = None
        self.n_regimes: Optional[int] = None
        self.bic_scores: dict[int, float] = {}
        self.regime_labels: dict[int, str] = {}
        self.regime_info: dict[int, RegimeInfo] = {}
        self.training_date: Optional[datetime] = None
        self.feature_columns: Optional[list[str]] = None
        self._last_fit_X: Optional[np.ndarray] = None

        # Cached log-space parameters, populated by _cache_log_params() after fit/load.
        self._log_startprob: Optional[np.ndarray] = None
        self._log_transmat: Optional[np.ndarray] = None

        self._reset_stability_state()

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, features: pd.DataFrame, returns: pd.Series) -> None:
        """Train with automatic model selection over `hmm.n_candidates` by BIC.

        `returns` must be per-bar (non-standardized) returns aligned to
        `features.index` — e.g. 1-period log returns. It is used ONLY for
        the one-time post-training labeling step (sorting states by mean
        return); it never touches the live/backtest inference path.
        """
        n_samples = len(features)
        min_train_bars = self.config["min_train_bars"]
        if n_samples < min_train_bars:
            raise ValueError(
                f"Need at least {min_train_bars} bars to train, got {n_samples}"
            )

        aligned_returns = returns.reindex(features.index)
        if aligned_returns.isna().any():
            raise ValueError("returns must be fully aligned to features.index (no NaNs)")

        X = features.to_numpy()
        n_init = self.config["n_init"]
        covariance_type = self.config["covariance_type"]

        best_model: Optional[GaussianHMM] = None
        best_bic = np.inf
        best_k: Optional[int] = None
        bic_scores: dict[int, float] = {}

        for k in self.config["n_candidates"]:
            best_ll_for_k = -np.inf
            best_model_for_k: Optional[GaussianHMM] = None

            for init_idx in range(n_init):
                candidate = GaussianHMM(
                    n_components=k,
                    covariance_type=covariance_type,
                    n_iter=200,
                    tol=1e-4,
                    random_state=init_idx,
                )
                try:
                    candidate.fit(X)
                    ll = candidate.score(X)
                except Exception as exc:  # noqa: BLE001 - log and skip a bad init
                    logger.warning("HMM fit failed for k=%d init=%d: %s", k, init_idx, exc)
                    continue

                if ll > best_ll_for_k:
                    best_ll_for_k = ll
                    best_model_for_k = candidate

            if best_model_for_k is None:
                logger.warning("All initializations failed for k=%d, skipping", k)
                continue

            n_params = self._count_free_params(k, X.shape[1], covariance_type)
            bic = -2.0 * best_ll_for_k + n_params * np.log(n_samples)
            bic_scores[k] = bic
            logger.info(
                "HMM candidate k=%d: log_likelihood=%.2f n_params=%d BIC=%.2f",
                k, best_ll_for_k, n_params, bic,
            )

            if bic < best_bic:
                best_bic = bic
                best_model = best_model_for_k
                best_k = k

        if best_model is None or best_k is None:
            raise RuntimeError("HMM training failed for all candidate values of n_components")

        logger.info("Selected n_components=%d (BIC=%.2f) among candidates %s", best_k, best_bic, bic_scores)

        self.model = best_model
        self.n_regimes = best_k
        self.bic_scores = bic_scores
        self.feature_columns = list(features.columns)
        self.training_date = datetime.now()
        self._last_fit_X = X

        self._cache_log_params()
        self._label_regimes(aligned_returns)
        self._reset_stability_state()

    @staticmethod
    def _count_free_params(n_states: int, n_features: int, covariance_type: str) -> int:
        """Free parameter count for a GaussianHMM, for BIC."""
        n_start = n_states - 1
        n_trans = n_states * (n_states - 1)
        n_means = n_states * n_features
        if covariance_type == "full":
            n_cov = n_states * n_features * (n_features + 1) // 2
        elif covariance_type == "diag":
            n_cov = n_states * n_features
        elif covariance_type == "tied":
            n_cov = n_features * (n_features + 1) // 2
        elif covariance_type == "spherical":
            n_cov = n_states
        else:
            raise ValueError(f"Unknown covariance_type: {covariance_type}")
        return n_start + n_trans + n_means + n_cov

    def _label_regimes(self, returns: pd.Series) -> None:
        """Assign human-readable labels by sorting states by mean return.

        Uses Viterbi decoding (`model.predict`) purely to assign each
        TRAINING bar to a state for computing descriptive statistics. This
        is a one-time, training-only step — it is never used for live or
        backtest regime inference (see predict_regime_filtered).
        """
        # Decode against the model's own training features, not `returns` itself —
        # `returns` only supplies the per-bar value used to rank states afterward.
        state_sequence = self.model.predict(self._last_fit_X)

        return_values = returns.to_numpy()
        mean_return = {}
        mean_vol = {}
        for state in range(self.n_regimes):
            mask = state_sequence == state
            state_returns = return_values[mask]
            mean_return[state] = float(state_returns.mean()) if state_returns.size else 0.0
            mean_vol[state] = float(state_returns.std(ddof=0)) if state_returns.size else 0.0

        # Labels: sorted by mean return ascending (human-readable only).
        states_by_return = sorted(range(self.n_regimes), key=lambda s: mean_return[s])
        label_names = LABEL_SETS[self.n_regimes]
        self.regime_labels = {
            state: label_names[rank] for rank, state in enumerate(states_by_return)
        }

        # Strategy-relevant metadata is ranked by VOLATILITY, independently
        # of the return-based label (see module docstring).
        states_by_vol = sorted(range(self.n_regimes), key=lambda s: mean_vol[s])
        n = self.n_regimes
        # This engine only owns the `hmm:` config section, so strategy/risk
        # tiering here uses documented defaults rather than reading the
        # strategy/risk config sections directly. The strategy layer applies
        # the authoritative allocation/leverage/sizing numbers from its own
        # config; these RegimeInfo values are descriptive metadata only.
        low_vol_leverage = 1.25
        max_single_position = 0.15
        min_confidence = self.config.get("min_confidence", 0.55)

        vol_rank = {state: rank for rank, state in enumerate(states_by_vol)}
        self.regime_info = {}
        for state in range(self.n_regimes):
            rank = vol_rank[state]
            tier_frac = rank / max(n - 1, 1)  # 0.0 = calmest, 1.0 = most turbulent

            if tier_frac <= 1 / 3:
                strategy_type = "low_vol"
            elif tier_frac <= 2 / 3:
                strategy_type = "mid_vol"
            else:
                strategy_type = "high_vol"

            self.regime_info[state] = RegimeInfo(
                regime_id=state,
                regime_name=self.regime_labels[state],
                expected_return=mean_return[state],
                expected_volatility=mean_vol[state],
                recommended_strategy_type=strategy_type,
                max_leverage_allowed=low_vol_leverage if rank == 0 else 1.0,
                max_position_size_pct=max_single_position * (1.0 - 0.5 * tier_frac),
                min_confidence_to_act=min_confidence + 0.1 * tier_frac,
            )

    # ------------------------------------------------------------------
    # Filtered (forward-algorithm) inference — NO LOOK-AHEAD BIAS
    # ------------------------------------------------------------------

    def _log_emission(self, X: np.ndarray) -> np.ndarray:
        """log P(observation_t | state) for every t, state. Shape (T, n_states)."""
        n_states = self.model.n_components
        T = X.shape[0]
        log_emission = np.empty((T, n_states))
        for state in range(n_states):
            mean = self.model.means_[state]
            cov = self.model.covars_[state]
            log_emission[:, state] = multivariate_normal.logpdf(X, mean=mean, cov=cov)
        return log_emission

    def _forward_log_alpha(self, X: np.ndarray) -> np.ndarray:
        """Full forward pass in log space. Depends only on X[0:t] for row t.

        alpha_0 = startprob * emission(obs_0)
        alpha_t = (alpha_{t-1} @ transmat) * emission(obs_t)
        computed in log space via logsumexp for numerical stability.
        """
        if self.model is None:
            raise RuntimeError("Model has not been trained or loaded yet")

        log_emission = self._log_emission(X)
        T, n_states = log_emission.shape
        log_alpha = np.empty((T, n_states))

        log_alpha[0] = self._log_startprob + log_emission[0]
        for t in range(1, T):
            # temp[i, j] = log_alpha[t-1, i] + log P(state i -> state j)
            temp = log_alpha[t - 1][:, None] + self._log_transmat
            log_alpha[t] = logsumexp(temp, axis=0) + log_emission[t]

        return log_alpha

    def predict_regime_filtered(self, features_up_to_now: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Filtered state path: state_t = argmax P(state_t | obs_1:t).

        Pure function of the input array — depends only on rows present in
        `features_up_to_now`, never on data outside it and never on any
        cached state from a previous call. This is what makes the
        no-look-ahead guarantee testable: calling this with data[0:T] and
        with data[0:T+k] must agree on every index < T.
        """
        X = self._to_array(features_up_to_now)
        log_alpha = self._forward_log_alpha(X)
        return log_alpha.argmax(axis=1)

    def predict_regime_proba(self, features_up_to_now: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Filtered state probability distribution for every t. Shape (T, n_states)."""
        X = self._to_array(features_up_to_now)
        log_alpha = self._forward_log_alpha(X)
        log_norm = logsumexp(log_alpha, axis=1, keepdims=True)
        return np.exp(log_alpha - log_norm)

    @staticmethod
    def _to_array(features: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        if isinstance(features, pd.DataFrame):
            return features.to_numpy()
        return np.asarray(features)

    def _cache_log_params(self) -> None:
        with np.errstate(divide="ignore"):
            self._log_startprob = np.log(self.model.startprob_)
            self._log_transmat = np.log(self.model.transmat_)

    # ------------------------------------------------------------------
    # Regime stability filter / flicker detection (stateful, for live use)
    # ------------------------------------------------------------------

    def _reset_stability_state(self) -> None:
        self._confirmed_state: Optional[int] = None
        self._pending_state: Optional[int] = None
        self._pending_count: int = 0
        self._consecutive_bars: int = 0
        self._raw_state_history: list[int] = []
        self._last_change_confirmed: bool = False

    def update(self, features_up_to_now: pd.DataFrame) -> RegimeState:
        """Advance the stability filter by one bar and return the current RegimeState.

        Regime changes are only "confirmed" after the raw filtered state
        persists for `hmm.stability_bars` consecutive bars. Until then the
        previously confirmed regime is kept (with is_confirmed=False,
        signalling the caller to reduce sizing during the transition).
        """
        X = self._to_array(features_up_to_now)
        log_alpha = self._forward_log_alpha(X)
        log_norm = logsumexp(log_alpha[-1])
        proba = np.exp(log_alpha[-1] - log_norm)
        raw_state = int(np.argmax(proba))

        self._raw_state_history.append(raw_state)
        window = self.config["flicker_window"]
        if len(self._raw_state_history) > window * 2:
            self._raw_state_history = self._raw_state_history[-window * 2:]

        stability_bars = self.config["stability_bars"]
        changed = False

        if self._confirmed_state is None:
            self._confirmed_state = raw_state
            self._consecutive_bars = 1
            self._pending_state = None
            self._pending_count = 0
        elif raw_state == self._confirmed_state:
            self._consecutive_bars += 1
            self._pending_state = None
            self._pending_count = 0
        else:
            if self._pending_state == raw_state:
                self._pending_count += 1
            else:
                self._pending_state = raw_state
                self._pending_count = 1

            if self._pending_count >= stability_bars:
                self._confirmed_state = raw_state
                self._consecutive_bars = self._pending_count
                self._pending_state = None
                self._pending_count = 0
                changed = True

        self._last_change_confirmed = changed
        flickering = self.is_flickering()
        is_confirmed = (self._pending_state is None) and not flickering

        label = self.regime_labels.get(self._confirmed_state, "UNKNOWN")
        if changed:
            logger.warning("Regime change confirmed: -> %s (state %d)", label, self._confirmed_state)
        else:
            logger.info(
                "Regime %s (state %d), consecutive_bars=%d, confirmed=%s",
                label, self._confirmed_state, self._consecutive_bars, is_confirmed,
            )

        timestamp = features_up_to_now.index[-1] if isinstance(features_up_to_now, pd.DataFrame) else None

        return RegimeState(
            label=label,
            state_id=self._confirmed_state,
            probability=float(proba[raw_state]),
            state_probabilities=proba,
            timestamp=timestamp,
            is_confirmed=is_confirmed,
            consecutive_bars=self._consecutive_bars,
        )

    def get_regime_stability(self) -> int:
        """Consecutive bars the currently confirmed regime has persisted."""
        return self._consecutive_bars

    def get_transition_matrix(self) -> np.ndarray:
        """Learned state transition probability matrix."""
        return self.model.transmat_.copy()

    def detect_regime_change(self) -> bool:
        """True only if the most recent `update()` call confirmed a NEW regime."""
        return self._last_change_confirmed

    def get_regime_flicker_rate(self) -> int:
        """Number of raw (unconfirmed) regime changes within the trailing flicker_window bars."""
        window = self.config["flicker_window"]
        recent = self._raw_state_history[-window:]
        return sum(1 for prev, curr in zip(recent, recent[1:]) if prev != curr)

    def is_flickering(self) -> bool:
        """True if the flicker rate exceeds hmm.flicker_threshold (forces uncertainty mode)."""
        return self.get_regime_flicker_rate() > self.config["flicker_threshold"]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        payload = {
            "model": self.model,
            "n_regimes": self.n_regimes,
            "bic_scores": self.bic_scores,
            "regime_labels": self.regime_labels,
            "regime_info": self.regime_info,
            "training_date": self.training_date,
            "feature_columns": self.feature_columns,
            "config": self.config,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)

    @classmethod
    def load(cls, path: str) -> "HMMRegimeEngine":
        with open(path, "rb") as f:
            payload = pickle.load(f)
        engine = cls(payload["config"])
        engine.model = payload["model"]
        engine.n_regimes = payload["n_regimes"]
        engine.bic_scores = payload["bic_scores"]
        engine.regime_labels = payload["regime_labels"]
        engine.regime_info = payload["regime_info"]
        engine.training_date = payload["training_date"]
        engine.feature_columns = payload["feature_columns"]
        engine._cache_log_params()
        engine._reset_stability_state()
        return engine
