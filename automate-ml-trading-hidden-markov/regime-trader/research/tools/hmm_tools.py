"""Read-only HMM snapshot / regime-history tools.

Nothing here trains a model or touches live trading state. It loads its OWN
copy of the saved model (`hmm_model.pkl`), runs the existing causal forward
filter over recent bars, and replays the existing stability/flicker filter
(`HMMRegimeEngine.update`) to recover consecutive-bars / flicker metadata —
exactly what the live bot computes, on a throwaway engine instance.
"""

from __future__ import annotations

import copy
import hashlib
import logging
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

from core.hmm_engine import HMMRegimeEngine
from data.feature_engineering import compute_features
from research.config import PROJECT_ROOT
from research.exceptions import QuantContextUnavailableError, ToolError
from research.schemas import QuantContext, RegimeHistory, RegimeHistoryPoint
from research.tools.market_tools import fetch_daily_bars

DEFAULT_MODEL_PATH = PROJECT_ROOT / "hmm_model.pkl"
RECENT_WINDOW_BARS = 20


class _PrecomputedAlphaEngine(HMMRegimeEngine):
    """Replays `update()` over a whole history without N forward passes.

    The forward filter is causal: alpha for rows 0..t depends only on rows
    0..t. So the filtered probabilities `predict_regime_proba` returns for
    the full history, sliced to the first t+1 rows, are exactly what a
    fresh forward pass over features[:t+1] would produce. Only
    `_forward_log_alpha` is overridden; the stability/flicker logic in
    `update()` is the engine's own, unmodified.
    """

    _precomputed_proba: np.ndarray

    def _forward_log_alpha(self, X: np.ndarray) -> np.ndarray:
        with np.errstate(divide="ignore"):
            return np.log(self._precomputed_proba[: len(X)])


@contextmanager
def _quiet_engine_logs():
    """`update()` logs a WARNING on every confirmed regime change; during a
    replay that would spam stderr with historical (non-live) events."""
    engine_logger = logging.getLogger("core.hmm_engine")
    previous = engine_logger.level
    engine_logger.setLevel(logging.CRITICAL)
    try:
        yield
    finally:
        engine_logger.setLevel(previous)


class _Replay:
    def __init__(self, context: QuantContext, history: RegimeHistory) -> None:
        self.context = context
        self.history = history


class HMMRuntimeReader:
    """Builds QuantContext / RegimeHistory for a symbol from the saved HMM."""

    def __init__(
        self,
        model_path: Path | str = DEFAULT_MODEL_PATH,
        bars_provider: Callable[[str], pd.DataFrame] = fetch_daily_bars,
        engine_loader: Callable[[str], HMMRegimeEngine] = HMMRegimeEngine.load,
        memo_seconds: float = 300.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.model_path = Path(model_path)
        self._bars_provider = bars_provider
        self._engine_loader = engine_loader
        self._memo_seconds = memo_seconds
        self._clock = clock
        self._memo: dict[str, tuple[float, _Replay]] = {}

    # ------------------------------------------------------------------

    def _replay(self, symbol: str) -> _Replay:
        hit = self._memo.get(symbol)
        if hit and self._clock() - hit[0] < self._memo_seconds:
            return hit[1]

        if not self.model_path.exists():
            raise QuantContextUnavailableError(
                f"no trained HMM at {self.model_path} — run `python main.py run --train-only` first"
            )
        try:
            engine = self._engine_loader(str(self.model_path))
        except Exception as exc:  # noqa: BLE001 - corrupt/incompatible pickle -> controlled error
            raise QuantContextUnavailableError(f"could not load HMM model: {type(exc).__name__}") from exc
        if engine.model is None or not engine.regime_info:
            raise QuantContextUnavailableError("loaded HMM model is untrained")

        bars = self._bars_provider(symbol)
        features = compute_features(bars)
        if len(features) < 2:
            raise QuantContextUnavailableError(f"not enough bars for {symbol} to compute HMM features")
        if engine.feature_columns and list(features.columns) != list(engine.feature_columns):
            raise QuantContextUnavailableError("feature set does not match the saved HMM model")

        replayer = copy.copy(engine)
        replayer.__class__ = _PrecomputedAlphaEngine
        replayer._precomputed_proba = engine.predict_regime_proba(features)
        replayer._reset_stability_state()

        points: list[RegimeHistoryPoint] = []
        last_state = None
        with _quiet_engine_logs():
            for t in range(len(features)):
                state = replayer.update(features.iloc[: t + 1])
                info = engine.regime_info.get(state.state_id)
                points.append(
                    RegimeHistoryPoint(
                        timestamp=pd.Timestamp(features.index[t]).to_pydatetime(),
                        regime_id=state.state_id,
                        label=state.label,
                        probability=state.probability,
                        expected_volatility=info.expected_volatility if info else None,
                        is_confirmed=state.is_confirmed,
                        consecutive_bars=state.consecutive_bars,
                        flicker_rate=float(replayer.get_regime_flicker_rate()),
                        is_flickering=replayer.is_flickering(),
                    )
                )
                last_state = state

        context = self._build_context(symbol, engine, replayer, last_state, features, bars)
        replay = _Replay(context, RegimeHistory(symbol=symbol, points=points))
        self._memo[symbol] = (self._clock(), replay)
        return replay

    @staticmethod
    def _build_context(symbol, engine, replayer, state, features, bars) -> QuantContext:
        info = engine.regime_info.get(state.state_id)
        vols = sorted(i.expected_volatility for i in engine.regime_info.values())
        n = len(vols)
        vol_rank = (vols.index(info.expected_volatility) / max(n - 1, 1)) if info and n else None

        closes = bars["close"].astype(float)
        recent_return = recent_vol = None
        if len(closes) > RECENT_WINDOW_BARS:
            recent_return = float(closes.iloc[-1] / closes.iloc[-1 - RECENT_WINDOW_BARS] - 1.0)
            log_ret = np.log(closes / closes.shift(1)).dropna().iloc[-RECENT_WINDOW_BARS:]
            recent_vol = float(log_ret.std(ddof=0))

        feature_hash = hashlib.sha1(",".join(engine.feature_columns or []).encode()).hexdigest()[:8]
        trained = engine.training_date.strftime("%Y-%m-%d") if engine.training_date else "unknown"
        return QuantContext(
            symbol=symbol,
            timestamp=pd.Timestamp(features.index[-1]).to_pydatetime(),
            regime_id=state.state_id,
            regime_label=state.label,
            regime_probability=float(state.probability),
            state_probabilities={
                engine.regime_labels.get(i, str(i)): float(p) for i, p in enumerate(state.state_probabilities)
            },
            expected_volatility=info.expected_volatility if info else None,
            expected_return=info.expected_return if info else None,
            volatility_rank=vol_rank,
            is_confirmed=bool(state.is_confirmed),
            consecutive_bars=int(state.consecutive_bars),
            flicker_rate=float(replayer.get_regime_flicker_rate()),
            is_flickering=bool(replayer.is_flickering()),
            model_version=f"hmm-k{engine.n_regimes}-trained-{trained}",
            feature_version=f"{len(engine.feature_columns or [])}f-{feature_hash}",
            recent_return=recent_return,
            recent_volatility=recent_vol,
        )

    # ------------------------------------------------------------------
    # Tools
    # ------------------------------------------------------------------

    def get_hmm_snapshot(self, symbol: str) -> QuantContext:
        """Latest QuantContext for `symbol`, from the existing saved model."""
        return self._replay(symbol.upper()).context

    def get_regime_history(self, symbol: str, bars: int = 60) -> RegimeHistory:
        """The most recent `bars` regime observations (oldest first)."""
        if bars < 1:
            raise ToolError("bars must be >= 1")
        history = self._replay(symbol.upper()).history
        return RegimeHistory(symbol=history.symbol, points=history.points[-bars:])
