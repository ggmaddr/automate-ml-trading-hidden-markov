"""Read-only tools: HMM snapshot/history mapping, backtest summary, registry."""

import copy
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core.hmm_engine import HMMRegimeEngine
from data.feature_engineering import compute_features
from research.exceptions import (
    QuantContextUnavailableError, ToolBudgetExceededError, ToolError, ToolRejectedError,
)
from research.tools import ALLOWED_RESEARCH_TOOLS, ResearchToolRegistry
from research.tools.backtest_tools import BacktestReader
from research.tools.hmm_tools import HMMRuntimeReader, _PrecomputedAlphaEngine
from research_fakes import StubBacktestReader, StubHMMReader


@pytest.fixture
def model_path(trained_hmm, tmp_path) -> Path:
    path = tmp_path / "hmm_model.pkl"
    trained_hmm.save(str(path))
    return path


@pytest.fixture
def reader(model_path, synthetic_ohlcv):
    return HMMRuntimeReader(model_path, bars_provider=lambda symbol: synthetic_ohlcv)


def test_snapshot_maps_engine_state(reader, model_path, synthetic_ohlcv):
    q = reader.get_hmm_snapshot("spy")
    engine = HMMRegimeEngine.load(str(model_path))
    assert q.symbol == "SPY"
    assert q.regime_id in engine.regime_info
    assert q.regime_label == engine.regime_labels[q.regime_id]
    # state probabilities preserved: one per regime, summing to 1
    assert set(q.state_probabilities) == set(engine.regime_labels.values())
    assert sum(q.state_probabilities.values()) == pytest.approx(1.0)
    # volatility metadata preserved from regime_info
    info = engine.regime_info[q.regime_id]
    assert q.expected_volatility == pytest.approx(info.expected_volatility)
    assert q.expected_return == pytest.approx(info.expected_return)
    assert 0.0 <= q.volatility_rank <= 1.0
    assert q.consecutive_bars >= 1 and q.flicker_rate is not None
    assert q.timestamp == compute_features(synthetic_ohlcv).index[-1].to_pydatetime()
    assert q.model_version and q.feature_version
    assert q.recent_return is not None and q.recent_volatility > 0


def test_replay_matches_real_engine_update_path(trained_hmm, synthetic_ohlcv):
    """The precomputed-alpha replay must equal calling the real update() per bar."""
    features = compute_features(synthetic_ohlcv).iloc[-90:]
    real = copy.copy(trained_hmm)
    real._reset_stability_state()
    fast = copy.copy(trained_hmm)
    fast.__class__ = _PrecomputedAlphaEngine
    fast._precomputed_proba = trained_hmm.predict_regime_proba(features)
    fast._reset_stability_state()
    for t in range(len(features)):
        a = real.update(features.iloc[: t + 1])
        b = fast.update(features.iloc[: t + 1])
        assert (a.state_id, a.is_confirmed, a.consecutive_bars) == (b.state_id, b.is_confirmed, b.consecutive_bars)
        assert a.probability == pytest.approx(b.probability)
        assert real.get_regime_flicker_rate() == fast.get_regime_flicker_rate()


def test_history_is_chronological_and_bounded(reader):
    h = reader.get_regime_history("SPY", bars=30)
    assert len(h.points) == 30
    ts = [p.timestamp for p in h.points]
    assert ts == sorted(ts) and len(set(ts)) == 30
    assert h.points[-1].timestamp == reader.get_hmm_snapshot("SPY").timestamp


def test_snapshot_does_not_mutate_saved_model_or_other_state(reader, model_path):
    before = hashlib.sha256(model_path.read_bytes()).hexdigest()
    reader.get_hmm_snapshot("SPY")
    reader.get_regime_history("SPY")
    assert hashlib.sha256(model_path.read_bytes()).hexdigest() == before


def test_snapshot_without_model_is_controlled_error(tmp_path, synthetic_ohlcv):
    r = HMMRuntimeReader(tmp_path / "missing.pkl", bars_provider=lambda s: synthetic_ohlcv)
    with pytest.raises(QuantContextUnavailableError):
        r.get_hmm_snapshot("SPY")


def test_snapshot_with_too_little_data_is_controlled_error(model_path, synthetic_ohlcv):
    r = HMMRuntimeReader(model_path, bars_provider=lambda s: synthetic_ohlcv.iloc[:50])
    with pytest.raises(QuantContextUnavailableError):
        r.get_hmm_snapshot("SPY")


# ---- backtest ---------------------------------------------------------------

def _write_backtest(dir_: Path, symbol="SPY"):
    dates = pd.bdate_range("2024-01-02", periods=60)
    equity = pd.Series(100_000 * (1 + 0.001 * np.arange(60)), index=dates)
    equity.to_frame(symbol).to_csv(dir_ / "per_symbol_equity.csv")
    pd.DataFrame({
        "symbol": [symbol, symbol], "decision_date": [dates[1], dates[30]], "fill_date": [dates[2], dates[31]],
        "old_allocation": [0.0, 0.6], "new_allocation": [0.6, 0.95], "fill_price": [100.0, 110.0],
        "shares_delta": [10, 5], "slippage_cost": [1.0, 1.0], "regime_label": ["BEAR", "BULL"], "regime_confidence": [0.9, 0.8],
    }).to_csv(dir_ / "trade_log.csv", index=False)
    pd.DataFrame({
        "date": dates, "symbol": symbol, "regime_id": 0, "regime_label": ["BEAR"] * 30 + ["BULL"] * 30,
        "confidence": 0.9, "is_confirmed": True, "is_flickering": False, "consecutive_bars": 5, "target_allocation": 0.6,
    }).to_csv(dir_ / "regime_history.csv", index=False)


def test_backtest_summary_reads_saved_results(tmp_path):
    _write_backtest(tmp_path)
    b = BacktestReader(tmp_path).get_backtest_summary("spy")
    assert b.available and b.period_start.isoformat() == "2024-01-02"
    assert b.total_return == pytest.approx(0.059, abs=1e-3)
    assert b.max_drawdown is not None and "by_regime" in b.regime_metrics
    assert b.model_dump_json()  # JSON-safe (no NaN/inf)


def test_missing_backtest_returns_unavailable_not_error(tmp_path):
    assert BacktestReader(tmp_path).get_backtest_summary("SPY").available is False
    _write_backtest(tmp_path, "SPY")
    assert BacktestReader(tmp_path).get_backtest_summary("NVDA").available is False


# ---- registry ---------------------------------------------------------------

def _registry(max_calls=5):
    return ResearchToolRegistry(StubHMMReader(), StubBacktestReader(), max_calls=max_calls, request_id="t")


def test_registry_runs_allowlisted_tools_and_records_trace():
    reg = _registry()
    assert reg.call("get_hmm_snapshot", symbol="SPY").symbol == "SPY"
    assert [c.tool for c in reg.calls] == ["get_hmm_snapshot"] and reg.calls[0].ok
    assert reg.tool_names == ALLOWED_RESEARCH_TOOLS


@pytest.mark.parametrize("name", ["submit_order", "__import__", "get_hmm_snapshot ", "", "set_risk_limit", "call"])
def test_registry_rejects_unknown_tools(name):
    reg = _registry()
    with pytest.raises(ToolRejectedError):
        reg.call(name)
    assert reg.calls == []


def test_registry_enforces_budget_and_wraps_failures():
    reg = _registry(max_calls=1)
    reg.call("get_hmm_snapshot", symbol="SPY")
    with pytest.raises(ToolBudgetExceededError):
        reg.call("get_backtest_summary", symbol="SPY")

    failing = ResearchToolRegistry(StubHMMReader(), StubBacktestReader(fail=True), max_calls=3)
    with pytest.raises(ToolError):
        failing.call("get_backtest_summary", symbol="SPY")
    assert failing.calls[0].ok is False
