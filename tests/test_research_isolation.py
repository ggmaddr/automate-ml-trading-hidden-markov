"""Main-system isolation: a Gemini outage must not affect Phases 1-9."""

import numpy as np
import pytest

from core.hmm_engine import HMMRegimeEngine
from core.regime_strategies import StrategyOrchestrator
from core.risk_manager import RiskManager
from data.feature_engineering import compute_features
from research.agent import GeminiAgent
from research.cache import ResearchCache
from research.orchestrator import ResearchOrchestrator
from research.presentation import age_label, evidence_markdown, quant_rows
from research.schemas import ResearchRequest
from research.tools.hmm_tools import HMMRuntimeReader
from research_fakes import FakeGeminiClient, StubBacktestReader, api_error, happy_script, make_config


def test_gemini_outage_leaves_hmm_and_risk_manager_working(tmp_path, trained_hmm, synthetic_ohlcv, backtest_config):
    model_path = tmp_path / "m.pkl"
    trained_hmm.save(str(model_path))

    script = happy_script()
    script["search"] = [api_error(503, "Gemini is down")]
    config = make_config(max_retries=1)
    orch = ResearchOrchestrator(
        config, {"backtest": {}},
        agent=GeminiAgent(config, client=FakeGeminiClient(script), sleep=lambda s: None),
        cache=ResearchCache(tmp_path / "c.db", 60),
        hmm_reader=HMMRuntimeReader(model_path, bars_provider=lambda s: synthetic_ohlcv),   # REAL HMM, real snapshot
        backtest_reader=StubBacktestReader(),
    )

    # research page reports unavailable (typed state, no exception) but keeps the quant snapshot
    outcome = orch.research(ResearchRequest(symbol="SPY"))
    assert outcome.ok is False and outcome.error.code == "unavailable"
    assert outcome.quant_context is not None and outcome.quant_context.regime_label

    # HMM still predicts
    features = compute_features(synthetic_ohlcv)
    engine = HMMRegimeEngine.load(str(model_path))
    state = engine.update(features)
    assert state.label in engine.regime_labels.values()

    # strategy + risk manager still operate
    strategies = StrategyOrchestrator(backtest_config["strategy"], engine.regime_info)
    assert strategies is not None
    assert RiskManager(backtest_config["risk"]) is not None


def test_research_disabled_changes_nothing_for_the_rest_of_the_app(tmp_path):
    from research.orchestrator import ResearchOrchestrator

    orch = ResearchOrchestrator(make_config(enabled=False), {"backtest": {}}, cache=ResearchCache(tmp_path / "c.db", 60))
    assert orch.research(ResearchRequest(symbol="SPY")).error.code == "disabled"


def test_streamlit_research_view_renders_without_a_key(monkeypatch):
    """The AI Research page degrades to a message instead of crashing the dashboard."""
    from streamlit.testing.v1 import AppTest

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    at = AppTest.from_file("dashboard_app.py", default_timeout=60)
    at.session_state["auto_refresh"] = False
    at.run()
    at.switch_page("views/research.py").run()
    assert not at.exception


def test_presentation_helpers():
    from datetime import datetime, timezone
    from research.schemas import Evidence

    e = Evidence(claim="Beat estimates", category="earnings", impact="positive", source_title="reuters.com",
                 source_url="https://reuters.com/a", published_at=datetime(2026, 10, 1), retrieved_at=datetime(2026, 10, 2), confidence=0.8)
    md = evidence_markdown(e)
    assert "[reuters.com](https://reuters.com/a)" in md and "2026-10-01" in md
    assert age_label(datetime(2026, 10, 10, 12, tzinfo=timezone.utc), now=datetime(2026, 10, 10, 18, tzinfo=timezone.utc)) == "6 h ago"
    from research_fakes import make_quant
    assert dict(quant_rows(make_quant()))["Confidence"] == "82%"
