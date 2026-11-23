"""Happy-path and degraded-path orchestration with mocked Gemini."""

import pytest

from research.agent import GeminiAgent
from research.cache import ResearchCache
from research.orchestrator import ResearchOrchestrator
from research.prompts import RESEARCH_PROMPT_VERSION
from research.schemas import ResearchRequest, ResearchResult
from research_fakes import (
    FakeGeminiClient, StubBacktestReader, StubHMMReader, fake_resolver, grounded_response, happy_script,
    load_json, make_config, make_quant, ungrounded_response,
)


def build(tmp_path, script=None, hmm=None, backtest=None, **cfg):
    config = make_config(**cfg)
    client = FakeGeminiClient(script or happy_script())
    hmm = hmm or StubHMMReader()
    backtest = backtest or StubBacktestReader()
    orch = ResearchOrchestrator(
        config, {"backtest": {}}, agent=GeminiAgent(config, client=client, sleep=lambda s: None),
        cache=ResearchCache(tmp_path / "c.db", 60), hmm_reader=hmm, backtest_reader=backtest,
        redirect_resolver=fake_resolver,
    )
    return orch, client, hmm, backtest


def test_normal_success_flow(tmp_path):
    orch, client, hmm, _ = build(tmp_path)
    out = orch.research(ResearchRequest(symbol="googl", question="What changed after earnings?"))
    assert out.ok and out.error is None
    r = out.result
    ResearchResult.model_validate_json(r.model_dump_json())  # fully valid, serializable
    assert client.models.kinds() == ["search", "brief", "comparison"]
    # internal context -> grounded research -> structured brief -> comparison
    assert r.quant_context.symbol == "GOOGL" and r.backtest_context.available
    assert r.brief.source_count == 3 and r.brief.evidence_stance == "mixed"
    assert r.brief.search_queries == ["GOOGL latest earnings results", "Alphabet regulatory probe EU ad tech"]
    assert r.comparison.relationship == "orthogonal"
    assert r.diagnostics.model == "test-model" and r.diagnostics.prompt_version == RESEARCH_PROMPT_VERSION
    assert [t.tool for t in r.diagnostics.tool_calls] == [
        "get_hmm_snapshot", "get_regime_history", "get_backtest_summary"]  # question mentions "changed"
    assert r.cache_hit is False and r.latency_ms >= 0


def test_evidence_provenance_comes_from_grounding_not_the_model(tmp_path):
    orch, *_ = build(tmp_path)
    brief = orch.research(ResearchRequest(symbol="GOOGL")).result.brief
    grounded_urls = {"https://www.reuters.com/technology/alphabet-results-2026-10-01/?utm_source=x",
                     "https://www.sec.gov/Archives/edgar/data/1652044/10q.htm",
                     "https://www.cnbc.com/2026/10/02/alphabet-probe.html"}
    evidence = brief.catalysts + brief.risks + brief.recent_events
    assert evidence and {e.source_url for e in evidence} <= grounded_urls
    # redirect resolved to the publisher URL; unparseable model date dropped, not guessed
    risk = brief.risks[0]
    assert risk.source_domain == "cnbc.com" and risk.published_at is None


def test_unsupported_claims_are_not_converted_to_evidence(tmp_path):
    orch, *_ = build(tmp_path)
    r = orch.research(ResearchRequest(symbol="GOOGL")).result
    assert r.brief.recent_events == []                       # S99 does not exist in grounding
    assert any("Unverified (no grounded source)" in u for u in r.brief.unknowns)
    assert r.diagnostics.dropped_unsupported_claims == 1


def test_conflicting_evidence_is_preserved(tmp_path):
    orch, *_ = build(tmp_path)
    r = orch.research(ResearchRequest(symbol="GOOGL")).result
    assert r.brief.conflicting_evidence == ["Analysts disagree on whether near-term capex will pressure margins."]
    assert r.comparison.human_review_recommended is True


def test_regime_history_collected_for_trend_questions_or_instability(tmp_path):
    orch, _, hmm, _ = build(tmp_path)
    orch.research(ResearchRequest(symbol="GOOGL"))
    assert "get_regime_history" not in hmm.calls
    orch2, _, hmm2, _ = build(tmp_path, hmm=StubHMMReader(make_quant(is_flickering=True)))
    orch2.research(ResearchRequest(symbol="GOOGL", force_refresh=True))
    assert "get_regime_history" in hmm2.calls
    orch3, _, hmm3, _ = build(tmp_path, hmm=StubHMMReader())
    orch3.research(ResearchRequest(symbol="GOOGL", question="Did the volatility regime change?", force_refresh=True))
    assert "get_regime_history" in hmm3.calls


def test_missing_optional_backtest_context(tmp_path):
    orch, *_ = build(tmp_path, backtest=StubBacktestReader(available=False))
    r = orch.research(ResearchRequest(symbol="GOOGL")).result
    assert r.backtest_context.available is False


def test_tool_budget_is_enforced(tmp_path):
    orch, _, _, backtest = build(tmp_path, max_internal_tool_calls=1)
    out = orch.research(ResearchRequest(symbol="GOOGL"))
    assert out.ok and backtest.calls == []                   # snapshot used the whole budget


def test_second_search_round_when_sources_are_thin(tmp_path):
    thin = grounded_response()
    meta = thin.candidates[0].grounding_metadata
    meta.grounding_chunks = meta.grounding_chunks[:1]
    meta.grounding_supports = []
    script = happy_script()
    script["search"] = [thin, grounded_response()]
    orch, client, *_ = build(tmp_path, script=script)
    r = orch.research(ResearchRequest(symbol="GOOGL")).result
    assert client.models.kinds().count("search") == 2 and r.diagnostics.search_rounds == 2
    assert r.brief.source_count == 3


def test_fewer_than_min_sources_yields_insufficient_evidence_not_failure(tmp_path):
    only_one = grounded_response()
    only_one.candidates[0].grounding_metadata.grounding_chunks = only_one.candidates[0].grounding_metadata.grounding_chunks[:1]
    only_one.candidates[0].grounding_metadata.grounding_supports = []
    script = happy_script()
    script["search"] = [only_one]
    script["brief"] = [{**load_json("brief_draft.json"), "catalysts": [load_json("brief_draft.json")["catalysts"][0]],
                        "risks": [], "recent_events": [], "overall_confidence": 0.9}]
    orch, *_ = build(tmp_path, script=script)
    r = orch.research(ResearchRequest(symbol="GOOGL")).result
    assert r.brief.evidence_stance == "insufficient_evidence" and r.brief.overall_confidence <= 0.3
    assert r.comparison.relationship == "insufficient_evidence"


def test_standalone_research_only_when_explicitly_requested(tmp_path):
    orch, *_ = build(tmp_path, hmm=StubHMMReader(fail=True))
    out = orch.research(ResearchRequest(symbol="GOOGL"))
    assert out.ok is False and out.error.code == "quant_context_unavailable"

    orch2, client, *_ = build(tmp_path, hmm=StubHMMReader(fail=True))
    out2 = orch2.research(ResearchRequest(symbol="GOOGL", allow_without_quant=True))
    assert out2.ok and out2.result.quant_context.regime_id is None
    assert out2.result.comparison.relationship == "insufficient_evidence"
    assert "comparison" not in client.models.kinds()


def test_prompts_carry_no_secrets_or_trade_instructions(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "AIzaSyFAKEFAKEFAKEFAKEFAKEFAKEFAKE12345")
    orch, client, *_ = build(tmp_path)
    orch.research(ResearchRequest(symbol="GOOGL"))
    for call in client.models.calls:
        assert "AIzaSy" not in call["contents"]
        assert "Do not output BUY, SELL, or HOLD" in call["config"].system_instruction
