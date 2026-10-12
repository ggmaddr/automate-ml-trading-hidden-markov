"""Schema validation for the research layer's boundaries."""

from datetime import datetime

import pytest
from pydantic import ValidationError

from research.config import ResearchConfig, load_research_config
from research.schemas import (
    ComparisonDraft, Evidence, QuantResearchComparison, ResearchBrief, ResearchRequest,
)

NOW = datetime(2026, 10, 10)


def _evidence(**kw):
    base = dict(claim="c", category="earnings", impact="positive", source_title="t",
                source_url="https://example.com/a", retrieved_at=NOW, confidence=0.5)
    base.update(kw)
    return Evidence(**base)


def test_request_normalizes_symbol_and_sanitizes_question():
    r = ResearchRequest(symbol=" googl ", question="  what\x00 happened\n\n  today? ", lookback_days=7)
    assert r.symbol == "GOOGL"
    assert r.question == "what happened today?"


@pytest.mark.parametrize("bad", ["", "   ", "GOOGL; DROP", "A" * 20, "../etc"])
def test_request_rejects_bad_symbols(bad):
    with pytest.raises(ValidationError):
        ResearchRequest(symbol=bad)


def test_request_caps_lookback_and_question_length():
    assert ResearchRequest(symbol="A", lookback_days=10_000).lookback_days == 90
    assert ResearchRequest(symbol="A", lookback_days=-5).lookback_days == 1
    assert len(ResearchRequest(symbol="A", question="x" * 5000).question) == 500
    assert ResearchRequest(symbol="A", question="   ").question is None


def test_valid_evidence_and_brief_roundtrip():
    brief = ResearchBrief(
        symbol="A", generated_at=NOW, question="q", summary="s", catalysts=[_evidence()],
        evidence_stance="mixed", overall_confidence=0.6, source_count=1,
    )
    assert ResearchBrief.model_validate_json(brief.model_dump_json()) == brief


@pytest.mark.parametrize("conf", [-0.1, 1.1])
def test_evidence_rejects_invalid_confidence(conf):
    with pytest.raises(ValidationError):
        _evidence(confidence=conf)


def test_evidence_rejects_invalid_category_and_malformed_url():
    with pytest.raises(ValidationError):
        _evidence(category="rumor")
    for url in ["not a url", "ftp://x.com/a", "javascript:alert(1)", "https://"]:
        with pytest.raises(ValidationError):
            _evidence(source_url=url)


def test_malformed_brief_rejected():
    with pytest.raises(ValidationError):
        ResearchBrief(symbol="A", generated_at=NOW, question="q", summary="s",
                      evidence_stance="bullish", overall_confidence=0.5, source_count=0)
    with pytest.raises(ValidationError):
        ResearchBrief(symbol="A", generated_at=NOW, question="q", summary="s",
                      evidence_stance="mixed", overall_confidence=2.0, source_count=0)


def test_malformed_comparison_rejected():
    with pytest.raises(ValidationError):
        QuantResearchComparison(symbol="A", generated_at=NOW, quant_summary="q", research_summary="r",
                                relationship="bullish", human_review_recommended=True, explanation="e")
    with pytest.raises(ValidationError):
        ComparisonDraft.model_validate_json('{"relationship": "supportive"}')


def test_config_rails_cannot_be_enabled():
    for flag in ("allow_broker_tools", "allow_strategy_mutation", "allow_risk_mutation"):
        with pytest.raises(ValidationError):
            ResearchConfig(**{flag: True})


def test_config_loads_from_settings_with_env_model_override():
    cfg = load_research_config({"research": {"model": "from-yaml", "max_sources": 5}}, env={})
    assert (cfg.model, cfg.max_sources) == ("from-yaml", 5)
    assert load_research_config({"research": {"model": "from-yaml"}}, env={"GEMINI_MODEL": "from-env"}).model == "from-env"
    assert load_research_config({}, env={"GEMINI_MODEL": ""}).enabled is True
