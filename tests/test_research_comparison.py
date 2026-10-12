"""Quant-vs-research comparison: deterministic guardrails around the model draft."""

import pytest

from research.agent import Deadline, GeminiAgent
from research.comparison import apply_guardrails, compare_quant_and_research
from research.schemas import ComparisonDraft, ResearchBrief
from datetime import datetime
from research_fakes import FakeGeminiClient, happy_script, load_json, make_config, make_quant


def _brief(stance="mixed", conflicts=None, evidence=True):
    from research.schemas import Evidence
    ev = [Evidence(claim="c", category="other", impact="mixed", source_title="t", source_url="https://a.com/x",
                   retrieved_at=datetime(2026, 10, 10), confidence=0.5)] if evidence else []
    return ResearchBrief(symbol="GOOGL", generated_at=datetime(2026, 10, 10), question="q", summary="s", catalysts=ev,
                         conflicting_evidence=conflicts or [], evidence_stance=stance, overall_confidence=0.5, source_count=3)


def _draft(**kw):
    return ComparisonDraft.model_validate({**load_json("comparison_draft.json"), **kw})


def test_orthogonal_comparison_passes_through_with_python_quant_summary():
    c = apply_guardrails(_draft(), make_quant(), _brief())
    assert c.relationship == "orthogonal" and c.human_review_recommended is False
    assert "NEUTRAL" in c.quant_summary and "82%" in c.quant_summary  # numbers come from Python, not the model


def test_insufficient_evidence_forces_insufficient_relationship_and_review():
    c = apply_guardrails(_draft(relationship="supportive"), make_quant(), _brief(stance="insufficient_evidence"))
    assert c.relationship == "insufficient_evidence" and c.human_review_recommended


@pytest.mark.parametrize("quant_kw,conflicts", [
    ({"is_flickering": True}, []), ({"is_confirmed": False}, []), ({}, ["sources disagree"]),
])
def test_review_forced_on_instability_or_conflict(quant_kw, conflicts):
    c = apply_guardrails(_draft(), make_quant(**quant_kw), _brief(conflicts=conflicts))
    assert c.human_review_recommended is True


def test_no_quant_context_is_never_compared_by_model():
    client = FakeGeminiClient()
    agent = GeminiAgent(make_config(), client=client)
    c = compare_quant_and_research(agent, make_quant(regime_id=None, regime_label=None), _brief(), None, None, Deadline(60))
    assert c.relationship == "insufficient_evidence" and client.models.calls == []


def test_empty_evidence_skips_the_model_call():
    client = FakeGeminiClient()
    agent = GeminiAgent(make_config(), client=client)
    c = compare_quant_and_research(agent, make_quant(), _brief(stance="insufficient_evidence", evidence=False), None, None, Deadline(60))
    assert c.relationship == "insufficient_evidence" and client.models.calls == []


def test_prompt_never_asks_for_trade_instructions():
    client = FakeGeminiClient(happy_script())
    agent = GeminiAgent(make_config(), client=client)
    compare_quant_and_research(agent, make_quant(), _brief(), None, None, Deadline(60))
    assert "NOT a trading recommendation" in client.models.calls[0]["contents"]


def test_overridden_relationship_is_explained_not_contradicted():
    c = apply_guardrails(_draft(relationship="partially_supportive", explanation="Both show uncertainty."),
                         make_quant(), _brief(stance="insufficient_evidence"))
    assert c.relationship == "insufficient_evidence"
    assert "deterministic check" in c.explanation and "partially_supportive" in c.explanation
