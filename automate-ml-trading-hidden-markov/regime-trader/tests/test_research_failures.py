"""Failure handling: every Gemini/grounding/schema failure is a typed, non-fatal outcome."""

import json

import pytest

from research.agent import Deadline, GeminiAgent
from research.cache import ResearchCache
from research.exceptions import SchemaValidationError
from research.orchestrator import ResearchOrchestrator
from research.schemas import BriefDraft, ResearchRequest
from research_fakes import (
    FakeGeminiClient, FakeModels, StubBacktestReader, StubHMMReader, api_error, fake_resolver, grounded_response,
    happy_script, load_json, make_config, ungrounded_response,
)

sleeps: list[float] = []


def build(tmp_path, script, hmm=None, backtest=None, agent=True, **cfg):
    config = make_config(**cfg)
    client = FakeGeminiClient(script)
    orch = ResearchOrchestrator(
        config, {"backtest": {}}, agent=GeminiAgent(config, client=client, sleep=sleeps.append) if agent else None,
        agent_factory=None if agent else (lambda c: GeminiAgent(c, api_key=None, client=None)),
        cache=ResearchCache(tmp_path / "c.db", 60), hmm_reader=hmm or StubHMMReader(),
        backtest_reader=backtest or StubBacktestReader(), redirect_resolver=fake_resolver,
    )
    return orch, client


def failed(outcome, code):
    assert outcome.ok is False and outcome.error.code == code
    assert outcome.quant_context is not None or code in ("disabled", "quant_context_unavailable", "not_configured")
    return outcome


def test_quota_error_is_typed_and_retried(tmp_path):
    script = happy_script()
    script["search"] = [api_error(429, "quota")]
    sleeps.clear()
    orch, client = build(tmp_path, script)
    out = failed(orch.research(ResearchRequest(symbol="GOOGL")), "quota_exceeded")
    assert out.error.retryable and "Quantitative system continues normally" in out.error.message
    assert out.quant_context.symbol == "GOOGL"            # quant snapshot survives for the UI
    assert client.models.kinds().count("search") == 3 and len(sleeps) == 2   # 1 + max_retries


def test_retry_succeeds_after_transient_error(tmp_path):
    script = happy_script()
    script["search"] = [api_error(503), grounded_response()]
    orch, _ = build(tmp_path, script)
    assert orch.research(ResearchRequest(symbol="GOOGL")).ok


@pytest.mark.parametrize("code,expected", [(401, "auth_failed"), (403, "auth_failed"), (404, "bad_request"), (400, "bad_request")])
def test_non_retryable_api_errors_fail_fast(tmp_path, code, expected):
    script = happy_script()
    script["search"] = [api_error(code, "x")]
    orch, client = build(tmp_path, script)
    failed(orch.research(ResearchRequest(symbol="GOOGL")), expected)
    assert client.models.kinds().count("search") == 1


def test_timeout_is_typed(tmp_path):
    script = happy_script()
    script["search"] = [TimeoutError("read timed out")]
    orch, _ = build(tmp_path, script)
    failed(orch.research(ResearchRequest(symbol="GOOGL")), "timeout")


def test_overall_deadline_stops_retries():
    config = make_config()
    t = [0.0]
    deadline = Deadline(10, clock=lambda: t[0])
    client = FakeGeminiClient({"search": [api_error(503)], "brief": [{}], "comparison": [{}]})
    agent = GeminiAgent(config, client=client, sleep=lambda s: t.__setitem__(0, t[0] + 20))
    from research.exceptions import ResearchError
    with pytest.raises(ResearchError):
        agent.grounded_search("p", system="s", deadline=deadline)
    assert len(client.models.calls) == 1


def test_grounding_unavailable_returns_insufficient_evidence_not_ungrounded_claims(tmp_path):
    script = happy_script()
    script["search"] = [ungrounded_response("GOOGL will surge 40% next week!")]
    orch, client = build(tmp_path, script)
    out = orch.research(ResearchRequest(symbol="GOOGL"))
    assert out.ok
    r = out.result
    assert client.models.kinds().count("search") == 2          # retried via the second round
    assert "brief" not in client.models.kinds()                # nothing to synthesize from
    assert r.brief.evidence_stance == "insufficient_evidence" and r.brief.source_count == 0
    assert "surge" not in r.brief.summary and not (r.brief.catalysts or r.brief.risks or r.brief.recent_events)
    assert r.diagnostics.grounding_available is False


def test_backtest_tool_failure_is_not_fatal(tmp_path):
    orch, _ = build(tmp_path, happy_script(), backtest=StubBacktestReader(fail=True))
    out = orch.research(ResearchRequest(symbol="GOOGL"))
    assert out.ok and out.result.backtest_context is None
    assert out.result.diagnostics.tool_calls[-1].ok is False


# ---- structured output -------------------------------------------------------

def test_malformed_json_then_retry_succeeds(tmp_path):
    script = happy_script()
    script["brief"] = ["this is not json {", load_json("brief_draft.json")]
    orch, client = build(tmp_path, script)
    assert orch.research(ResearchRequest(symbol="GOOGL")).ok
    assert client.models.kinds().count("brief") == 2
    assert "rejected" in client.models.calls[2]["contents"]     # repair note sent on retry


def test_pydantic_validation_failure_retries_then_exhausts(tmp_path):
    bad = {**load_json("brief_draft.json"), "overall_confidence": 7}
    script = happy_script()
    script["brief"] = [bad]
    orch, client = build(tmp_path, script)
    out = failed(orch.research(ResearchRequest(symbol="GOOGL")), "schema_validation_failed")
    assert client.models.kinds().count("brief") == 3            # 1 + max_retries
    assert out.result is None                                   # invalid data never rendered as success


def test_fenced_json_is_accepted_but_invalid_stance_is_not_coerced():
    config = make_config(max_retries=0)
    fenced = FakeGeminiClient({"search": [{}], "brief": ["```json\n" + json.dumps(load_json("brief_draft.json")) + "\n```"], "comparison": [{}]})
    assert GeminiAgent(config, client=fenced).generate_structured(
        lambda r: "p", BriefDraft, system="s", deadline=Deadline(60), stage="synthesis").evidence_stance == "mixed"
    bad = FakeGeminiClient({"search": [{}], "brief": [{**load_json("brief_draft.json"), "evidence_stance": "bullish"}], "comparison": [{}]})
    with pytest.raises(SchemaValidationError):
        GeminiAgent(config, client=bad).generate_structured(
            lambda r: "p", BriefDraft, system="s", deadline=Deadline(60), stage="synthesis")


# ---- config / secrets --------------------------------------------------------

def test_disabled_research_is_a_clean_state(tmp_path):
    orch, client = build(tmp_path, happy_script(), enabled=False)
    failed(orch.research(ResearchRequest(symbol="GOOGL")), "disabled")
    assert client.models.calls == []


def test_missing_api_key_is_a_clean_state(tmp_path, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    config = make_config()
    orch = ResearchOrchestrator(config, {"backtest": {}}, cache=ResearchCache(tmp_path / "c.db", 60),
                                hmm_reader=StubHMMReader(), backtest_reader=StubBacktestReader())
    out = orch.research(ResearchRequest(symbol="GOOGL"))
    failed(out, "not_configured")
    assert out.quant_context is not None


def test_api_key_never_appears_in_error_messages(tmp_path, monkeypatch):
    key = "AIzaSyFAKEFAKEFAKEFAKEFAKEFAKEFAKE12345"
    monkeypatch.setenv("GEMINI_API_KEY", key)
    script = happy_script()
    script["search"] = [api_error(500, f"upstream failure for key {key}")]
    orch, _ = build(tmp_path, script)
    out = orch.research(ResearchRequest(symbol="GOOGL"))
    assert key not in out.model_dump_json()
    from research.logging_util import redact
    assert key not in redact(f"error: {key}")


def test_unexpected_exception_becomes_internal_error(tmp_path):
    class Exploding(StubBacktestReader):
        def get_backtest_summary(self, symbol):
            raise MemoryError("boom")
    orch, _ = build(tmp_path, happy_script(), backtest=Exploding())
    out = orch.research(ResearchRequest(symbol="GOOGL"))
    assert out.ok or out.error.code == "internal_error"          # never propagates


def test_insufficient_stance_from_the_model_also_caps_confidence(tmp_path):
    script = happy_script()
    script["brief"] = [{**load_json("brief_draft.json"), "evidence_stance": "insufficient_evidence", "overall_confidence": 0.85}]
    orch, _ = build(tmp_path, script)
    brief = orch.research(ResearchRequest(symbol="GOOGL")).result.brief
    assert brief.evidence_stance == "insufficient_evidence" and brief.overall_confidence <= 0.3
