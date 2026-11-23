"""Usage tracking, cost estimates, and the local budget guard."""

from datetime import datetime, timezone

import pytest
from google.genai import types

from research.agent import GeminiAgent
from research.cache import ResearchCache
from research.orchestrator import ResearchOrchestrator
from research.schemas import ResearchRequest
from research.usage import estimate_run_cost_usd, get_usage_summary, search_overage_cost_usd, usage_lines
from research_fakes import (
    FakeGeminiClient, StubBacktestReader, StubHMMReader, fake_resolver, grounded_response, happy_script, make_config,
)


def _with_tokens(response, prompt=10_000, out=2_000):
    response.usage_metadata = types.GenerateContentResponseUsageMetadata(
        prompt_token_count=prompt, candidates_token_count=out, thoughts_token_count=500)
    return response


def _orch(tmp_path, **cfg):
    config = make_config(**cfg)
    script = happy_script()
    script["search"] = [_with_tokens(grounded_response())]
    client = FakeGeminiClient(script)
    return ResearchOrchestrator(
        config, {"backtest": {}}, agent=GeminiAgent(config, client=client, sleep=lambda s: None),
        cache=ResearchCache(tmp_path / "c.db", 60), hmm_reader=StubHMMReader(),
        backtest_reader=StubBacktestReader(), redirect_resolver=fake_resolver,
    ), config


def test_overage_only_charged_past_free_allowance():
    cfg = make_config(free_grounded_searches_per_month=100, grounded_search_price_per_1000_usd=14)
    assert search_overage_cost_usd(cfg, 0, 100) == 0
    assert search_overage_cost_usd(cfg, 90, 20) == pytest.approx(10 / 1000 * 14)
    assert search_overage_cost_usd(cfg, 500, 10) == pytest.approx(10 / 1000 * 14)
    cost = estimate_run_cost_usd(make_config(input_price_per_million_tokens_usd=1, output_price_per_million_tokens_usd=2,
                                             free_grounded_searches_per_month=10**6), 1_000_000, 500_000, 6, 0)
    assert cost == pytest.approx(2.0)


def test_run_records_tokens_and_searches(tmp_path):
    orch, config = _orch(tmp_path)
    out = orch.research(ResearchRequest(symbol="GOOGL"))
    u = out.result.diagnostics.usage
    assert u.grounded_searches == 2                      # the fixture issued 2 search queries
    assert u.input_tokens == 10_000 and u.output_tokens == 2_500   # thinking tokens billed as output
    assert 0 < u.estimated_cost_usd < 0.05               # tokens only: inside the free search allowance
    month = orch.cache.month_usage()
    assert (month.grounded_searches, month.input_tokens, month.gemini_runs) == (2, 10_000, 1)


def test_cache_hit_costs_nothing(tmp_path):
    orch, _ = _orch(tmp_path)
    orch.research(ResearchRequest(symbol="GOOGL"))
    orch.research(ResearchRequest(symbol="GOOGL"))
    month = orch.cache.month_usage()
    assert month.grounded_searches == 2 and month.requests == 2 and month.gemini_runs == 1


def test_failed_run_still_counts_consumed_searches(tmp_path):
    orch, config = _orch(tmp_path)
    orch._llm._client.models.script["brief"] = ["not json"]
    out = orch.research(ResearchRequest(symbol="GOOGL"))
    assert out.error.code == "schema_validation_failed"
    assert orch.cache.month_usage().grounded_searches == 2


def test_budget_guard_blocks_new_runs_but_not_cache(tmp_path):
    orch, _ = _orch(tmp_path, monthly_budget_usd=0.0001)
    first = orch.research(ResearchRequest(symbol="GOOGL"))
    assert first.ok                                     # nothing spent yet
    blocked = orch.research(ResearchRequest(symbol="GOOGL", question="something new"))
    assert blocked.ok is False and blocked.error.code == "budget_exceeded" and not blocked.error.retryable
    assert orch.research(ResearchRequest(symbol="GOOGL")).result.cache_hit is True


def test_budget_zero_disables_guard(tmp_path):
    orch, _ = _orch(tmp_path, monthly_budget_usd=0)
    orch.research(ResearchRequest(symbol="GOOGL"))
    assert orch.research(ResearchRequest(symbol="GOOGL", question="again", force_refresh=True)).ok


def test_summary_and_ui_lines_state_the_limits(tmp_path):
    orch, config = _orch(tmp_path)
    orch.research(ResearchRequest(symbol="GOOGL"))
    s = get_usage_summary(config, orch.cache)
    assert (s.grounded_searches, s.free_remaining) == (2, 4998) and s.est_runs_left_free == 4998 // 2
    text = "\n".join(usage_lines(config, s))
    assert "of 5,000 free" in text and "NOT on Gemini's free tier" in text and "billing" in text
    assert "midnight Pacific" in text and "aistudio.google.com/rate-limit" in text


def test_old_usage_table_is_migrated(tmp_path):
    import sqlite3
    db = tmp_path / "old.db"
    with sqlite3.connect(db) as conn:
        conn.executescript("CREATE TABLE usage (id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL, request_id TEXT,"
                           " symbol TEXT, outcome TEXT NOT NULL, error_code TEXT, cache_hit INTEGER NOT NULL, latency_ms INTEGER,"
                           " search_queries INTEGER, source_count INTEGER, model TEXT);")
    cache = ResearchCache(db, 60)
    assert cache.month_usage().grounded_searches == 0
    cache.record_usage(request_id="r", symbol="A", ok=True, cache_hit=False, latency_ms=1, grounded_searches=3)
    assert cache.month_usage().grounded_searches == 3
