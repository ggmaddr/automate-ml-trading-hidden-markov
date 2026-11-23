"""Result cache: keys, TTL, force refresh, invalidation, no caching of failures."""

import copy
from datetime import datetime, timedelta, timezone

import pytest

from research.agent import GeminiAgent
from research.cache import ResearchCache, make_cache_key, quant_fingerprint
from research.orchestrator import ResearchOrchestrator
from research.schemas import ResearchRequest
from research_fakes import (
    FakeGeminiClient, StubBacktestReader, StubHMMReader, api_error, fake_resolver, happy_script, make_config, make_quant,
)


class Clock:
    def __init__(self):
        self.now = datetime(2026, 10, 10, 12, tzinfo=timezone.utc)

    def __call__(self):
        return self.now


def _setup(tmp_path, script=None, config=None, clock=None):
    config = config or make_config(cache_ttl_minutes=60)
    clock = clock or Clock()
    client = FakeGeminiClient(script or happy_script())
    orch = ResearchOrchestrator(
        config, {"backtest": {}}, agent=GeminiAgent(config, client=client, sleep=lambda s: None),
        cache=ResearchCache(tmp_path / "c.db", config.cache_ttl_minutes, clock=clock),
        hmm_reader=StubHMMReader(), backtest_reader=StubBacktestReader(), redirect_resolver=fake_resolver, clock=clock,
    )
    return orch, client, clock


def test_identical_fresh_request_is_a_cache_hit(tmp_path):
    orch, client, _ = _setup(tmp_path)
    first = orch.research(ResearchRequest(symbol="GOOGL"))
    n_calls = len(client.models.calls)
    second = orch.research(ResearchRequest(symbol="GOOGL"))
    assert first.result.cache_hit is False and second.result.cache_hit is True
    assert len(client.models.calls) == n_calls          # no new Gemini traffic
    assert second.result.brief.generated_at == first.result.brief.generated_at   # original timestamp kept


def test_force_refresh_bypasses_cache(tmp_path):
    orch, client, _ = _setup(tmp_path)
    orch.research(ResearchRequest(symbol="GOOGL"))
    n = len(client.models.calls)
    out = orch.research(ResearchRequest(symbol="GOOGL", force_refresh=True))
    assert out.result.cache_hit is False and len(client.models.calls) > n


def test_expired_entry_refreshes(tmp_path):
    orch, client, clock = _setup(tmp_path)
    orch.research(ResearchRequest(symbol="GOOGL"))
    n = len(client.models.calls)
    clock.now += timedelta(minutes=61)
    out = orch.research(ResearchRequest(symbol="GOOGL"))
    assert out.result.cache_hit is False and len(client.models.calls) > n


def test_key_changes_with_question_model_prompt_lookback_and_quant():
    req = ResearchRequest(symbol="GOOGL", question="What changed?")
    cfg = make_config()
    fp = quant_fingerprint(make_quant())
    base = make_cache_key(req, cfg, fp)
    assert make_cache_key(ResearchRequest(symbol="GOOGL", question="  what   CHANGED? "), cfg, fp) == base  # normalized
    assert make_cache_key(ResearchRequest(symbol="GOOGL", question="Other"), cfg, fp) != base
    assert make_cache_key(ResearchRequest(symbol="GOOGL", question="What changed?", lookback_days=7), cfg, fp) != base
    assert make_cache_key(req, make_config(model="other-model"), fp) != base
    assert make_cache_key(req, cfg, quant_fingerprint(make_quant(regime_id=2))) != base
    from research import cache as cache_mod
    original = cache_mod.RESEARCH_PROMPT_VERSION
    try:
        cache_mod.RESEARCH_PROMPT_VERSION = "9.9"
        assert make_cache_key(req, cfg, fp) != base
    finally:
        cache_mod.RESEARCH_PROMPT_VERSION = original


def test_changed_model_does_not_reuse_entry(tmp_path):
    orch, _, clock = _setup(tmp_path)
    orch.research(ResearchRequest(symbol="GOOGL"))
    orch2, client2, _ = _setup(tmp_path, config=make_config(model="newer-model", cache_ttl_minutes=60), clock=clock)
    assert orch2.research(ResearchRequest(symbol="GOOGL")).result.cache_hit is False


def test_failed_responses_are_never_cached(tmp_path):
    script = happy_script()
    script["search"] = [api_error(503)]
    orch, client, _ = _setup(tmp_path, script=script)
    assert orch.research(ResearchRequest(symbol="GOOGL")).ok is False
    # next request, with Gemini healthy again, must do real work
    client.models.script = {k: list(v) for k, v in happy_script().items()}
    out = orch.research(ResearchRequest(symbol="GOOGL"))
    assert out.ok and out.result.cache_hit is False


def test_corrupt_cache_entry_is_treated_as_miss(tmp_path):
    cache = ResearchCache(tmp_path / "c.db", 60)
    import sqlite3
    cache.stats()  # initializes schema
    with sqlite3.connect(tmp_path / "c.db") as conn:
        conn.execute("INSERT INTO results VALUES ('k','A','m','1','2026-10-10T00:00:00+00:00','2999-01-01T00:00:00+00:00','{bad json')")
    assert cache.get("k") is None


def test_usage_stats(tmp_path):
    orch, _, _ = _setup(tmp_path)
    orch.research(ResearchRequest(symbol="GOOGL"))
    orch.research(ResearchRequest(symbol="GOOGL"))
    s = orch.cache.stats()
    assert (s.requests, s.cache_hits, s.cache_misses) == (2, 1, 1)
    assert s.hit_rate == 0.5 and s.last_outcome == "success" and s.last_success_at is not None
