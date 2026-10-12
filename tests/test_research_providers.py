"""Config-only provider switching, end to end through the orchestrator (no network)."""

from research.cache import ResearchCache, make_cache_key, quant_fingerprint
from research.exceptions import SearchUnavailableError
from research.llm.base import UsageCounters
from research.orchestrator import ResearchOrchestrator
from research.schemas import (
    BriefDraft, ComparisonDraft, NoteDraft, NotesDraft, QueryPlanDraft, ResearchRequest,
)
from research.search import SearchHit
from research_fakes import StubBacktestReader, StubHMMReader, load_json, make_config, make_quant

HITS = [
    SearchHit("Reuters", "https://reuters.com/a", "Results beat", "2026-10-01"),
    SearchHit("SEC", "https://sec.gov/f", "10-Q"),
    SearchHit("CNBC", "https://cnbc.com/p", "EU probe"),
]


class ScriptedLLM:
    """A local-model stand-in answering every pipeline stage."""

    provider = "ollama"
    model = "qwen3:8b"

    def __init__(self):
        self.usage, self.call_count, self.stages = UsageCounters(), 0, []

    def generate_structured(self, prompt_for, schema, *, system, deadline, stage, request_id=""):
        self.call_count += 1
        self.usage.input_tokens += 500
        self.usage.output_tokens += 100
        self.stages.append(stage)
        if schema is QueryPlanDraft:
            return QueryPlanDraft(queries=["a", "b"])
        if schema is NotesDraft:
            return NotesDraft(notes=[NoteDraft(text="Results beat expectations.", source_numbers=[1]),
                                     NoteDraft(text="10-Q notes supply limits.", source_numbers=[2]),
                                     NoteDraft(text="EU opened a probe.", source_numbers=[3])])
        if schema is BriefDraft:
            return BriefDraft.model_validate(load_json("brief_draft.json"))
        if schema is ComparisonDraft:
            return ComparisonDraft.model_validate(load_json("comparison_draft.json"))
        raise AssertionError(schema)

    def list_models(self):
        return []


class Search:
    name = "searxng"

    def __init__(self, fail=False):
        self.fail = fail

    def search(self, query, *, max_results, lookback_days, deadline):
        if self.fail:
            raise SearchUnavailableError("cannot connect to SearXNG at http://localhost:8080")
        return list(HITS)

    def check(self):
        return "ok"


class NoFetch:
    def fetch(self, url):
        return None


def orch(tmp_path, llm=None, search=None, **cfg):
    config = make_config(provider="ollama", model="qwen3:8b", **cfg)
    return ResearchOrchestrator(
        config, {"backtest": {}}, llm=llm or ScriptedLLM(), search_provider=search or Search(), page_fetcher=NoFetch(),
        cache=ResearchCache(tmp_path / "c.db", 60), hmm_reader=StubHMMReader(), backtest_reader=StubBacktestReader(),
    )


def test_local_provider_full_pipeline_with_real_search_provenance(tmp_path):
    llm = ScriptedLLM()
    out = orch(tmp_path, llm).research(ResearchRequest(symbol="NVDA"))
    assert out.ok
    r = out.result
    assert (r.diagnostics.provider, r.diagnostics.search_provider, r.diagnostics.model) == ("ollama", "searxng", "qwen3:8b")
    assert llm.stages == ["query_plan", "research_notes", "synthesis", "comparison"]
    urls = {e.source_url for e in r.brief.catalysts + r.brief.risks + r.brief.recent_events}
    assert urls <= {h.url for h in HITS} and urls                 # every citation is a real search result
    assert r.brief.catalysts[0].published_at is not None          # model date or real SearXNG metadata
    assert r.brief.source_count == 3
    assert r.diagnostics.usage.estimated_cost_usd == 0 and r.diagnostics.usage.input_tokens == 2000


def test_local_runs_are_free_and_never_hit_the_budget_guard(tmp_path):
    o = orch(tmp_path, monthly_budget_usd=0.0001)
    assert o.research(ResearchRequest(symbol="NVDA")).ok
    assert o.research(ResearchRequest(symbol="NVDA", force_refresh=True)).ok
    from research.usage import get_usage_summary, usage_lines
    summary = get_usage_summary(o.config, o.cache)
    assert summary.over_budget is False and summary.est_spend_usd == 0
    assert "no per-request cost" in usage_lines(o.config, summary)[0]


def test_cache_is_separate_per_provider_and_search():
    req = ResearchRequest(symbol="NVDA")
    fp = quant_fingerprint(make_quant())
    keys = {
        make_cache_key(req, make_config(provider="ollama", model="m"), fp),
        make_cache_key(req, make_config(provider="gemini", model="m"), fp),
        make_cache_key(req, make_config(provider="ollama", model="m", search_provider="gemini_grounding"), fp),
    }
    assert len(keys) == 3


def test_search_outage_is_a_typed_outcome_and_keeps_quant(tmp_path):
    out = orch(tmp_path, search=Search(fail=True)).research(ResearchRequest(symbol="NVDA"))
    assert out.ok is False and out.error.code == "search_unavailable" and out.error.retryable
    assert "docker compose" in out.error.message and out.quant_context.symbol == "NVDA"


def test_llm_outage_is_a_typed_outcome(tmp_path):
    from research.exceptions import LLMUnavailableError

    class Down(ScriptedLLM):
        def generate_structured(self, *a, **k):
            raise LLMUnavailableError("cannot connect to Ollama at http://localhost:11434")

    out = orch(tmp_path, Down()).research(ResearchRequest(symbol="NVDA"))
    assert out.error.code == "llm_unavailable" and "ollama serve" in out.error.message


def test_gemini_grounding_requires_gemini_provider(tmp_path):
    o = ResearchOrchestrator(make_config(provider="ollama", model="m", search_provider="gemini_grounding"), {"backtest": {}},
                             llm=ScriptedLLM(), cache=ResearchCache(tmp_path / "c.db", 60),
                             hmm_reader=StubHMMReader(), backtest_reader=StubBacktestReader())
    out = o.research(ResearchRequest(symbol="NVDA"))
    assert out.error.code == "not_configured" and "needs provider 'gemini'" in out.error.message
