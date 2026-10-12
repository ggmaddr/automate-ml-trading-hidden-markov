"""SerpApi Google Finance news as an extra, real-link evidence source (mocked HTTP)."""

import json
from datetime import date, datetime, timezone

import httpx
import pytest

from research.exceptions import ResearchAuthError, ResearchNotConfiguredError, ResearchQuotaError
from research.llm.base import Deadline, UsageCounters
from research.schemas import NoteDraft, NotesDraft, QueryPlanDraft
from research.search import SearchHit
from research.search.serpapi_finance import SerpApiFinanceNews, parse_news_date
from research.search_researcher import ToolSearchResearcher
from research_fakes import make_config

NOW = datetime(2026, 10, 11, 12, tzinfo=timezone.utc)
NEWS = {"news_results": [
    {"title": "Nvidia eyes Reflection AI", "snippet": "FT reports talks.", "link": "https://www.ft.com/x", "source": "FT", "date": "3 days ago"},
    {"snippet": "Chip export rules.", "link": "https://reuters.com/y", "source": "Reuters", "date": "Oct 5, 2026"},
    {"title": "bad link", "link": "javascript:alert(1)"},
]}


def serp(handler, **cfg):
    config = make_config(finance_news_provider="serpapi", **cfg)
    return SerpApiFinanceNews(config, client=httpx.Client(transport=httpx.MockTransport(handler)), api_key="serp-test-key")


def test_date_parsing_never_guesses():
    assert parse_news_date("3 days ago", NOW) == "2026-10-08"
    assert parse_news_date("2 hours ago", NOW) == "2026-10-11"
    assert parse_news_date("1 week ago", NOW) == "2026-10-04"
    assert parse_news_date("Oct 5, 2026") == "2026-10-05"
    assert parse_news_date("sometime recently") is None and parse_news_date(None) is None


def test_news_hits_parsed_with_real_links_and_dates():
    seen = []

    def handler(request):
        seen.append(dict(request.url.params))
        return httpx.Response(200, json=NEWS)

    client = serp(handler)
    hits = client.news("NVDA", max_results=5, deadline=Deadline(30), now=NOW)
    assert [h.url for h in hits] == ["https://www.ft.com/x", "https://reuters.com/y"]
    assert hits[0].published == "2026-10-08" and hits[1].published == "2026-10-05"
    assert hits[1].title.startswith("Reuters:")                  # untitled item: built from source + snippet
    assert seen[0]["engine"] == "google_finance" and seen[0]["q"] == "NVDA:NASDAQ"
    assert client.searches == 1


def test_grouped_news_format_and_exchange_fallback():
    def handler(request):
        if request.url.params["q"] == "NVDA:NASDAQ":
            return httpx.Response(200, json={"error": "Google Finance hasn't returned any results for this query."})
        return httpx.Response(200, json={"news_results": [{"items": [NEWS["news_results"][0]]}]})

    client = serp(handler)
    hits = client.news("NVDA", max_results=5, deadline=Deadline(30), now=NOW)
    assert len(hits) == 1 and client.searches == 1               # the errored NASDAQ attempt was not billable


@pytest.mark.parametrize("status,body,exc", [(401, {}, ResearchAuthError), (200, {"error": "Invalid API key."}, ResearchAuthError),
                                             (429, {}, ResearchQuotaError),
                                             (200, {"error": "Your account has run out of searches."}, ResearchQuotaError)])
def test_key_and_quota_errors_are_typed(status, body, exc):
    with pytest.raises(exc):
        serp(lambda r: httpx.Response(status, json=body)).news("NVDA", max_results=3, deadline=Deadline(30))


def test_missing_key_is_a_clear_config_error(monkeypatch):
    monkeypatch.delenv("SERPAPI_API_KEY", raising=False)
    with pytest.raises(ResearchNotConfiguredError) as exc:
        SerpApiFinanceNews(make_config(finance_news_provider="serpapi"))
    assert "SERPAPI_API_KEY" in exc.value.message


# ---- integrated into the tool-based researcher ----------------------------------

class LLM:
    provider, model, call_count = "fake", "m", 0

    def __init__(self):
        self.usage = UsageCounters()

    def generate_structured(self, prompt_for, schema, *, system, deadline, stage, request_id=""):
        if schema is QueryPlanDraft:
            return QueryPlanDraft(queries=["q"])
        return NotesDraft(notes=[NoteDraft(text="FT: Nvidia in talks (Oct 8).", source_numbers=[1]),
                                 NoteDraft(text="Web result fact.", source_numbers=[3])])

    def list_models(self):
        return []


class Web:
    name = "searxng"

    def search(self, query, *, max_results, lookback_days, deadline):
        return [SearchHit("Web", "https://example.com/web", "snippet")]

    def check(self):
        return "ok"


class NoFetch:
    def fetch(self, url):
        return None


def run(r):
    return r.run(symbol="NVDA", question="q", lookback_days=7, quant_line="", today=date(2026, 10, 11),
                 deadline=Deadline(60), request_id="t")


def test_finance_news_comes_first_with_real_metadata_and_is_counted():
    llm = LLM()
    finance = serp(lambda r: httpx.Response(200, json=NEWS))
    r = ToolSearchResearcher(llm, make_config(provider="ollama", min_sources=1, finance_news_provider="serpapi"),
                             search_provider=Web(), page_fetcher=NoFetch(), finance_news=finance)
    material = run(r)
    sources = list(material.grounding.sources.values())
    assert sources[0].url == "https://www.ft.com/x" and sources[0].published == parse_news_date("3 days ago")   # S1 is the Google Finance item
    assert "example.com" in sources[1].url
    assert llm.usage.finance_searches == 1


def test_finance_news_failure_never_fails_the_run():
    llm = LLM()
    finance = serp(lambda r: httpx.Response(429, json={}))
    r = ToolSearchResearcher(llm, make_config(provider="ollama", min_sources=1, finance_news_provider="serpapi"),
                             search_provider=Web(), page_fetcher=NoFetch(), finance_news=finance)
    material = run(r)
    assert material.grounding.has_grounding and llm.usage.finance_searches == 0


def test_usage_panel_mentions_serpapi_free_plan(tmp_path):
    from research.cache import ResearchCache
    from research.usage import get_usage_summary, usage_lines

    config = make_config(provider="ollama", finance_news_provider="serpapi")
    cache = ResearchCache(tmp_path / "c.db", 60)
    cache.record_usage(request_id="r", symbol="A", ok=True, cache_hit=False, latency_ms=1, finance_searches=3)
    text = "\n".join(usage_lines(config, get_usage_summary(config, cache)))
    assert "3 of 250" in text and "SerpApi" in text
