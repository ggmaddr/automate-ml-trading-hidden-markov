"""Gemini usage / cost estimates for the research UI and the budget guard.

These are LOCAL ESTIMATES from the API's own usage metadata and the prices in
config/settings.yaml. Google Cloud billing is the source of truth.

Key fact (Google's pricing page, Oct 2026): Google Search grounding on Gemini 3.x
is NOT available on the free tier. It needs a billing-enabled key, which then
includes 5,000 free search requests per month (shared across Gemini 3+ models),
and $14 per 1,000 beyond that.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from research.cache import MonthUsage, ResearchCache
from research.config import ResearchConfig

DEFAULT_SEARCHES_PER_RUN = 6     # used until there is history to average
RATE_LIMIT_URL = "https://aistudio.google.com/rate-limit"
PRICING_URL = "https://ai.google.dev/gemini-api/docs/pricing"


def token_prices(config: ResearchConfig) -> Optional[tuple[float, float]]:
    """(input, output) USD per million tokens for the active provider, or None if unknown."""
    if config.provider == "ollama":
        return (0.0, 0.0)
    if config.provider == "gemini":
        return (config.input_price_per_million_tokens_usd, config.output_price_per_million_tokens_usd)
    entry = config.pricing.get(config.provider)
    if entry and "input_per_mtok" in entry and "output_per_mtok" in entry:
        return (entry["input_per_mtok"], entry["output_per_mtok"])
    return None


def token_cost_usd(config: ResearchConfig, input_tokens: int, output_tokens: int) -> float:
    prices = token_prices(config) or (0.0, 0.0)
    return input_tokens / 1e6 * prices[0] + output_tokens / 1e6 * prices[1]


def search_overage_cost_usd(config: ResearchConfig, searches_before: int, searches_added: int) -> float:
    """Cost of `searches_added` given `searches_before` already used this month
    (only searches past the monthly free allowance are billed)."""
    if config.provider != "gemini":      # only Gemini's Google Search grounding is billed per search
        return 0.0
    free = config.free_grounded_searches_per_month
    billed = max(0, searches_before + searches_added - free) - max(0, searches_before - free)
    return billed / 1000 * config.grounded_search_price_per_1000_usd


def estimate_run_cost_usd(config: ResearchConfig, input_tokens: int, output_tokens: int,
                          grounded_searches: int, month_searches_before: int) -> float:
    return token_cost_usd(config, input_tokens, output_tokens) + search_overage_cost_usd(
        config, month_searches_before, grounded_searches
    )


@dataclass
class UsageSummary:
    provider: str
    cost_known: bool                   # False: provider price not configured -> no spend estimate / budget guard
    input_tokens: int
    output_tokens: int
    grounded_searches: int
    finance_searches: int
    free_allowance: int
    free_remaining: int
    gemini_runs: int
    est_runs_left_free: int            # runs left inside the free search allowance, at the average searches/run
    est_spend_usd: float
    budget_usd: float
    budget_remaining_usd: Optional[float]   # None when the budget guard is off
    over_budget: bool
    avg_searches_per_run: float


def summarize_usage(config: ResearchConfig, month: MonthUsage) -> UsageSummary:
    avg = (month.grounded_searches / month.gemini_runs) if month.gemini_runs else float(DEFAULT_SEARCHES_PER_RUN)
    avg = max(avg, 1.0)
    free_remaining = max(0, config.free_grounded_searches_per_month - month.grounded_searches)
    spend = token_cost_usd(config, month.input_tokens, month.output_tokens) + search_overage_cost_usd(
        config, 0, month.grounded_searches
    )
    budget = config.monthly_budget_usd
    cost_known = token_prices(config) is not None
    return UsageSummary(
        provider=config.provider,
        cost_known=cost_known,
        input_tokens=month.input_tokens,
        output_tokens=month.output_tokens,
        grounded_searches=month.grounded_searches,
        finance_searches=month.finance_searches,
        free_allowance=config.free_grounded_searches_per_month,
        free_remaining=free_remaining,
        gemini_runs=month.gemini_runs,
        est_runs_left_free=int(free_remaining // avg),
        est_spend_usd=spend,
        budget_usd=budget,
        budget_remaining_usd=(max(0.0, budget - spend) if budget > 0 else None),
        over_budget=bool(budget > 0 and cost_known and config.provider != "ollama" and spend >= budget),
        avg_searches_per_run=avg,
    )


def get_usage_summary(config: ResearchConfig, cache: Optional[ResearchCache] = None,
                      now: Optional[datetime] = None) -> UsageSummary:
    cache = cache or ResearchCache(config.resolved_cache_path, config.cache_ttl_minutes)
    return summarize_usage(config, cache.month_usage(now))


def usage_lines(config: ResearchConfig, s: UsageSummary) -> list[str]:
    """Human-readable lines for the UI/CLI."""
    lines = _provider_lines(config, s)
    if config.finance_news_provider == "serpapi":
        lines.insert(0, (
            f"SerpApi Google Finance searches this month: {s.finance_searches:,} of "
            f"{config.serpapi_free_searches_per_month:,} on the free plan (1 per research run; SerpApi doesn't bill its "
            "cached results, so this slightly overcounts). Paid plans start at $25/month for 1,000 searches."
        ))
    return lines


def _provider_lines(config: ResearchConfig, s: UsageSummary) -> list[str]:
    if config.provider == "ollama":
        return [
            f"Local model ({config.model}) via Ollama: no per-request cost and no rate limit; "
            "speed depends on your Mac and the model size.",
            f"Tokens this month: {s.input_tokens:,} in / {s.output_tokens:,} out. Web search is self-hosted SearXNG (free).",
            "Cached results are instant and skip the model entirely.",
        ]
    if config.provider != "gemini":
        if not s.cost_known:
            return [
                f"Provider {config.provider}: cost estimate unavailable. Add research.pricing.{config.provider} "
                "(input_per_mtok / output_per_mtok) in config/settings.yaml to see spend and enable the budget guard.",
                f"Tokens this month: {s.input_tokens:,} in / {s.output_tokens:,} out. Web search is self-hosted SearXNG (free).",
                "Check your provider's dashboard for real billing and rate limits.",
            ]
        return [
            f"Tokens this month: {s.input_tokens:,} in / {s.output_tokens:,} out.",
            f"Estimated spend this month: ${s.est_spend_usd:.2f}"
            + (f" of your ${s.budget_usd:.2f} local budget." if s.budget_usd > 0 else " (no local budget set)."),
            "Estimates come from your configured prices; your provider's billing is authoritative. Cached results are free.",
        ]
    lines = [
        f"Grounded searches this month: {s.grounded_searches:,} of {s.free_allowance:,} free "
        f"(~{s.est_runs_left_free:,} more research runs inside the free allowance).",
        f"Estimated spend this month: ${s.est_spend_usd:.2f}"
        + (f" of your ${s.budget_usd:.2f} local budget." if s.budget_usd > 0 else " (no local budget set)."),
        "Gemini 3.x Google Search grounding is NOT on Gemini's free tier: your API key's project needs billing "
        f"enabled. The first {s.free_allowance:,} search requests/month are free (shared across Gemini 3+ models), then "
        f"${config.grounded_search_price_per_1000_usd:g} per 1,000.",
        "Per-minute / per-day request limits are per project and reset at midnight Pacific; "
        f"see your live limits at {RATE_LIMIT_URL}.",
        "Cached results are free and don't count. These are local estimates; Google Cloud billing is authoritative.",
    ]
    return lines
