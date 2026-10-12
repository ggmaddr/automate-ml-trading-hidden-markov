"""Live readiness probes for `python main.py research --check` (network I/O)."""

from __future__ import annotations

from dataclasses import dataclass

from research.config import ResearchConfig, get_api_key
from research.exceptions import ResearchError


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str


def run_checks(config: ResearchConfig) -> list[CheckResult]:
    results: list[CheckResult] = [
        CheckResult("config", True, f"provider={config.provider} model={config.model or '(none)'} "
                                    f"search={config.effective_search_provider}")
    ]

    # LLM backend
    try:
        if config.provider == "ollama":
            from research.llm import build_llm

            models = build_llm(config).list_models()
            wanted = config.model
            have = wanted in models or f"{wanted}:latest" in models
            results.append(CheckResult(
                "llm", have, f"Ollama reachable; model '{wanted}' " + ("is available" if have else
                f"NOT pulled (have: {', '.join(models) or 'none'}). Run `ollama pull {wanted}`"),
            ))
        else:
            has_key = bool(get_api_key(config=config))
            results.append(CheckResult(
                "llm", has_key and bool(config.model),
                f"{config.resolved_api_key_env} {'set' if has_key else 'MISSING'}; model "
                f"{config.model or 'NOT configured (research.models.' + config.provider + ')'} (key not validated offline)",
            ))
    except ResearchError as exc:
        results.append(CheckResult("llm", False, exc.message))

    # Search
    if config.effective_search_provider == "gemini_grounding":
        results.append(CheckResult("search", config.provider == "gemini", "Gemini native Google Search grounding"
                                   if config.provider == "gemini" else "gemini_grounding requires provider gemini"))
    else:
        try:
            from research.search import build_search_provider

            results.append(CheckResult("search", True, build_search_provider(config).check()))
        except ResearchError as exc:
            results.append(CheckResult("search", False, exc.message))
    if config.finance_news_provider == "serpapi":
        try:
            from research.search.serpapi_finance import SerpApiFinanceNews

            results.append(CheckResult("finance_news", True, "SerpApi Google Finance: " + SerpApiFinanceNews(config).check()
                                       + " (uses 1 of your SerpApi searches)"))
        except ResearchError as exc:
            results.append(CheckResult("finance_news", False, exc.message))
    return results
