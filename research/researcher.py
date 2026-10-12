"""Stage B: produce grounded research material, whatever the provider.

A `Researcher` returns prose plus `GroundingData` (sources with S-ids,
queries, source-attributed passages). Two implementations:

- `NativeGroundedResearcher` — Gemini + its built-in Google Search grounding.
- `ToolSearchResearcher` (research/search_researcher.py) — any LLM plus a
  `SearchProvider` such as SearXNG and a safe page fetcher.

In both, source provenance comes from search results the application
actually holds, never from text the model wrote.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Callable, Optional, Protocol

from research.agent import GeminiAgent
from research.config import ResearchConfig
from research.grounding import GroundingData, resolve_redirects
from research.llm.base import Deadline
from research.logging_util import log_event
from research.prompts import SYSTEM_PROMPT, build_grounded_search_prompt


@dataclass
class ResearchMaterial:
    grounding: GroundingData      # finalized: capped, S-ids assigned
    text: str                     # prose from rounds that had real grounding
    rounds: int                   # search rounds actually run


class Researcher(Protocol):
    name: str

    def run(
        self, *, symbol: str, question: str, lookback_days: int, quant_line: str, today: date,
        deadline: Deadline, request_id: str,
    ) -> ResearchMaterial: ...


class NativeGroundedResearcher:
    """Gemini + Google Search grounding, up to `max_search_rounds` rounds."""

    name = "gemini_grounding"

    def __init__(
        self,
        agent: GeminiAgent,
        config: ResearchConfig,
        redirect_resolver: Optional[Callable[[str], Optional[str]]] = None,
    ) -> None:
        self.agent = agent
        self.config = config
        self._redirect_resolver = redirect_resolver

    def run(
        self, *, symbol: str, question: str, lookback_days: int, quant_line: str, today: date,
        deadline: Deadline, request_id: str,
    ) -> ResearchMaterial:
        cfg = self.config
        merged = GroundingData()
        texts: list[str] = []
        rounds = 0
        if not cfg.enable_google_search:
            return ResearchMaterial(merged.finalize(cfg.max_sources), "", 0)

        for round_no in range(1, cfg.max_search_rounds + 1):
            if round_no > 1 and len(merged.sources) >= cfg.min_sources:
                break
            rounds = round_no
            prompt = build_grounded_search_prompt(
                symbol, question, lookback_days, quant_line, today, round_no,
                prior_summary="\n\n".join(texts)[:2000] or None,
            )
            log_event("research.search.started", request_id=request_id, symbol=symbol, round=round_no)
            response = self.agent.grounded_search(prompt, system=SYSTEM_PROMPT, deadline=deadline, request_id=request_id)
            log_event(
                "research.search.completed", request_id=request_id, symbol=symbol, round=round_no,
                source_count=len(response.grounding.sources), query_count=len(response.grounding.queries),
            )
            # Prose from a round with no grounding metadata is discarded: it
            # may come from the model's memory, not from search.
            if response.grounding.has_grounding:
                texts.append(response.text)
                merged.merge(response.grounding)

        if merged.has_grounding and cfg.resolve_source_redirects:
            kwargs = {"resolver": self._redirect_resolver} if self._redirect_resolver else {}
            merged = resolve_redirects(merged, **kwargs)
        return ResearchMaterial(merged.finalize(cfg.max_sources), "\n\n".join(texts), rounds)


def build_researcher(
    config: ResearchConfig,
    llm,
    *,
    redirect_resolver: Optional[Callable[[str], Optional[str]]] = None,
    search_provider=None,
    page_fetcher=None,
) -> Researcher:
    """Pick the Stage B implementation from `research.search_provider`."""
    from research.exceptions import ResearchNotConfiguredError

    mode = config.effective_search_provider
    if mode == "gemini_grounding":
        if not hasattr(llm, "grounded_search"):
            raise ResearchNotConfiguredError(
                "search_provider 'gemini_grounding' needs provider 'gemini'; use 'searxng' with other providers"
            )
        return NativeGroundedResearcher(llm, config, redirect_resolver)
    from research.search_researcher import ToolSearchResearcher

    return ToolSearchResearcher(llm, config, search_provider=search_provider, page_fetcher=page_fetcher)
