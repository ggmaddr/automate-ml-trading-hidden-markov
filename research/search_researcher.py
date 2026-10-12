"""Stage B for providers without built-in search: any LLM + a SearchProvider.

    LLM plans queries -> SearchProvider -> (fetch top pages) -> LLM writes notes
    citing the numbered results -> GroundingData

Provenance is real: sources are search results this process holds, and a
source only counts if at least one note validly cites it. The model never
supplies a URL.
"""

from __future__ import annotations

from datetime import date
from typing import Optional

from research.config import ResearchConfig
from research.exceptions import ResearchError, SchemaValidationError
from research.grounding import GroundedPassage, GroundedSource, GroundingData, host_of, normalize_url
from research.llm.base import Deadline, LLMBackend
from research.logging_util import log_event
from research.prompts import SYSTEM_PROMPT, build_notes_prompt, build_query_plan_prompt
from research.researcher import ResearchMaterial
from research.schemas import NotesDraft, QueryPlanDraft
from research.search import SearchHit, SearchProvider, build_search_provider
from research.search.fetch import PageFetcher

SNIPPET_ONLY_CHARS = 600
MAX_EXCERPT_CHARS = 2500


def _fallback_queries(symbol: str, question: str, round_no: int) -> list[str]:
    if round_no == 1:
        return [f"{symbol} stock news this week", f"{symbol} {question}"[:150], f"{symbol} earnings guidance analyst"]
    return [f"{symbol} latest news", f"{symbol} press release OR 8-K OR SEC filing", f"{symbol} Reuters OR Bloomberg OR CNBC"]


class ToolSearchResearcher:
    name = "searxng"

    def __init__(
        self,
        llm: LLMBackend,
        config: ResearchConfig,
        search_provider: Optional[SearchProvider] = None,
        page_fetcher: Optional[PageFetcher] = None,
        finance_news=None,
    ) -> None:
        self.llm = llm
        self.config = config
        self.search = search_provider or build_search_provider(config)
        self.finance_news = finance_news if finance_news is not None else self._build_finance_news(config)
        self.name = self.search.name
        self.fetcher = page_fetcher or PageFetcher()

    @staticmethod
    def _build_finance_news(config: ResearchConfig):
        if config.finance_news_provider != "serpapi":
            return None
        from research.search.serpapi_finance import SerpApiFinanceNews

        return SerpApiFinanceNews(config)

    def _finance_hits(self, symbol: str, deadline: Deadline, request_id: str) -> list[SearchHit]:
        """Google Finance's news list for the ticker (SerpApi). Auxiliary: any failure is logged and skipped."""
        if self.finance_news is None:
            return []
        before = getattr(self.finance_news, "searches", 0)
        try:
            hits = self.finance_news.news(symbol, max_results=self.config.search_max_results, deadline=deadline)
        except ResearchError as exc:
            log_event("research.search.finance_news_failed", level=30, request_id=request_id, symbol=symbol,
                      error_type=exc.code)
            hits = []
        self.llm.usage.finance_searches += getattr(self.finance_news, "searches", before) - before
        return hits

    # ------------------------------------------------------------------

    def _plan_queries(self, symbol, question, lookback_days, today, round_no, deadline, request_id) -> list[str]:
        try:
            plan = self.llm.generate_structured(
                lambda repair: build_query_plan_prompt(symbol, question, lookback_days, today, round_no, repair),
                QueryPlanDraft, system=SYSTEM_PROMPT, deadline=deadline, stage="query_plan", request_id=request_id,
            )
            queries = [q.strip() for q in plan.queries if q.strip()]
        except SchemaValidationError:
            queries = []   # a weak model must not sink the run: deterministic queries instead
        return queries or _fallback_queries(symbol, question, round_no)

    def run(
        self, *, symbol: str, question: str, lookback_days: int, quant_line: str, today: date,
        deadline: Deadline, request_id: str,
    ) -> ResearchMaterial:
        cfg = self.config
        merged = GroundingData()
        texts: list[str] = []
        rounds = 0
        seen_urls: set[str] = set()

        for round_no in range(1, cfg.max_search_rounds + 1):
            if round_no > 1 and len(merged.sources) >= cfg.min_sources:
                break
            rounds = round_no
            log_event("research.search.started", request_id=request_id, symbol=symbol, round=round_no)
            queries = self._plan_queries(symbol, question, lookback_days, today, round_no, deadline, request_id)

            hits: list[SearchHit] = []
            if round_no == 1:                      # Google Finance news first: curated, dated, real links
                for hit in self._finance_hits(symbol, deadline, request_id):
                    key = normalize_url(hit.url)
                    if key not in seen_urls:
                        seen_urls.add(key)
                        hits.append(hit)
            for query in queries:
                for hit in self.search.search(
                    query, max_results=cfg.search_max_results, lookback_days=lookback_days, deadline=deadline
                ):
                    key = normalize_url(hit.url)
                    if key not in seen_urls:
                        seen_urls.add(key)
                        hits.append(hit)
            hits = hits[: cfg.search_max_results + (cfg.search_max_results // 2 if round_no == 1 and self.finance_news else 0)]
            if not hits:
                log_event("research.search.completed", request_id=request_id, symbol=symbol, round=round_no,
                          source_count=0, query_count=len(queries))
                continue

            numbered: list[tuple[int, GroundedSource, str]] = []
            for number, hit in enumerate(hits, start=1):
                page = self.fetcher.fetch(hit.url) if number <= cfg.fetch_top_n else None
                excerpt = (page[:MAX_EXCERPT_CHARS] if page else hit.snippet[:SNIPPET_ONLY_CHARS]) or hit.title
                source = GroundedSource(
                    key=normalize_url(hit.url), url=hit.url, title=hit.title, domain=host_of(hit.url),
                    published=hit.published,
                )
                numbered.append((number, source, excerpt))

            try:
                notes = self.llm.generate_structured(
                    lambda repair: build_notes_prompt(symbol, question, lookback_days, today, numbered, repair),
                    NotesDraft, system=SYSTEM_PROMPT, deadline=deadline, stage="research_notes", request_id=request_id,
                )
            except SchemaValidationError:
                notes = NotesDraft(notes=[])

            by_number = {n: src for n, src, _ in numbered}
            by_key = {src.key: src for _, src, _ in numbered}
            note_lines: list[str] = []
            for note in notes.notes:
                keys = []
                for number in note.source_numbers:
                    src = by_number.get(number)
                    if src is not None and src.key not in keys:
                        keys.append(src.key)
                if not keys:          # cites nothing real -> not grounded, discard
                    continue
                for key in keys:
                    merged.sources.setdefault(key, by_key[key])
                merged.passages.append(GroundedPassage(text=note.text, source_keys=keys))
                note_lines.append(note.text)
            for query in queries:
                if query not in merged.queries:
                    merged.queries.append(query)
            if note_lines:
                texts.append("\n".join(f"- {line}" for line in note_lines))
            if notes.not_found:
                texts.append("Not covered by the results: " + "; ".join(notes.not_found))
            log_event("research.search.completed", request_id=request_id, symbol=symbol, round=round_no,
                      source_count=len(merged.sources), query_count=len(queries))

        return ResearchMaterial(merged.finalize(cfg.max_sources), "\n\n".join(texts), rounds)
