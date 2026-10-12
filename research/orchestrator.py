"""End-to-end research workflow.

    request
      -> Stage A  internal read-only tools (allowlisted)       -> QuantContext (+ history, backtest)
      -> cache lookup
      -> Stage B  Gemini + Google Search grounding              -> prose + REAL grounding metadata
      -> Stage C  structured synthesis (Pydantic-validated)     -> ResearchBrief
                  quant-vs-research comparison                  -> QuantResearchComparison
      -> ResearchResult (cached)

`ResearchOrchestrator.research()` never raises: every failure becomes a
typed `ResearchOutcome` error, and nothing here can touch trading state.
"""

from __future__ import annotations

import re
import time
import uuid
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from research.llm import build_llm
from research.llm.base import Deadline, LLMBackend
from research.researcher import Researcher, build_researcher
from research.cache import ResearchCache, make_cache_key, quant_fingerprint
from research.comparison import compare_quant_and_research
from research.config import ResearchConfig, load_research_config, load_settings
from research.exceptions import (
    QuantContextUnavailableError,
    ResearchBudgetError,
    ResearchDisabledError,
    ResearchError,
    ResearchNotConfiguredError,
    ToolBudgetExceededError,
    ToolError,
)
from research.grounding import GroundingData, host_of
from research.logging_util import log_event, redact
from research.prompts import (
    RESEARCH_PROMPT_VERSION,
    SYSTEM_PROMPT,
    build_synthesis_prompt,
    effective_question,
    summarize_quant_line,
)
from research.schemas import (
    BacktestContext,
    BriefDraft,
    Evidence,
    EvidenceDraft,
    QuantContext,
    RegimeHistory,
    ResearchBrief,
    ResearchDiagnostics,
    ResearchUsage,
    ResearchErrorInfo,
    ResearchOutcome,
    ResearchRequest,
    ResearchResult,
)
from research.tools import ResearchToolRegistry
from research.usage import estimate_run_cost_usd, summarize_usage
from research.tools.backtest_tools import DEFAULT_RESULTS_DIR, BacktestReader
from research.tools.hmm_tools import DEFAULT_MODEL_PATH, HMMRuntimeReader

_REGIME_WORDS = re.compile(r"\b(regime|trend|change|changed|shift|volatil\w*|transition|stabil\w*)\b", re.IGNORECASE)
_INSUFFICIENT_CONFIDENCE_CAP = 0.3


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _parse_published(value: Optional[str]) -> Optional[datetime]:
    """Parse the model's date string; anything unparseable becomes None (never a guessed date)."""
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed


class ResearchOrchestrator:
    def __init__(
        self,
        config: Optional[ResearchConfig] = None,
        settings: Optional[dict] = None,
        *,
        llm: Optional[LLMBackend] = None,
        researcher: Optional[Researcher] = None,
        agent: Optional[LLMBackend] = None,                      # legacy alias for `llm` (Gemini-native research)
        agent_factory: Optional[Callable[[ResearchConfig], LLMBackend]] = None,
        search_provider=None,
        page_fetcher=None,
        cache: Optional[ResearchCache] = None,
        hmm_reader: Optional[HMMRuntimeReader] = None,
        backtest_reader: Optional[BacktestReader] = None,
        redirect_resolver: Optional[Callable[[str], Optional[str]]] = None,
        clock: Callable[[], datetime] = _utcnow,
    ) -> None:
        if settings is None:
            settings = load_settings()
        self.config = config or load_research_config(settings)
        self._llm = llm or agent
        self._researcher = researcher
        self._agent_factory = agent_factory
        self._search_provider = search_provider
        self._page_fetcher = page_fetcher
        self.cache = cache or ResearchCache(self.config.resolved_cache_path, self.config.cache_ttl_minutes)
        self.hmm_reader = hmm_reader or HMMRuntimeReader(DEFAULT_MODEL_PATH)
        self.backtest_reader = backtest_reader or BacktestReader(
            DEFAULT_RESULTS_DIR, risk_free_rate=settings.get("backtest", {}).get("risk_free_rate", 0.045)
        )
        self._redirect_resolver = redirect_resolver
        self._clock = clock
        self._run_usage: Optional[tuple[LLMBackend, object]] = None   # (llm, counters at start of this run)

    def _get_backends(self) -> tuple[LLMBackend, Researcher]:
        """Build (lazily) the configured LLM + researcher. Raises typed config errors."""
        if self._llm is None:
            self._llm = self._agent_factory(self.config) if self._agent_factory else build_llm(self.config)
        if self._researcher is None:
            self._researcher = build_researcher(
                self.config, self._llm, redirect_resolver=self._redirect_resolver,
                search_provider=self._search_provider, page_fetcher=self._page_fetcher,
            )
        return self._llm, self._researcher

    def get_llm(self) -> LLMBackend:
        """The configured LLM backend (built lazily); raises typed config errors."""
        return self._get_backends()[0]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def research(self, request: ResearchRequest) -> ResearchOutcome:
        request_id = uuid.uuid4().hex[:12]
        started = time.monotonic()
        self._run_usage = None
        quant: Optional[QuantContext] = None
        model = self.config.model
        log_event("research.request.started", request_id=request_id, symbol=request.symbol,
                  model=model, prompt_version=RESEARCH_PROMPT_VERSION, force_refresh=request.force_refresh)
        error: ResearchError
        try:
            result = self._run(request, request_id, started)
            self._record(request_id, request, ok=True, result=result)
            return ResearchOutcome(ok=True, result=result, quant_context=result.quant_context, request_id=request_id)
        except _QuantCarrier as carrier:  # failed after quant was collected: keep the snapshot for the UI
            quant, error = carrier.quant, carrier.error
        except ResearchError as exc:
            error = exc
        except Exception as exc:  # noqa: BLE001 - the sidecar must never take down the caller
            error = _InternalError(f"{type(exc).__name__}: {redact(exc)[:200]}")

        latency_ms = int((time.monotonic() - started) * 1000)
        log_event("research.failed", level=40, request_id=request_id, symbol=request.symbol, model=model,
                  error_type=error.code, latency_ms=latency_ms)
        self._record(request_id, request, ok=False, error_code=error.code, latency_ms=latency_ms)
        return ResearchOutcome(
            ok=False,
            error=ResearchErrorInfo(code=error.code, message=error.message, retryable=error.retryable),
            quant_context=quant,
            request_id=request_id,
        )

    # ------------------------------------------------------------------
    # Pipeline
    # ------------------------------------------------------------------

    def _run(self, request: ResearchRequest, request_id: str, started: float) -> ResearchResult:
        cfg = self.config
        if not cfg.enabled:
            raise ResearchDisabledError()

        # ---- Stage A: internal read-only context ---------------------------------
        registry = ResearchToolRegistry(self.hmm_reader, self.backtest_reader, cfg.max_internal_tool_calls, request_id)
        quant, history, backtest = self._collect_context(request, registry)

        try:
            return self._run_external(request, request_id, started, registry, quant, history, backtest)
        except ResearchError as exc:
            raise _QuantCarrier(quant, exc) from exc

    def _collect_context(
        self, request: ResearchRequest, registry: ResearchToolRegistry
    ) -> tuple[QuantContext, Optional[RegimeHistory], Optional[BacktestContext]]:
        symbol = request.symbol
        try:
            quant: QuantContext = registry.call("get_hmm_snapshot", symbol=symbol)
        except ToolError as exc:
            if not request.allow_without_quant:
                raise QuantContextUnavailableError(exc.detail) from exc
            quant = QuantContext(symbol=symbol, timestamp=self._clock().replace(tzinfo=None))

        history: Optional[RegimeHistory] = None
        wants_history = bool(_REGIME_WORDS.search(request.question or "")) or not quant.is_confirmed or quant.is_flickering
        if quant.regime_id is not None and wants_history:
            history = self._optional_tool(registry, "get_regime_history", symbol=symbol, bars=60)

        backtest: Optional[BacktestContext] = self._optional_tool(registry, "get_backtest_summary", symbol=symbol)
        return quant, history, backtest

    @staticmethod
    def _optional_tool(registry: ResearchToolRegistry, name: str, **kwargs):
        """Optional context: a failure is logged and recorded in the trace, never fatal."""
        try:
            return registry.call(name, **kwargs)
        except ToolBudgetExceededError:
            return None
        except ToolError:
            return None

    def _run_external(
        self, request, request_id, started, registry, quant, history, backtest,
    ) -> ResearchResult:
        cfg = self.config
        cache_key = make_cache_key(request, cfg, quant_fingerprint(quant))

        if not request.force_refresh:
            cached = self.cache.get(cache_key)
            if cached is not None:
                log_event("research.cache.hit", request_id=request_id, symbol=request.symbol, model=cfg.model)
                return cached
            log_event("research.cache.miss", request_id=request_id, symbol=request.symbol, model=cfg.model)
        else:
            log_event("research.cache.miss", request_id=request_id, symbol=request.symbol, model=cfg.model,
                      reason="force_refresh")

        agent, researcher = self._get_backends()  # typed config errors (no key / no model / ...)

        month = self.cache.month_usage()
        if summarize_usage(cfg, month).over_budget:
            raise ResearchBudgetError()
        self._run_usage = (agent, agent.usage.snapshot())
        deadline = Deadline(cfg.overall_timeout_seconds)
        question = effective_question(request.symbol, request.question, request.lookback_days)

        # ---- Stage B: grounded external research ---------------------------------
        material = researcher.run(
            symbol=request.symbol, question=question, lookback_days=request.lookback_days,
            quant_line=summarize_quant_line(quant), today=self._clock().date(), deadline=deadline, request_id=request_id,
        )
        grounding, grounded_text, rounds = material.grounding, material.text, material.rounds
        diagnostics = ResearchDiagnostics(
            request_id=request_id, provider=cfg.provider, search_provider=researcher.name, model=cfg.model,
            prompt_version=RESEARCH_PROMPT_VERSION, search_rounds=rounds, grounding_available=grounding.has_grounding,
            source_domains=sorted({s.domain or host_of(s.url) or "" for s in grounding.sources.values()} - {""}),
        )

        # ---- Stage C: structured synthesis + comparison ---------------------------
        if grounding.has_grounding:
            brief, dropped = self._synthesize(
                agent, request, question, quant, history, backtest, grounded_text, grounding, deadline, request_id
            )
            diagnostics.dropped_unsupported_claims = dropped
        else:
            diagnostics.notes.append("Web search returned no usable sources; no ungrounded claims were used.")
            brief = self._empty_brief(request, question, grounding)

        comparison = compare_quant_and_research(agent, quant, brief, history, backtest, deadline, request_id)

        diagnostics.llm_calls = agent.call_count
        used = self._consumed()
        diagnostics.usage = ResearchUsage(
            input_tokens=used.input_tokens, output_tokens=used.output_tokens, grounded_searches=used.grounded_searches,
            finance_searches=used.finance_searches,
            estimated_cost_usd=estimate_run_cost_usd(
                cfg, used.input_tokens, used.output_tokens, used.grounded_searches, month.grounded_searches
            ),
        )
        diagnostics.tool_calls = list(registry.calls)
        latency_ms = int((time.monotonic() - started) * 1000)
        result = ResearchResult(
            request=request, quant_context=quant, backtest_context=backtest, brief=brief,
            comparison=comparison, cache_hit=False, latency_ms=latency_ms, diagnostics=diagnostics,
        )
        self.cache.put(cache_key, result)
        log_event(
            "research.completed", request_id=request_id, symbol=request.symbol, model=cfg.model,
            prompt_version=RESEARCH_PROMPT_VERSION, latency_ms=latency_ms, source_count=brief.source_count,
            cache_hit=False, stance=brief.evidence_stance,
        )
        return result

    # ------------------------------------------------------------------
    # Stage C
    # ------------------------------------------------------------------

    def _synthesize(
        self, agent, request, question, quant, history, backtest, grounded_text, grounding, deadline, request_id,
    ) -> tuple[ResearchBrief, int]:
        cfg = self.config
        draft: BriefDraft = agent.generate_structured(
            lambda repair: build_synthesis_prompt(
                request.symbol, question, request.lookback_days, quant, history, backtest,
                grounded_text, grounding, repair,
            ),
            BriefDraft,
            system=SYSTEM_PROMPT, deadline=deadline, stage="synthesis", request_id=request_id,
        )

        retrieved_at = self._clock()
        sources = grounding.source_by_id()
        unsupported: list[str] = []

        def to_evidence(items: list[EvidenceDraft]) -> list[Evidence]:
            out: list[Evidence] = []
            for item in items:
                src = sources.get(item.source_id.strip().upper())
                if src is None:
                    unsupported.append(item.claim)
                    continue
                out.append(Evidence(
                    claim=item.claim, category=item.category, impact=item.impact,
                    source_title=src.title, source_url=src.url,
                    source_domain=src.domain or host_of(src.url),
                    published_at=_parse_published(item.published_at) or _parse_published(src.published),
                    retrieved_at=retrieved_at, confidence=item.confidence,
                ))
            return out

        catalysts, risks, events = to_evidence(draft.catalysts), to_evidence(draft.risks), to_evidence(draft.recent_events)
        unknowns = list(draft.unknowns) + [f"Unverified (no grounded source): {c}" for c in unsupported]

        source_count = len(sources)
        stance = draft.evidence_stance
        confidence = draft.overall_confidence
        if source_count < cfg.min_sources or not (catalysts or risks or events):
            stance = "insufficient_evidence"
            if source_count < cfg.min_sources:
                unknowns.append(
                    f"Only {source_count} distinct grounded source(s) found (target: at least {cfg.min_sources})."
                )

        if stance == "insufficient_evidence":      # whether the model said so or the rules above forced it
            confidence = min(confidence, _INSUFFICIENT_CONFIDENCE_CAP)

        brief = ResearchBrief(
            symbol=request.symbol, generated_at=retrieved_at, question=question, summary=draft.summary,
            catalysts=catalysts, risks=risks, recent_events=events,
            conflicting_evidence=list(draft.conflicting_evidence), unknowns=unknowns,
            evidence_stance=stance, overall_confidence=confidence,
            source_count=source_count, search_queries=list(grounding.queries),
        )
        return brief, len(unsupported)

    def _empty_brief(self, request, question, grounding: GroundingData) -> ResearchBrief:
        return ResearchBrief(
            symbol=request.symbol, generated_at=self._clock(), question=question,
            summary="No grounded external sources were returned, so no current external evidence is reported.",
            unknowns=["Google Search grounding returned no sources for this request."],
            evidence_stance="insufficient_evidence", overall_confidence=0.0, source_count=0,
            search_queries=list(grounding.queries),
        )

    # ------------------------------------------------------------------

    def _usage_kwargs(self) -> dict:
        used = self._consumed()
        return {"input_tokens": used.input_tokens, "output_tokens": used.output_tokens,
                "grounded_searches": used.grounded_searches, "finance_searches": used.finance_searches}

    def _consumed(self):
        """Gemini consumption of the current run (zero if it never reached Gemini)."""
        from research.llm.base import UsageCounters

        if self._run_usage is None:
            return UsageCounters()
        agent, start = self._run_usage
        return agent.usage.since(start)

    def _record(self, request_id, request, ok, result: Optional[ResearchResult] = None,
                error_code: Optional[str] = None, latency_ms: int = 0) -> None:
        try:
            if result is not None:
                self.cache.record_usage(
                    request_id=request_id, symbol=request.symbol, ok=True, cache_hit=result.cache_hit,
                    latency_ms=result.latency_ms, search_queries=len(result.brief.search_queries),
                    source_count=result.brief.source_count, model=self.config.model,
                    **({} if result.cache_hit else self._usage_kwargs()),
                )
            else:
                self.cache.record_usage(
                    request_id=request_id, symbol=request.symbol, ok=False, cache_hit=False,
                    latency_ms=latency_ms, model=self.config.model, error_code=error_code,
                    **self._usage_kwargs(),
                )
        except Exception as exc:  # noqa: BLE001 - metrics must never break research
            log_event("research.metrics.failed", level=30, request_id=request_id, error_type=type(exc).__name__)


class _QuantCarrier(Exception):
    """Internal: carries the already-collected quant snapshot alongside a failure."""

    def __init__(self, quant: QuantContext, error: ResearchError) -> None:
        super().__init__(str(error))
        self.quant = quant
        self.error = error


class _InternalError(ResearchError):
    code = "internal_error"
    user_message = "Research failed unexpectedly. Quantitative system continues normally."
