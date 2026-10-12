"""Strict Pydantic schemas for every research-agent boundary."""

from __future__ import annotations

import re
import unicodedata
from datetime import date, datetime
from typing import Any, Literal, Optional
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, field_validator

MIN_LOOKBACK_DAYS = 1
MAX_LOOKBACK_DAYS = 90
MAX_QUESTION_CHARS = 500
_SYMBOL_PATTERN = re.compile(r"^[A-Z0-9][A-Z0-9.\-]{0,9}$")

EvidenceCategory = Literal[
    "earnings", "guidance", "product", "ai", "cloud", "competition",
    "regulation", "management", "macro", "valuation", "other",
]
EvidenceImpact = Literal["positive", "negative", "mixed", "unclear"]
EvidenceStance = Literal["positive", "negative", "mixed", "insufficient_evidence"]
Relationship = Literal["supportive", "partially_supportive", "conflicting", "orthogonal", "insufficient_evidence"]


def _check_http_url(value: str) -> str:
    parsed = urlparse(value)
    if parsed.scheme not in ("http", "https") or not parsed.netloc or " " in value:
        raise ValueError(f"not a valid http(s) URL: {value!r}")
    return value


class ResearchRequest(BaseModel):
    symbol: str
    question: Optional[str] = None
    lookback_days: int = 14
    force_refresh: bool = False
    # Explicit opt-in: still produce external research when no HMM snapshot is available.
    allow_without_quant: bool = False

    @field_validator("symbol")
    @classmethod
    def _normalize_symbol(cls, v: str) -> str:
        v = (v or "").strip().upper()
        if not v:
            raise ValueError("symbol must not be empty")
        if not _SYMBOL_PATTERN.match(v):
            raise ValueError(f"invalid ticker symbol: {v!r}")
        return v

    @field_validator("question")
    @classmethod
    def _sanitize_question(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        v = "".join(ch for ch in v if unicodedata.category(ch)[0] != "C" or ch in "\n\t ")
        v = re.sub(r"\s+", " ", v).strip()
        return v[:MAX_QUESTION_CHARS] or None

    @field_validator("lookback_days")
    @classmethod
    def _cap_lookback(cls, v: int) -> int:
        return max(MIN_LOOKBACK_DAYS, min(int(v), MAX_LOOKBACK_DAYS))


class QuantContext(BaseModel):
    """Snapshot of the existing HMM system. Built by research.tools.hmm_tools
    from the existing engine — nothing in `research` recomputes HMM logic.

    `regime_label` is the return-ordered readability label; strategy behaviour
    is driven by volatility (see expected_volatility / volatility_rank).
    `flicker_rate` is the raw count of regime flips in the engine's flicker window.
    """

    symbol: str
    timestamp: datetime  # as-of time of the last bar the snapshot used

    regime_id: Optional[int] = None
    regime_label: Optional[str] = None
    regime_probability: Optional[float] = None
    state_probabilities: dict[str, float] = Field(default_factory=dict)

    expected_volatility: Optional[float] = None
    expected_return: Optional[float] = None
    volatility_rank: Optional[float] = None  # 0.0 = calmest regime, 1.0 = most turbulent

    is_confirmed: bool = False
    consecutive_bars: int = 0
    flicker_rate: Optional[float] = None
    is_flickering: bool = False

    model_version: Optional[str] = None
    feature_version: Optional[str] = None

    recent_return: Optional[float] = None       # return over the last 20 bars
    recent_volatility: Optional[float] = None   # per-bar std of log returns over the last 20 bars


class BacktestContext(BaseModel):
    symbol: str
    available: bool
    period_start: Optional[date] = None
    period_end: Optional[date] = None
    total_return: Optional[float] = None
    sharpe: Optional[float] = None
    max_drawdown: Optional[float] = None
    win_rate: Optional[float] = None
    regime_metrics: dict[str, Any] = Field(default_factory=dict)


class RegimeHistoryPoint(BaseModel):
    timestamp: datetime
    regime_id: int
    label: str
    probability: float
    expected_volatility: Optional[float] = None
    is_confirmed: bool
    consecutive_bars: int
    flicker_rate: float
    is_flickering: bool


class RegimeHistory(BaseModel):
    symbol: str
    points: list[RegimeHistoryPoint]  # chronological (oldest first)


class Evidence(BaseModel):
    claim: str = Field(min_length=1)
    category: EvidenceCategory
    impact: EvidenceImpact
    source_title: str
    source_url: str
    source_domain: Optional[str] = None
    published_at: Optional[datetime] = None
    retrieved_at: datetime
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("source_url")
    @classmethod
    def _valid_url(cls, v: str) -> str:
        return _check_http_url(v)


class ResearchBrief(BaseModel):
    symbol: str
    generated_at: datetime
    question: str

    summary: str
    catalysts: list[Evidence] = Field(default_factory=list)
    risks: list[Evidence] = Field(default_factory=list)
    recent_events: list[Evidence] = Field(default_factory=list)

    conflicting_evidence: list[str] = Field(default_factory=list)
    unknowns: list[str] = Field(default_factory=list)

    evidence_stance: EvidenceStance
    overall_confidence: float = Field(ge=0.0, le=1.0)
    source_count: int = Field(ge=0)
    search_queries: list[str] = Field(default_factory=list)


class QuantResearchComparison(BaseModel):
    symbol: str
    generated_at: datetime

    quant_summary: str
    research_summary: str
    relationship: Relationship

    agreements: list[str] = Field(default_factory=list)
    contradictions: list[str] = Field(default_factory=list)
    information_quant_model_cannot_see: list[str] = Field(default_factory=list)
    information_research_cannot_establish: list[str] = Field(default_factory=list)

    human_review_recommended: bool
    explanation: str


class ToolCallRecord(BaseModel):
    tool: str
    args: dict[str, Any] = Field(default_factory=dict)
    ok: bool
    error: Optional[str] = None
    duration_ms: int = 0


class ResearchUsage(BaseModel):
    """What one research run consumed (Gemini usage metadata; searches = grounding queries issued)."""

    input_tokens: int = 0
    output_tokens: int = 0
    grounded_searches: int = 0
    finance_searches: int = 0
    estimated_cost_usd: float = 0.0


class ResearchDiagnostics(BaseModel):
    """Trace for the developer expander. Final structured fields only — no model reasoning."""

    request_id: str = ""
    provider: str = ""
    search_provider: str = ""
    model: str = ""
    prompt_version: str = ""
    search_rounds: int = 0
    llm_calls: int = 0
    source_domains: list[str] = Field(default_factory=list)
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    grounding_available: bool = True
    dropped_unsupported_claims: int = 0
    usage: ResearchUsage = Field(default_factory=ResearchUsage)
    notes: list[str] = Field(default_factory=list)


class ResearchResult(BaseModel):
    request: ResearchRequest
    quant_context: QuantContext
    backtest_context: Optional[BacktestContext] = None
    brief: ResearchBrief
    comparison: QuantResearchComparison
    cache_hit: bool = False
    latency_ms: int = 0
    diagnostics: ResearchDiagnostics = Field(default_factory=ResearchDiagnostics)


class ResearchErrorInfo(BaseModel):
    code: str
    message: str
    retryable: bool = False


class ResearchOutcome(BaseModel):
    """What callers (CLI/Streamlit) receive. The orchestrator never raises:
    failure is a typed state, and the quant snapshot is kept when available
    so the UI can still show it."""

    ok: bool
    result: Optional[ResearchResult] = None
    error: Optional[ResearchErrorInfo] = None
    quant_context: Optional[QuantContext] = None
    request_id: str = ""


# ---------------------------------------------------------------------------
# Model-facing draft schemas (what Gemini is asked to emit)
#
# Deliberately NOT the same as the public schemas: the model cites sources by
# id, and Python resolves ids to real URLs/titles/domains from grounding
# metadata. The model can therefore never fabricate a citation.
# ---------------------------------------------------------------------------

class EvidenceDraft(BaseModel):
    claim: str = Field(min_length=1)
    category: EvidenceCategory
    impact: EvidenceImpact
    source_id: str
    published_at: Optional[str] = None
    confidence: float = Field(ge=0.0, le=1.0)


class BriefDraft(BaseModel):
    summary: str = Field(min_length=1)
    catalysts: list[EvidenceDraft]
    risks: list[EvidenceDraft]
    recent_events: list[EvidenceDraft]
    conflicting_evidence: list[str]
    unknowns: list[str]
    evidence_stance: EvidenceStance
    overall_confidence: float = Field(ge=0.0, le=1.0)


class ComparisonDraft(BaseModel):
    research_summary: str = Field(min_length=1)
    relationship: Relationship
    agreements: list[str]
    contradictions: list[str]
    information_quant_model_cannot_see: list[str]
    information_research_cannot_establish: list[str]
    human_review_recommended: bool
    explanation: str = Field(min_length=1)


class QueryPlanDraft(BaseModel):
    queries: list[str] = Field(min_length=1, max_length=5)


class NoteDraft(BaseModel):
    text: str = Field(min_length=1)
    source_numbers: list[int] = Field(min_length=1)   # numbers from the SEARCH RESULTS list the note is based on


class NotesDraft(BaseModel):
    notes: list[NoteDraft]
    not_found: list[str] = Field(default_factory=list)


class ReferenceComparisonDraft(BaseModel):
    agreements: list[str]
    only_in_reference: list[str]
    only_in_agent: list[str]
    conflicts: list[str]
    reference_quality_notes: list[str]


class ReferenceComparison(ReferenceComparisonDraft):
    symbol: str
    generated_at: datetime
    reference_is_user_supplied: bool = True   # the pasted report is unverified text, never ground truth
