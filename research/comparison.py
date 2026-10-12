"""Quant-vs-research comparison.

Answers one question: what does current external evidence add that the
quantitative model cannot see, and do the two sources look supportive,
conflicting, orthogonal, or insufficient? It never emits a trade
instruction. Numbers in `quant_summary` come from Python; Gemini supplies
only the qualitative comparison, which deterministic guardrails then bound.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from research.agent import Deadline, GeminiAgent
from research.prompts import SYSTEM_PROMPT, build_comparison_prompt, summarize_quant_line
from research.schemas import (
    BacktestContext,
    ComparisonDraft,
    QuantContext,
    QuantResearchComparison,
    RegimeHistory,
    ResearchBrief,
)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def deterministic_comparison(
    quant: QuantContext, brief: ResearchBrief, reason: str
) -> QuantResearchComparison:
    """No-LLM comparison for the cases where there is nothing to compare."""
    return QuantResearchComparison(
        symbol=quant.symbol,
        generated_at=_now(),
        quant_summary=summarize_quant_line(quant),
        research_summary=brief.summary,
        relationship="insufficient_evidence",
        agreements=[],
        contradictions=[],
        information_quant_model_cannot_see=[],
        information_research_cannot_establish=[reason],
        human_review_recommended=True,
        explanation=reason,
    )


def apply_guardrails(
    draft: ComparisonDraft, quant: QuantContext, brief: ResearchBrief
) -> QuantResearchComparison:
    """Bound the model's judgement with facts Python knows for certain."""
    relationship = draft.relationship
    review = draft.human_review_recommended
    explanation = draft.explanation

    if brief.evidence_stance == "insufficient_evidence":
        relationship = "insufficient_evidence"
    if quant.regime_id is None:
        relationship = "insufficient_evidence"

    if relationship != draft.relationship:
        explanation = (
            f"[Relationship set to '{relationship}' by a deterministic check: the sourced evidence or the quant "
            f"snapshot is too thin to compare reliably. The model's draft said '{draft.relationship}'.] {explanation}"
        )

    if (
        relationship in ("conflicting", "insufficient_evidence")
        or bool(brief.conflicting_evidence)
        or quant.is_flickering
        or not quant.is_confirmed
    ):
        review = True

    return QuantResearchComparison(
        symbol=quant.symbol,
        generated_at=_now(),
        quant_summary=summarize_quant_line(quant),
        research_summary=draft.research_summary,
        relationship=relationship,
        agreements=draft.agreements,
        contradictions=draft.contradictions,
        information_quant_model_cannot_see=draft.information_quant_model_cannot_see,
        information_research_cannot_establish=draft.information_research_cannot_establish,
        human_review_recommended=review,
        explanation=explanation,
    )


def compare_quant_and_research(
    agent: GeminiAgent,
    quant: QuantContext,
    brief: ResearchBrief,
    history: Optional[RegimeHistory],
    backtest: Optional[BacktestContext],
    deadline: Deadline,
    request_id: str = "",
) -> QuantResearchComparison:
    if quant.regime_id is None:
        return deterministic_comparison(quant, brief, "No quantitative (HMM) context was available to compare against.")
    if brief.evidence_stance == "insufficient_evidence" and not (brief.catalysts or brief.risks or brief.recent_events):
        return deterministic_comparison(quant, brief, "External research produced too little sourced evidence to compare.")

    brief_json = brief.model_dump_json(indent=1, exclude={"search_queries"})
    draft = agent.generate_structured(
        lambda repair: build_comparison_prompt(quant, history, backtest, brief_json, repair),
        ComparisonDraft,
        system=SYSTEM_PROMPT,
        deadline=deadline,
        stage="comparison",
        request_id=request_id,
    )
    return apply_guardrails(draft, quant, brief)
