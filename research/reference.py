"""Three-way comparison with a report the user pasted from elsewhere
(e.g. Google Finance Deep Search, copied by hand). The pasted text is
labelled user-supplied and unverified; nothing here fetches or automates
any third-party site."""

from __future__ import annotations

from datetime import datetime, timezone

from research.llm.base import Deadline, LLMBackend
from research.prompts import SYSTEM_PROMPT, build_reference_comparison_prompt, summarize_quant_line
from research.schemas import QuantContext, ReferenceComparison, ReferenceComparisonDraft, ResearchBrief


def compare_with_reference(
    llm: LLMBackend, quant: QuantContext, brief: ResearchBrief, report: str, deadline: Deadline, request_id: str = ""
) -> ReferenceComparison:
    brief_json = brief.model_dump_json(indent=1, exclude={"search_queries"})
    draft = llm.generate_structured(
        lambda repair: build_reference_comparison_prompt(
            quant.symbol, brief_json, summarize_quant_line(quant), report, repair
        ),
        ReferenceComparisonDraft, system=SYSTEM_PROMPT, deadline=deadline, stage="reference_comparison",
        request_id=request_id,
    )
    return ReferenceComparison(symbol=quant.symbol, generated_at=datetime.now(timezone.utc), **draft.model_dump())


def format_markdown(c: ReferenceComparison) -> str:
    def section(title: str, items: list[str]) -> str:
        return f"### {title}\n" + ("\n".join(f"- {i}" for i in items) if items else "- (none)") + "\n"

    return (
        f"\n## Three-way comparison (agent brief vs pasted report) — {c.generated_at:%Y-%m-%d %H:%M} UTC\n"
        "_The pasted report is user-supplied and unverified; this is not investment advice._\n\n"
        + section("Agreements", c.agreements) + section("Only in the pasted report", c.only_in_reference)
        + section("Only in this system's research", c.only_in_agent) + section("Conflicts", c.conflicts)
        + section("Notes on the pasted report's sourcing", c.reference_quality_notes)
    )
