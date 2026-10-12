"""AI Research view (Phase 10) -- read-only; never places orders.

All research text (claims, titles, URLs) comes from the web or a model and is
UNTRUSTED: it is only ever rendered through ui.html builders, which escape it.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from ui import charts
from ui import html as h
from views.common import load_settings


def _research_config():
    from research.config import load_research_config

    return load_research_config(load_settings())


def _usage_panel(config) -> None:
    """Free-tier / spend limits, shown right where the user clicks Research. Never raises."""
    try:
        from research.usage import get_usage_summary, usage_lines

        summary = get_usage_summary(config)
        lines = usage_lines(config, summary)
        st.html(h.notice_list(lines, tone="halt" if summary.over_budget else "neutral",
                              title=f"Usage & limits — {config.provider} ({config.model or 'no model set'})"))
        if summary.over_budget:
            st.caption("Budget reached: new paid runs are blocked (cached results still load). "
                       "Raise research.monthly_budget_usd in config/settings.yaml to continue.")
    except Exception as exc:  # noqa: BLE001 - never block the page over a usage display
        st.caption(f"Usage information unavailable ({type(exc).__name__}).")


def _evidence_section(title: str, items) -> None:
    from research.presentation import age_label  # noqa: F401  (kept for parity with other views)

    if not items:
        return
    html_items = "".join(
        h.evidence_item(
            e.claim, e.category, e.impact, e.confidence, e.source_title, e.source_url,
            f"{e.published_at:%Y-%m-%d}" if e.published_at else None,
        )
        for e in items
    )
    st.html(f'<div class="rt-label" style="margin-top:.8rem">{h.esc(title)}</div>{html_items}')


def _page() -> None:
    from research.orchestrator import ResearchOrchestrator
    from research.presentation import RELATIONSHIP_LABELS, STANCE_LABELS, age_label, quant_rows
    from research.schemas import ResearchRequest

    st.html(h.lead(
        "Informational only. The research agent is read-only: it cannot place orders or change strategy, "
        "HMM, or risk settings, and it is not financial advice."
    ))
    try:
        config = _research_config()
    except Exception as exc:  # noqa: BLE001
        st.html(h.notice(f"Research configuration error: {exc}", "halt"))
        return
    if not config.enabled:
        st.html(h.notice("AI research is disabled (research.enabled: false in config/settings.yaml). "
                         "The rest of the dashboard is unaffected."))
        return

    symbols = load_settings().get("broker", {}).get("symbols", [])
    with st.form("research_form"):
        c1, c2, c3 = st.columns([1, 3, 1])
        picked = c1.selectbox("Ticker", symbols + ["Other..."], key="research_symbol")
        custom = c1.text_input("Custom ticker (if Other)", key="research_custom")
        question = c2.text_area("Research question (optional)", key="research_question", height=100)
        lookback = c3.number_input("Lookback (days)", 1, config.max_lookback_days, config.default_lookback_days, key="research_lookback")
        force = c3.checkbox("Force refresh", key="research_force")
        _usage_panel(config)
        submitted = st.form_submit_button("Research")

    if submitted:
        symbol = custom if picked == "Other..." else picked
        try:
            request = ResearchRequest(symbol=symbol, question=question or None, lookback_days=int(lookback), force_refresh=force)
        except ValueError as exc:
            st.html(h.notice(f"Invalid request: {exc}", "halt"))
            return
        with st.spinner(f"Collecting HMM context and researching with {config.provider} ({config.model or 'no model'})..."):
            st.session_state["research_outcome"] = ResearchOrchestrator(config).research(request)

    outcome = st.session_state.get("research_outcome")
    if outcome is None:
        st.html(h.notice("Choose a ticker and press Research. Results are cached; nothing runs on page refresh."))
        return

    # ---- A. quantitative context ----
    st.html(h.eyebrow("A") + h.section_title("Quantitative context"))
    if outcome.quant_context is not None:
        q = outcome.quant_context
        st.html(h.kpi_row([
            h.kpi("Regime", q.regime_label or "n/a", f"{(q.regime_probability or 0):.0%} confidence"),
            h.kpi("Expected vol (per bar)", "n/a" if q.expected_volatility is None else f"{q.expected_volatility:.2%}"),
            h.kpi("Stability", f"{q.consecutive_bars} bars", "confirmed" if q.is_confirmed else "UNCONFIRMED",
                  None if q.is_confirmed else "warn"),
            h.kpi("Flicker", "FLICKERING" if q.is_flickering else "stable", None, "warn" if q.is_flickering else None),
        ]))
        if q.state_probabilities:
            st.plotly_chart(charts.state_probabilities(q.state_probabilities), width="stretch", config=charts.PLOT_CONFIG)
        st.dataframe(pd.DataFrame(quant_rows(q), columns=["metric", "value"]), hide_index=True, width="stretch")
    else:
        st.html(h.notice("No HMM snapshot is available for this symbol.", "warn"))

    if not outcome.ok:
        st.html(h.notice(f"{outcome.error.message} ({outcome.error.code}). The quantitative system continues normally; "
                         "no orders were placed. Press Research to retry.", "halt", "Research temporarily unavailable"))
        return

    result = outcome.result
    brief, comparison, diag = result.brief, result.comparison, result.diagnostics

    # ---- B. current research ----
    st.html(h.eyebrow("B") + h.section_title("Current research"))
    st.html(
        f'<div class="rt-mono-muted">Generated {brief.generated_at:%Y-%m-%d %H:%M} UTC ({h.esc(age_label(brief.generated_at))}) · '
        f'{"served from CACHE" if result.cache_hit else "fresh request"} · {brief.source_count} sources</div>'
        f'<p class="rt-lead" style="margin-top:.7rem"><strong>{h.esc(brief.question)}</strong></p>'
        f'<p style="max-width:78ch;line-height:1.6;color:var(--rt-body)">{h.esc(brief.summary)}</p>'
        f'<div>{h.pill(STANCE_LABELS[brief.evidence_stance], "warn" if brief.evidence_stance == "insufficient_evidence" else "accent")}'
        f'<span class="rt-mono-muted">overall confidence {brief.overall_confidence:.0%}</span></div>'
    )
    if brief.evidence_stance == "insufficient_evidence":
        st.html(h.notice("Insufficient sourced evidence — treat this report as incomplete.", "warn"))
    _evidence_section("Catalysts", brief.catalysts)
    _evidence_section("Risks", brief.risks)
    _evidence_section("Recent material events", brief.recent_events)
    if brief.conflicting_evidence:
        st.html(f'<div class="rt-label" style="margin-top:.8rem">Conflicting evidence</div>{h.bullets(brief.conflicting_evidence)}')
    if brief.unknowns:
        st.html(f'<div class="rt-label" style="margin-top:.8rem">Unknowns</div>{h.bullets(brief.unknowns)}')

    # ---- C. quant vs research ----
    st.html(h.eyebrow("C") + h.section_title("Quant vs research"))
    st.html(
        f'<div>{h.pill(RELATIONSHIP_LABELS[comparison.relationship], "accent")}'
        f'{h.pill("human review recommended", "warn") if comparison.human_review_recommended else ""}</div>'
        f'<p style="max-width:78ch;line-height:1.6;color:var(--rt-body);margin-top:.7rem">{h.esc(comparison.explanation)}</p>'
    )
    left, right = st.columns(2)
    with left:
        for title, items in (("Agreements", comparison.agreements), ("Contradictions", comparison.contradictions)):
            st.html(f'<div class="rt-label">{h.esc(title)}</div>{h.bullets(items or ["(none)"])}')
    with right:
        for title, items in (
            ("What the HMM cannot directly see", comparison.information_quant_model_cannot_see),
            ("What research cannot establish", comparison.information_research_cannot_establish),
        ):
            st.html(f'<div class="rt-label">{h.esc(title)}</div>{h.bullets(items or ["(none)"])}')

    # ---- D. diagnostics ----
    with st.expander("D. Research trace / diagnostics"):
        u = diag.usage
        lines = [
            f"Cache: {'HIT' if result.cache_hit else 'MISS'} · latency {result.latency_ms} ms",
            f"Provider {diag.provider} · model {diag.model} · search {diag.search_provider} · prompt v{diag.prompt_version}",
            f"Search rounds {diag.search_rounds} · LLM calls {diag.llm_calls} · dropped unsupported claims {diag.dropped_unsupported_claims}",
        ]
        if result.cache_hit:
            lines.append(f"This request: free (served from cache; original run used {u.grounded_searches} grounded searches, ≈${u.estimated_cost_usd:.3f})")
        else:
            if u.finance_searches:
                lines.append(f"SerpApi Google Finance searches this run: {u.finance_searches}")
            lines.append(f"This run: {u.grounded_searches} grounded searches, {u.input_tokens:,} input / "
                         f"{u.output_tokens:,} output tokens, estimated ≈${u.estimated_cost_usd:.3f}")
        lines.append("Search queries: " + (", ".join(brief.search_queries) or "(none)"))
        lines.append("Source domains: " + (", ".join(diag.source_domains) or "(none)"))
        st.html(h.bullets(lines))
        st.markdown("**Internal tool calls**")
        st.dataframe(
            pd.DataFrame([t.model_dump() for t in diag.tool_calls]) if diag.tool_calls else pd.DataFrame(columns=["tool"]),
            hide_index=True, width="stretch",
        )
        for note in diag.notes:
            st.caption(note)


def render() -> None:
    st.html(h.eyebrow("03 / Research") + h.hero("What changed"))
    _page()
