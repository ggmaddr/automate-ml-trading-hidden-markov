"""Live view: polls dashboard_state.json written by `python main.py run --web-dashboard`.

There is no IPC into an already-running bot, so this is a separate process
reading a file the bot keeps overwriting (see the README's dashboard FAQ).
The body is a Streamlit fragment, so auto-refresh re-runs only this section
instead of the whole page.
"""

from __future__ import annotations

import json
import os
import time

import pandas as pd
import streamlit as st

from ui import html as h
from ui import svg
from views.common import DASHBOARD_JSON_PATH, STATE_SNAPSHOT_PATH, load_settings


def _research_health_html() -> str:
    """AI research health card for the system panel. Never raises (the sidecar must not break the live view)."""
    try:
        from research.config import load_research_config
        from research.health import get_research_health

        config = load_research_config(load_settings())
        health = get_research_health(config)
        tone = {"healthy": "ok", "unavailable": "halt", "disabled": "neutral"}[health.status]
        rows = [f'{h.pill(health.status, tone)} <span class="rt-mono-muted">{h.esc(config.provider)} · {h.esc(health.detail)}</span>']
        if health.last_success_at:
            rows.append(f'<div class="rt-mono-muted">last success {health.last_success_at:%Y-%m-%d %H:%M} UTC</div>')
        if health.cache_hit_rate is not None:
            rows.append(f'<div class="rt-mono-muted">cache hit rate {health.cache_hit_rate:.0%}</div>')
        return "".join(rows)
    except Exception as exc:  # noqa: BLE001
        return f'{h.pill("unavailable", "halt")} <span class="rt-mono-muted">{h.esc(type(exc).__name__)}</span>'


def _status_pill(ok: bool, up: str = "OK", down: str = "DOWN") -> str:
    return h.pill(up if ok else down, "ok" if ok else "halt")


def _body() -> None:
    settings = load_settings()
    risk_cfg = settings.get("risk", {})

    if not os.path.exists(DASHBOARD_JSON_PATH):
        st.info(
            f"No `{DASHBOARD_JSON_PATH}` found. Start the bot with `python main.py run --web-dashboard` "
            "(add `--dry-run` to watch without placing orders)."
        )
        if os.path.exists(STATE_SNAPSHOT_PATH):
            with open(STATE_SNAPSHOT_PATH) as f:
                snapshot = json.load(f)
            st.caption(f"Last saved state_snapshot.json (from a prior shutdown/error), saved at {snapshot.get('saved_at')}:")
            st.json(snapshot)
        return

    age_seconds = time.time() - os.path.getmtime(DASHBOARD_JSON_PATH)
    with open(DASHBOARD_JSON_PATH) as f:
        state = json.load(f)

    refresh_seconds = settings.get("monitoring", {}).get("dashboard_refresh_seconds", 5)
    stale = age_seconds > max(refresh_seconds * 3, 15)
    st.html(
        h.pill(f"snapshot {age_seconds:.0f}s old — is the bot still running?" if stale else f"updated {age_seconds:.0f}s ago",
               "halt" if stale else "ok")
    )

    regime, portfolio, system = state.get("regime", {}), state.get("portfolio", {}), state.get("system", {})
    risk = state.get("risk", {})

    st.html(h.kpi_row([
        h.kpi("Regime", regime.get("label", "?"), f"{regime.get('probability', 0):.0%} confidence"),
        h.kpi("Stability", f"{regime.get('stability_bars', 0)} bars"),
        h.kpi("Flicker", f"{regime.get('flicker_rate', 0)}/{regime.get('flicker_window', 20)}"),
        h.kpi("Equity", f"${portfolio.get('equity', 0):,.0f}", f"{portfolio.get('daily_pnl', 0):+,.0f} today"),
        h.kpi("Allocation", f"{portfolio.get('allocation', 0):.0%}"),
        h.kpi("Leverage", f"{portfolio.get('leverage', 1.0):.2f}x"),
    ]))

    # ---- risk gauges (needle = share of each limit used) --------------------------
    st.html(h.eyebrow("Risk status") + h.section_title("Choose your limits."))
    daily_limit = risk.get("daily_dd_limit", risk_cfg.get("daily_dd_halt", 0.03))
    peak_limit = risk.get("peak_dd_limit", risk_cfg.get("max_dd_from_peak", 0.10))
    exposure_limit = risk_cfg.get("max_exposure", 0.80)
    leverage_limit = risk_cfg.get("max_leverage", 1.25)
    daily, peak = risk.get("daily_dd", 0), risk.get("peak_dd", 0)
    allocation, leverage = portfolio.get("allocation", 0), portfolio.get("leverage", 1.0)
    st.html(svg.gauge_row([
        svg.gauge(svg.gauge_fraction(daily, daily_limit), "Daily drawdown", f"{daily:.1%} of {daily_limit:.0%}", "halts trading at the limit", "shield"),
        svg.gauge(svg.gauge_fraction(peak, peak_limit), "From peak", f"{peak:.1%} of {peak_limit:.0%}", "full halt at the limit", "chart"),
        svg.gauge(svg.gauge_fraction(allocation, exposure_limit), "Total exposure", f"{allocation:.0%} of {exposure_limit:.0%}", "maximum invested", "layers"),
        svg.gauge(svg.gauge_fraction(leverage, leverage_limit), "Borrowing", f"{leverage:.2f}x of {leverage_limit:.2f}x", "leverage ceiling", "coin"),
    ]))

    # ---- positions -------------------------------------------------------------------
    st.html(h.section_title("Positions"))
    positions = state.get("positions", [])
    if positions:
        df = pd.DataFrame(positions).rename(
            columns={"symbol": "Symbol", "direction": "Dir", "price": "Price", "pnl_pct": "P&L %", "stop": "Stop", "held": "Held"}
        )
        st.dataframe(df, hide_index=True, width="stretch")
    else:
        st.info("No open positions.")

    left, right = st.columns(2)
    with left:
        st.html(h.section_title("System"))
        st.html(h.card(
            f'<div>Data feed {_status_pill(bool(system.get("data_ok")))}</div>'
            f'<div style="margin-top:.5rem">Broker API {_status_pill(bool(system.get("api_ok")))} '
            f'<span class="rt-mono-muted">{h.esc(system.get("api_latency_ms", 0))} ms</span></div>'
            f'<div style="margin-top:.5rem"><span class="rt-mono-muted">HMM last trained</span> {h.esc(system.get("hmm_age", "?"))}</div>'
            f'<div style="margin-top:.5rem">Mode {h.pill("PAPER", "accent") if system.get("paper", True) else h.pill("LIVE", "halt")}</div>'
            f'<div style="margin-top:.7rem" class="rt-label">AI research</div>{_research_health_html()}'
        ))
    with right:
        st.html(h.section_title("Recent signals"))
        signals = list(reversed(state.get("recent_signals", [])))
        if signals:
            st.dataframe(
                pd.DataFrame(signals).rename(columns={"time": "Time", "symbol": "Symbol", "action": "Action", "reason": "Reason"}),
                hide_index=True, width="stretch",
            )
        else:
            st.info("No signals yet.")


def render() -> None:
    st.html(h.eyebrow("01 / Live") + h.hero("Regime + risk"))
    auto = st.toggle("Auto-refresh", value=True, key="auto_refresh")
    refresh = load_settings().get("monitoring", {}).get("dashboard_refresh_seconds", 5)
    st.fragment(run_every=refresh if auto else None)(_body)()
