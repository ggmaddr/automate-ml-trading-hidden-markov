"""Live terminal dashboard (rich), matching the Phase 8 layout spec.

`render()` is pure — it builds a Rich renderable from a plain dict of
current state, with no I/O, so it's fully unit-testable. `run()` wraps it
in rich.live.Live for a periodic auto-refresh and blocks forever; like the
streaming code in Phase 6/7, that loop itself isn't unit-testable, but
everything it calls (render(), the caller's state_provider) is tested
directly.
"""

from __future__ import annotations

import time
from typing import Callable, Optional

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


def _risk_bar(current: float, limit: float) -> Text:
    pct_of_limit = (current / limit) if limit else 0.0
    color = "green" if pct_of_limit < 0.5 else "yellow" if pct_of_limit < 0.9 else "red"
    mark = "OK" if pct_of_limit < 1.0 else "!!"
    return Text(f"{current:.1%}/{limit:.0%} {mark}", style=color)


class Dashboard:
    """Renders live regime, positions, P&L, and risk state to the terminal."""

    def __init__(self, config: Optional[dict] = None) -> None:
        self.config = config or {}
        self.console = Console()

    def render(self, state: dict) -> Panel:
        """Build the dashboard panel from a plain state dict. Pure, no I/O."""
        regime = state.get("regime", {})
        portfolio = state.get("portfolio", {})
        positions = state.get("positions", [])
        signals = state.get("recent_signals", [])
        risk = state.get("risk", {})
        system = state.get("system", {})

        regime_text = Text(
            f"{regime.get('label', '?')} ({regime.get('probability', 0):.0%}) | "
            f"Stability: {regime.get('stability_bars', 0)} bars | "
            f"Flicker: {regime.get('flicker_rate', 0)}/{regime.get('flicker_window', 20)}"
        )

        portfolio_text = Text(
            f"Equity: ${portfolio.get('equity', 0):,.0f} | "
            f"Daily: {portfolio.get('daily_pnl', 0):+,.0f} ({portfolio.get('daily_pnl_pct', 0):+.2%}) | "
            f"Allocation: {portfolio.get('allocation', 0):.0%} | Leverage: {portfolio.get('leverage', 1.0):.2f}x"
        )

        positions_table = Table.grid(padding=(0, 2))
        for col in ["Symbol", "Dir", "Price", "P&L", "Stop", "Held"]:
            positions_table.add_column(col)
        for p in positions:
            positions_table.add_row(
                p["symbol"], p.get("direction", "LONG"), f"${p['price']:,.2f}",
                f"{p['pnl_pct']:+.1%}", f"${p.get('stop', 0):,.2f}", str(p.get("held", "")),
            )
        if not positions:
            positions_table.add_row("(none)", "", "", "", "", "")

        signals_table = Table.grid(padding=(0, 2))
        for col in ["Time", "Symbol", "Action", "Reason"]:
            signals_table.add_column(col)
        for s in signals[-5:]:
            signals_table.add_row(s.get("time", ""), s.get("symbol", ""), s.get("action", ""), str(s.get("reason", "")))
        if not signals:
            signals_table.add_row("(none yet)", "", "", "")

        risk_text = Text.assemble(
            "Daily DD: ", _risk_bar(risk.get("daily_dd", 0), risk.get("daily_dd_limit", 0.03)),
            " | From Peak: ", _risk_bar(risk.get("peak_dd", 0), risk.get("peak_dd_limit", 0.10)),
        )

        system_text = Text(
            f"Data: {'OK' if system.get('data_ok') else 'DOWN'} | "
            f"API: {'OK' if system.get('api_ok') else 'DOWN'} {system.get('api_latency_ms', 0)}ms | "
            f"HMM: {system.get('hmm_age', '?')} | {'PAPER' if system.get('paper', True) else 'LIVE'}"
        )

        body = Group(
            Panel(regime_text, title="REGIME"),
            Panel(portfolio_text, title="PORTFOLIO"),
            Panel(positions_table, title="POSITIONS"),
            Panel(signals_table, title="RECENT SIGNALS"),
            Panel(risk_text, title="RISK STATUS"),
            Panel(system_text, title="SYSTEM"),
        )
        return Panel(body, title="regime-trader")

    def render_to_text(self, state: dict, width: int = 100) -> str:
        """Render to a plain string (useful for tests and non-live snapshots)."""
        console = Console(record=True, width=width)
        console.print(self.render(state))
        return console.export_text()

    def run(self, state_provider: Callable[[], dict], refresh_seconds: Optional[float] = None) -> None:
        """Blocking — live-refreshes the dashboard. Call from its own thread/process."""
        refresh_seconds = refresh_seconds or self.config.get("dashboard_refresh_seconds", 5)
        with Live(self.render(state_provider()), console=self.console, refresh_per_second=1) as live:
            while True:
                time.sleep(refresh_seconds)
                live.update(self.render(state_provider()))
