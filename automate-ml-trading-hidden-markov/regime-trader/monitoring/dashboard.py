"""Terminal-based live dashboard (built on `rich`)."""

from __future__ import annotations


class Dashboard:
    """Renders live regime, positions, P&L, and risk state to the terminal."""

    def __init__(self, config: dict) -> None:
        raise NotImplementedError

    def render(self, state: dict) -> None:
        """Render one frame of the dashboard from the current bot state."""
        raise NotImplementedError

    def run(self) -> None:
        """Start the refresh loop at dashboard_refresh_seconds interval."""
        raise NotImplementedError
