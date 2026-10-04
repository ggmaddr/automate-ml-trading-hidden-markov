"""Tests for monitoring.dashboard. render()/render_to_text() are pure, so
these are plain unit tests — no live terminal or refresh loop involved.
"""

from __future__ import annotations

from rich.panel import Panel

from monitoring.dashboard import Dashboard


def _sample_state() -> dict:
    return {
        "regime": {"label": "BULL", "probability": 0.72, "stability_bars": 14, "flicker_rate": 1, "flicker_window": 20},
        "portfolio": {"equity": 105230, "daily_pnl": 340, "daily_pnl_pct": 0.0032, "allocation": 0.95, "leverage": 1.25},
        "positions": [
            {"symbol": "SPY", "direction": "LONG", "price": 520.30, "pnl_pct": 0.012, "stop": 508.0, "held": "3:00:00"}
        ],
        "recent_signals": [{"time": "14:30", "symbol": "SPY", "action": "rebalance", "reason": "60% -> 95%, low vol"}],
        "risk": {"daily_dd": 0.003, "daily_dd_limit": 0.03, "peak_dd": 0.012, "peak_dd_limit": 0.10},
        "system": {"data_ok": True, "api_ok": True, "api_latency_ms": 23, "hmm_age": "2d ago", "paper": True},
    }


def test_render_returns_a_panel():
    assert isinstance(Dashboard().render(_sample_state()), Panel)


def test_render_to_text_contains_key_values():
    text = Dashboard().render_to_text(_sample_state())
    assert "BULL" in text
    assert "SPY" in text
    assert "105,230" in text
    assert "PAPER" in text


def test_render_handles_completely_empty_state():
    text = Dashboard().render_to_text({})  # must not raise on missing keys
    assert "none" in text.lower()


def test_render_shows_placeholder_with_no_positions_or_signals():
    state = _sample_state()
    state["positions"] = []
    state["recent_signals"] = []
    text = Dashboard().render_to_text(state)
    assert "none" in text.lower()


def test_render_shows_live_mode_when_not_paper():
    state = _sample_state()
    state["system"]["paper"] = False
    text = Dashboard().render_to_text(state)
    assert "LIVE" in text
