"""Stress testing: crash injection and gap simulation."""

from __future__ import annotations

import pandas as pd


def inject_crash(data: pd.DataFrame, start_idx: int, magnitude: float, duration_bars: int) -> pd.DataFrame:
    """Return a copy of data with a synthetic crash inserted starting at start_idx."""
    raise NotImplementedError


def inject_gap(data: pd.DataFrame, idx: int, gap_pct: float) -> pd.DataFrame:
    """Return a copy of data with a synthetic overnight gap inserted at idx."""
    raise NotImplementedError


def run_stress_suite(data: pd.DataFrame, backtester) -> dict:
    """Run a standard suite of crash/gap scenarios and report strategy behavior."""
    raise NotImplementedError
