"""Shared helpers for the dashboard views."""

from __future__ import annotations

import streamlit as st
import yaml

DASHBOARD_JSON_PATH = "dashboard_state.json"
STATE_SNAPSHOT_PATH = "state_snapshot.json"
BACKTEST_DIR = "backtest_results"


@st.cache_data(ttl=60)
def load_settings() -> dict:
    with open("config/settings.yaml") as f:
        return yaml.safe_load(f)
