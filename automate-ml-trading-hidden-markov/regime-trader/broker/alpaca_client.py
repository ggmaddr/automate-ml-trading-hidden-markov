"""Thin wrapper around the Alpaca trading + market data APIs."""

from __future__ import annotations

from typing import Optional

import pandas as pd


class AlpacaClient:
    """Wraps alpaca-py trading and data clients behind a single interface."""

    def __init__(self, api_key: str, secret_key: str, paper: bool = True) -> None:
        raise NotImplementedError

    def get_account(self) -> dict:
        """Return account equity, buying power, and status."""
        raise NotImplementedError

    def get_bars(self, symbol: str, timeframe: str, start: str, end: Optional[str] = None) -> pd.DataFrame:
        """Fetch historical OHLCV bars for a symbol."""
        raise NotImplementedError

    def get_latest_quote(self, symbol: str) -> dict:
        """Fetch the latest quote for a symbol."""
        raise NotImplementedError

    def get_positions(self) -> list[dict]:
        """Return all open positions."""
        raise NotImplementedError

    def is_market_open(self) -> bool:
        """True if the market is currently open."""
        raise NotImplementedError
