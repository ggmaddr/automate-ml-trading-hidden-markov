"""Real-time and historical market data fetching."""

from __future__ import annotations

from typing import Optional

import pandas as pd

from broker.alpaca_client import AlpacaClient


class MarketDataFeed:
    """Fetches and caches OHLCV data for tracked symbols."""

    def __init__(self, client: AlpacaClient, config: dict) -> None:
        raise NotImplementedError

    def get_historical_bars(self, symbol: str, lookback_bars: int) -> pd.DataFrame:
        """Return the last `lookback_bars` OHLCV bars for a symbol."""
        raise NotImplementedError

    def get_latest_bar(self, symbol: str) -> Optional[pd.Series]:
        """Return the most recent completed bar for a symbol."""
        raise NotImplementedError

    def stream_bars(self, symbols: list[str]) -> None:
        """Subscribe to a live bar stream for the given symbols."""
        raise NotImplementedError

    def refresh_all(self) -> dict[str, pd.DataFrame]:
        """Refresh cached historical data for all configured symbols."""
        raise NotImplementedError
