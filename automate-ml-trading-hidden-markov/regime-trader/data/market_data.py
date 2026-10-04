"""Real-time and historical market data via Alpaca's market data API.

Historical/latest reads use StockHistoricalDataClient (plain REST). Live
bars/quotes use StockDataStream (WebSocket). As with broker.position_tracker,
`subscribe_*` methods only register callbacks; `run_stream()` blocks and is
left to the caller to run in its own thread/task.

Market data access doesn't depend on paper vs. live trading — the same
market data is returned either way — so this module only needs API
key/secret, not the `paper` flag.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Callable, Optional

import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import (
    StockBarsRequest,
    StockLatestBarRequest,
    StockLatestQuoteRequest,
    StockSnapshotRequest,
)
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

from broker.alpaca_client import load_alpaca_credentials

logger = logging.getLogger(__name__)

_TIMEFRAME_UNITS = {
    "min": TimeFrameUnit.Minute, "minute": TimeFrameUnit.Minute,
    "hour": TimeFrameUnit.Hour,
    "day": TimeFrameUnit.Day,
}


def _parse_timeframe(timeframe: str) -> TimeFrame:
    """Parse strings like '1Min', '5Min', '1Hour', '1Day' into a TimeFrame."""
    digits = "".join(c for c in timeframe if c.isdigit()) or "1"
    unit_str = "".join(c for c in timeframe if c.isalpha()).lower()
    if unit_str not in _TIMEFRAME_UNITS:
        raise ValueError(f"Unrecognized timeframe unit in '{timeframe}' (expected Min/Hour/Day)")
    return TimeFrame(int(digits), _TIMEFRAME_UNITS[unit_str])


_TRADING_HOURS_PER_DAY = 6.5
_DEFAULT_LOOKBACK_LIMIT = 2000  # used only to size the lookback window when the caller gives neither start nor limit


def _default_lookback_start(tf: TimeFrame, limit: Optional[int], end: Optional[datetime]) -> datetime:
    """Alpaca's bars endpoint needs an explicit `start` -- a request with only
    `limit` and no start/end returns ZERO bars, not "however many fit" (this
    is what broke get_historical_bars(symbol, limit=2000) with no start/end:
    it silently got back an empty, columnless DataFrame).

    This picks a `start` far enough back to comfortably cover `limit` bars
    for the given timeframe, padding 2x for weekends/holidays/non-trading
    hours. get_historical_bars then trims the (wider) fetched window down to
    exactly the most recent `limit` bars with .tail().
    """
    end = end or datetime.now(timezone.utc)
    limit = limit or _DEFAULT_LOOKBACK_LIMIT
    if tf.unit == TimeFrameUnit.Day:
        calendar_days = limit * 2 + 15  # ~252 trading days per 365 calendar days
    elif tf.unit == TimeFrameUnit.Hour:
        calendar_days = (limit * tf.amount / _TRADING_HOURS_PER_DAY) * 2 + 15
    else:
        calendar_days = (limit * tf.amount / (_TRADING_HOURS_PER_DAY * 60)) * 2 + 15
    return end - timedelta(days=calendar_days)


class MarketDataFeed:
    """Fetches historical/latest bars and quotes, and manages the live data stream."""

    def __init__(
        self, api_key: Optional[str] = None, secret_key: Optional[str] = None,
        historical_client: Optional[StockHistoricalDataClient] = None,
    ) -> None:
        api_key, secret_key, _ = load_alpaca_credentials(api_key, secret_key, paper=True)
        self._api_key = api_key
        self._secret_key = secret_key
        self._hist_client = historical_client or StockHistoricalDataClient(api_key, secret_key)
        self._stream = None  # alpaca.data.live.StockDataStream, set by subscribe_bars/quotes

    # ------------------------------------------------------------------
    # Historical / latest reads
    # ------------------------------------------------------------------

    def get_historical_bars(
        self, symbol: str, timeframe: str = "1Day", start=None, end=None, limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """Return the most recent `limit` OHLCV bars as a DataFrame indexed by
        timestamp, lowercase columns (or every bar in [start, end] if `limit`
        is None).

        Gaps from weekends/holidays/halts simply don't appear as rows —
        downstream rolling-window code (data.feature_engineering) operates
        on bar COUNT, not calendar time, so this is handled gracefully by
        construction rather than needing special-case gap-filling.
        """
        tf = _parse_timeframe(timeframe)
        if start is None:
            start = _default_lookback_start(tf, limit, end)

        # `limit` isn't passed to Alpaca here: its API applies `limit` by
        # truncating from the OLDEST end of [start, end], not the newest --
        # the opposite of "give me the most recent N bars" -- so the full
        # window is fetched and trimmed to the latest `limit` bars below.
        request = StockBarsRequest(symbol_or_symbols=symbol, timeframe=tf, start=start, end=end)
        bar_set = self._hist_client.get_stock_bars(request)
        df = bar_set.df
        if df.empty:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        if isinstance(df.index, pd.MultiIndex):
            df = df.xs(symbol, level=0)
        df = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
        return df.tail(limit) if limit else df

    def get_latest_bar(self, symbol: str) -> Optional[dict]:
        request = StockLatestBarRequest(symbol_or_symbols=symbol)
        result = self._hist_client.get_stock_latest_bar(request)
        bar = result.get(symbol)
        if bar is None:
            return None
        return {
            "symbol": bar.symbol, "timestamp": bar.timestamp, "open": bar.open, "high": bar.high,
            "low": bar.low, "close": bar.close, "volume": bar.volume,
        }

    def get_latest_quote(self, symbol: str) -> Optional[dict]:
        request = StockLatestQuoteRequest(symbol_or_symbols=symbol)
        result = self._hist_client.get_stock_latest_quote(request)
        quote = result.get(symbol)
        if quote is None:
            return None
        spread_pct = (
            (quote.ask_price - quote.bid_price) / quote.ask_price
            if quote.ask_price else 0.0
        )
        return {
            "symbol": quote.symbol, "timestamp": quote.timestamp, "bid_price": quote.bid_price,
            "ask_price": quote.ask_price, "bid_size": quote.bid_size, "ask_size": quote.ask_size,
            "spread_pct": spread_pct,
        }

    def get_snapshot(self, symbol: str) -> Optional[dict]:
        request = StockSnapshotRequest(symbol_or_symbols=symbol)
        result = self._hist_client.get_stock_snapshot(request)
        snapshot = result.get(symbol)
        if snapshot is None:
            return None
        return {
            "symbol": snapshot.symbol,
            "latest_trade": snapshot.latest_trade,
            "latest_quote": snapshot.latest_quote,
            "minute_bar": snapshot.minute_bar,
            "daily_bar": snapshot.daily_bar,
            "previous_daily_bar": snapshot.previous_daily_bar,
        }

    def refresh_all(self, symbols: list[str], timeframe: str = "1Day", limit: int = 1000) -> dict[str, pd.DataFrame]:
        """Refresh cached historical data for a list of symbols."""
        return {symbol: self.get_historical_bars(symbol, timeframe=timeframe, limit=limit) for symbol in symbols}

    # ------------------------------------------------------------------
    # Live streaming (construction only — caller runs the event loop)
    # ------------------------------------------------------------------

    def _ensure_stream(self):
        if self._stream is None:
            from alpaca.data.live import StockDataStream  # imported lazily: not needed outside live trading
            self._stream = StockDataStream(self._api_key, self._secret_key)
        return self._stream

    def subscribe_bars(self, symbols: list[str], callback: Callable, timeframe: str = "1Min") -> None:
        """Register `callback` for live bar updates. Does not block — see run_stream()."""
        stream = self._ensure_stream()
        stream.subscribe_bars(callback, *symbols)

    def subscribe_quotes(self, symbols: list[str], callback: Callable) -> None:
        """Register `callback` for live quote updates (e.g. for spread checks). Does not block."""
        stream = self._ensure_stream()
        stream.subscribe_quotes(callback, *symbols)

    def run_stream(self) -> None:
        """Blocking — runs the subscribed WebSocket event loop. Call from its own thread/task."""
        if self._stream is not None:
            self._stream.run()

    def stop_stream(self) -> None:
        if self._stream is not None:
            self._stream.stop()
