"""Tests for data.market_data. Uses a mocked StockHistoricalDataClient — no
real network access required.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import pytest

from data.market_data import MarketDataFeed, _parse_timeframe
from alpaca.data.timeframe import TimeFrameUnit


def _make_feed(hist_client) -> MarketDataFeed:
    return MarketDataFeed(api_key="k", secret_key="s", historical_client=hist_client)


# ---------------------------------------------------------------------------
# _parse_timeframe
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw,amount,unit", [
    ("1Day", 1, TimeFrameUnit.Day),
    ("5Min", 5, TimeFrameUnit.Minute),
    ("1Hour", 1, TimeFrameUnit.Hour),
])
def test_parse_timeframe(raw, amount, unit):
    tf = _parse_timeframe(raw)
    assert tf.amount == amount
    assert tf.unit == unit


def test_parse_timeframe_rejects_unknown_unit():
    with pytest.raises(ValueError):
        _parse_timeframe("1Fortnight")


# ---------------------------------------------------------------------------
# get_historical_bars
# ---------------------------------------------------------------------------

def test_get_historical_bars_slices_multiindex_and_lowercases():
    dates = pd.bdate_range("2024-01-01", periods=3)
    idx = pd.MultiIndex.from_product([["AAPL"], dates], names=["symbol", "timestamp"])
    df = pd.DataFrame(
        {"open": [1, 2, 3], "high": [1, 2, 3], "low": [1, 2, 3], "close": [1, 2, 3],
         "volume": [100, 200, 300], "trade_count": [1, 1, 1], "vwap": [1.0, 2.0, 3.0]},
        index=idx,
    )
    hist_client = MagicMock()
    hist_client.get_stock_bars.return_value = SimpleNamespace(df=df)

    feed = _make_feed(hist_client)
    result = feed.get_historical_bars("AAPL", timeframe="1Day")

    assert list(result.columns) == ["open", "high", "low", "close", "volume"]
    assert len(result) == 3
    assert not isinstance(result.index, pd.MultiIndex)


def test_get_historical_bars_handles_empty_result():
    hist_client = MagicMock()
    hist_client.get_stock_bars.return_value = SimpleNamespace(df=pd.DataFrame())

    feed = _make_feed(hist_client)
    result = feed.get_historical_bars("AAPL")

    assert result.empty


# ---------------------------------------------------------------------------
# get_latest_bar / get_latest_quote / get_snapshot
# ---------------------------------------------------------------------------

def test_get_latest_bar_returns_dict():
    hist_client = MagicMock()
    hist_client.get_stock_latest_bar.return_value = {
        "AAPL": SimpleNamespace(symbol="AAPL", timestamp="t", open=1, high=2, low=0.5, close=1.5, volume=1000)
    }
    feed = _make_feed(hist_client)
    bar = feed.get_latest_bar("AAPL")
    assert bar["close"] == 1.5


def test_get_latest_bar_returns_none_when_missing():
    hist_client = MagicMock()
    hist_client.get_stock_latest_bar.return_value = {}
    feed = _make_feed(hist_client)
    assert feed.get_latest_bar("AAPL") is None


def test_get_latest_quote_computes_spread_pct():
    hist_client = MagicMock()
    hist_client.get_stock_latest_quote.return_value = {
        "AAPL": SimpleNamespace(
            symbol="AAPL", timestamp="t", bid_price=99.0, ask_price=100.0, bid_size=10, ask_size=10,
        )
    }
    feed = _make_feed(hist_client)
    quote = feed.get_latest_quote("AAPL")
    assert quote["spread_pct"] == pytest.approx(0.01)


def test_get_snapshot_returns_dict():
    hist_client = MagicMock()
    hist_client.get_stock_snapshot.return_value = {
        "AAPL": SimpleNamespace(
            symbol="AAPL", latest_trade="trade", latest_quote="quote",
            minute_bar="mbar", daily_bar="dbar", previous_daily_bar="pbar",
        )
    }
    feed = _make_feed(hist_client)
    snapshot = feed.get_snapshot("AAPL")
    assert snapshot["daily_bar"] == "dbar"


def test_refresh_all_fetches_each_symbol():
    dates = pd.bdate_range("2024-01-01", periods=2)

    def _bars_for(request) -> SimpleNamespace:
        symbol = request.symbol_or_symbols
        idx = pd.MultiIndex.from_product([[symbol], dates], names=["symbol", "timestamp"])
        df = pd.DataFrame(
            {"open": [1, 2], "high": [1, 2], "low": [1, 2], "close": [1, 2], "volume": [1, 2]}, index=idx,
        )
        return SimpleNamespace(df=df)

    hist_client = MagicMock()
    hist_client.get_stock_bars.side_effect = _bars_for

    feed = _make_feed(hist_client)
    result = feed.refresh_all(["AAPL", "MSFT"])

    assert set(result.keys()) == {"AAPL", "MSFT"}
    assert hist_client.get_stock_bars.call_count == 2
