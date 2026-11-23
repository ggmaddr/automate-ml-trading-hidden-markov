"""Local market-data helper for the research tools (read-only, no broker).

Uses yfinance — the same free source `main.py backtest` and `demo.py` use —
so the research layer never needs Alpaca credentials.
"""

from __future__ import annotations

import pandas as pd

from research.exceptions import ToolError


def fetch_daily_bars(symbol: str, period: str = "4y") -> pd.DataFrame:
    """Daily OHLCV with lowercase columns. ~4y leaves room for the 252-bar
    z-score warm-up plus a few hundred usable feature rows."""
    import yfinance as yf

    try:
        df = yf.download(symbol, period=period, interval="1d", progress=False, auto_adjust=True)
    except Exception as exc:  # noqa: BLE001 - network/vendor errors surface as a controlled tool error
        raise ToolError(f"market data download failed for {symbol}: {type(exc).__name__}") from exc

    if df is None or df.empty:
        raise ToolError(f"no market data returned for {symbol}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [str(c).lower() for c in df.columns]
    return df[["open", "high", "low", "close", "volume"]].dropna()
