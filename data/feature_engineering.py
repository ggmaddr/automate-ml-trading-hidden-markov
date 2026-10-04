"""Pure functions that turn raw OHLCV bars into HMM observation features.

Every function takes/returns plain pandas Series/DataFrames and has no side
effects, so they can be unit tested in isolation and reused by both the live
pipeline and the backtester.

Input DataFrames are expected to have lowercase columns:
``open, high, low, close, volume`` indexed by a monotonically increasing
timestamp.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import ADXIndicator
from ta.volatility import AverageTrueRange

# Column names of the raw (pre-standardization) feature set, in the order
# they are assembled. Kept explicit so callers/tests can validate shape.
RAW_FEATURE_COLUMNS = [
    "log_ret_1",
    "log_ret_5",
    "log_ret_20",
    "realized_vol_20",
    "vol_ratio_5_20",
    "volume_zscore_50",
    "volume_trend_slope_10",
    "adx_14",
    "sma50_slope",
    "rsi_14",
    "dist_from_sma200_pct",
    "roc_10",
    "roc_20",
    "atr_norm_14",
]


# ---------------------------------------------------------------------------
# Returns
# ---------------------------------------------------------------------------

def log_returns(close: pd.Series, periods: int) -> pd.Series:
    """Log return over `periods` bars: ln(close_t / close_{t-periods})."""
    return np.log(close / close.shift(periods))


# ---------------------------------------------------------------------------
# Volatility
# ---------------------------------------------------------------------------

def realized_volatility(close: pd.Series, window: int = 20) -> pd.Series:
    """Rolling std of 1-period log returns over `window` bars."""
    ret_1 = log_returns(close, 1)
    return ret_1.rolling(window).std(ddof=0)


def volatility_ratio(close: pd.Series, short_window: int = 5, long_window: int = 20) -> pd.Series:
    """Ratio of short-window realized vol to long-window realized vol."""
    short_vol = realized_volatility(close, short_window)
    long_vol = realized_volatility(close, long_window)
    return short_vol / long_vol.replace(0, np.nan)


# ---------------------------------------------------------------------------
# Volume
# ---------------------------------------------------------------------------

def volume_zscore(volume: pd.Series, window: int = 50) -> pd.Series:
    """Z-score of volume against its own rolling `window` mean/std."""
    roll_mean = volume.rolling(window).mean()
    roll_std = volume.rolling(window).std(ddof=0)
    return (volume - roll_mean) / roll_std.replace(0, np.nan)


def _linreg_slope(y: np.ndarray) -> float:
    """Closed-form OLS slope of y against an evenly spaced x = 0..n-1."""
    n = len(y)
    if n < 2 or np.any(np.isnan(y)):
        return np.nan
    x = np.arange(n, dtype=float)
    x_mean = x.mean()
    y_mean = y.mean()
    denom = ((x - x_mean) ** 2).sum()
    if denom == 0:
        return np.nan
    return float(((x - x_mean) * (y - y_mean)).sum() / denom)


def normalized_slope(series: pd.Series, sma_window: int, slope_window: int = 5) -> pd.Series:
    """Slope of an SMA(series, sma_window), normalized by the SMA level.

    Normalizing by level makes the slope comparable across symbols/price
    ranges (a $5/day slope means something different at $20 vs $500).
    """
    sma_series = series.rolling(sma_window).mean()
    raw_slope = sma_series.rolling(slope_window).apply(_linreg_slope, raw=True)
    return raw_slope / sma_series.replace(0, np.nan)


# ---------------------------------------------------------------------------
# Trend
# ---------------------------------------------------------------------------

def adx(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """Average Directional Index — trend strength, 0-100."""
    return ADXIndicator(high=high, low=low, close=close, window=window, fillna=False).adx()


# ---------------------------------------------------------------------------
# Mean reversion
# ---------------------------------------------------------------------------

def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Relative Strength Index, 0-100."""
    return RSIIndicator(close=close, window=window, fillna=False).rsi()


def distance_from_sma_pct(close: pd.Series, window: int = 200) -> pd.Series:
    """(close - SMA(window)) / close, as a fraction of price."""
    sma_series = close.rolling(window).mean()
    return (close - sma_series) / close


# ---------------------------------------------------------------------------
# Momentum
# ---------------------------------------------------------------------------

def roc(close: pd.Series, periods: int) -> pd.Series:
    """Rate of change over `periods` bars: (close_t - close_{t-p}) / close_{t-p}."""
    return close.pct_change(periods)


# ---------------------------------------------------------------------------
# Range
# ---------------------------------------------------------------------------

def normalized_atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """ATR(window) / close — range normalized to price level."""
    atr_series = AverageTrueRange(high=high, low=low, close=close, window=window, fillna=False).average_true_range()
    return atr_series / close


# ---------------------------------------------------------------------------
# Standardization
# ---------------------------------------------------------------------------

def rolling_zscore(series: pd.Series, window: int = 252) -> pd.Series:
    """Rolling z-score of a series against its own trailing `window` mean/std."""
    roll_mean = series.rolling(window).mean()
    roll_std = series.rolling(window).std(ddof=0)
    return (series - roll_mean) / roll_std.replace(0, np.nan)


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def compute_raw_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the raw (pre-standardization) feature set from OHLCV bars."""
    close, high, low, volume = df["close"], df["high"], df["low"], df["volume"]

    features = pd.DataFrame(index=df.index)
    features["log_ret_1"] = log_returns(close, 1)
    features["log_ret_5"] = log_returns(close, 5)
    features["log_ret_20"] = log_returns(close, 20)
    features["realized_vol_20"] = realized_volatility(close, 20)
    features["vol_ratio_5_20"] = volatility_ratio(close, 5, 20)
    features["volume_zscore_50"] = volume_zscore(volume, 50)
    features["volume_trend_slope_10"] = normalized_slope(volume, sma_window=10)
    features["adx_14"] = adx(high, low, close, 14)
    features["sma50_slope"] = normalized_slope(close, sma_window=50)
    features["rsi_14"] = rsi(close, 14)
    features["dist_from_sma200_pct"] = distance_from_sma_pct(close, 200)
    features["roc_10"] = roc(close, 10)
    features["roc_20"] = roc(close, 20)
    features["atr_norm_14"] = normalized_atr(high, low, close, 14)

    return features[RAW_FEATURE_COLUMNS]


def compute_features(df: pd.DataFrame, zscore_window: int = 252, dropna: bool = True) -> pd.DataFrame:
    """Compute the full standardized feature matrix used as HMM observations.

    All raw features are standardized with a rolling `zscore_window`-period
    z-score so they share a comparable scale for the Gaussian HMM.
    """
    raw = compute_raw_features(df)
    standardized = raw.apply(lambda col: rolling_zscore(col, zscore_window))
    if dropna:
        standardized = standardized.dropna()
    return standardized
