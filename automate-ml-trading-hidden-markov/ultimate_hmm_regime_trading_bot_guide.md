# Claude Code Prompt: HMM Regime-Based Trading Bot — Final

**How to use:** Copy each phase into Claude Code one at a time. Complete and test each phase before moving to the next.

---

## PHASE 1: Project Scaffolding & Environment Setup

Create a Python project called "regime-trader" with the following structure:

```text
regime-trader/
├── config/
│   ├── settings.yaml          # All configurable parameters
│   └── credentials.yaml.example
├── core/
│   ├── __init__.py
│   ├── hmm_engine.py          # HMM regime detection engine
│   ├── regime_strategies.py   # Vol-based allocation strategies
│   ├── risk_manager.py        # Position sizing, leverage, drawdown limits
│   └── signal_generator.py    # Combines HMM + strategy into signals
├── broker/
│   ├── __init__.py
│   ├── alpaca_client.py       # Alpaca API wrapper
│   ├── order_executor.py      # Order placement, modification, cancellation
│   └── position_tracker.py    # Track open positions, P&L
├── data/
│   ├── __init__.py
│   ├── market_data.py         # Real-time and historical data fetching
│   └── feature_engineering.py # Technical indicators, feature computation
├── monitoring/
│   ├── __init__.py
│   ├── logger.py              # Structured logging
│   ├── dashboard.py           # Terminal-based live dashboard
│   └── alerts.py              # Email/webhook alerts for critical events
├── backtest/
│   ├── __init__.py
│   ├── backtester.py          # Walk-forward allocation backtester
│   ├── performance.py         # Sharpe, drawdown, regime breakdown, benchmarks
│   └── stress_test.py         # Crash injection, gap simulation
├── tests/
│   ├── test_hmm.py
│   ├── test_look_ahead.py     # Verify no look-ahead bias
│   ├── test_strategies.py
│   ├── test_risk.py
│   └── test_orders.py
├── main.py                    # Entry point
├── requirements.txt
├── .env.example
└── README.md
```

Set up requirements.txt with:
- hmmlearn
- alpaca-trade-api
- alpaca-py
- pandas, numpy, scipy
- ta (technical analysis library)
- scikit-learn
- pyyaml
- python-dotenv
- websocket-client
- schedule
- rich (for terminal dashboard)

Create settings.yaml with ALL parameters, grouped by section, with defaults and comments:
- broker (paper_trading: true, symbols: [SPY, QQQ, AAPL, MSFT, AMZN, GOOGL, NVDA, META, TSLA, AMD], timeframe: 1Day)
- hmm (n_candidates: [3, 4, 5, 6, 7], n_init: 10, covariance_type: full, min_train_bars: 252, stability_bars: 3, flicker_window: 20, flicker_threshold: 4, min_confidence: 0.55)
- strategy (low_vol_allocation: 0.95, mid_vol_allocation_trend: 0.95, mid_vol_allocation_no_trend: 0.60, high_vol_allocation: 0.60, low_vol_leverage: 1.25, rebalance_threshold: 0.10, uncertainty_size_mult: 0.50)
- risk (max_risk_per_trade: 0.01, max_exposure: 0.80, max_leverage: 1.25, max_single_position: 0.15, max_concurrent: 5, max_daily_trades: 20, daily_dd_reduce: 0.02, daily_dd_halt: 0.03, weekly_dd_reduce: 0.05, weekly_dd_halt: 0.07, max_dd_from_peak: 0.10)
- backtest (slippage_pct: 0.0005, initial_capital: 100000, train_window: 252, test_window: 126, step_size: 126, risk_free_rate: 0.045)
- monitoring (dashboard_refresh_seconds: 5, alert_rate_limit_minutes: 15)

Create .env.example with:

```text
ALPACA_API_KEY=your_key_here
ALPACA_SECRET_KEY=your_secret_here
ALPACA_PAPER=true
```

Do NOT implement any logic yet — just the skeleton with imports, class stubs, type hints, and docstrings.

Add .env and credentials.yaml to .gitignore.

---

## PHASE 2: HMM Regime Detection Engine

Implement core/hmm_engine.py and data/feature_engineering.py.

DESIGN PHILOSOPHY: The HMM is a VOLATILITY CLASSIFIER. It detects whether the market is in a calm, moderate, or turbulent volatility environment. It does NOT predict price direction. The strategy layer uses the volatility classification to set portfolio allocation — be fully invested when conditions are calm, reduce when turbulent.

REQUIREMENTS:

1. GAUSSIAN HMM WITH AUTOMATIC MODEL SELECTION:
   - Test n_components = [3, 4, 5, 6, 7] during training
   - For each candidate, train and compute BIC (Bayesian Information Criterion)
   - BIC = -2 * log_likelihood + n_params * log(n_samples)
   - Select lowest BIC score (simplest model that explains the data)
   - Run multiple random initializations per candidate (n_init=10)
   - Log ALL candidate BIC scores and which was selected

After training, sort regimes by mean return (ascending) for LABELING:
- Lowest return → CRASH / BEAR
- Highest return → BULL / EUPHORIA
- Assign labels based on selected count:
  3 regimes: BEAR, NEUTRAL, BULL
  4 regimes: CRASH, BEAR, BULL, EUPHORIA
  5 regimes: CRASH, BEAR, NEUTRAL, BULL, EUPHORIA
  6 regimes: CRASH, STRONG_BEAR, WEAK_BEAR, WEAK_BULL, STRONG_BULL, EUPHORIA
  7 regimes: CRASH, STRONG_BEAR, WEAK_BEAR, NEUTRAL, WEAK_BULL, STRONG_BULL, EUPHORIA

IMPORTANT: Labels are sorted by return for human readability. But the STRATEGY layer sorts by VOLATILITY independently. The labels don't drive strategy decisions.

2. OBSERVABLE FEATURES (inputs to HMM):
   Implement in data/feature_engineering.py as pure functions.

Compute from OHLCV:

- Returns: log returns over 1, 5, 20 periods
- Volatility: realized vol (20-period rolling std), vol ratio (5-period / 20-period)
- Volume: normalized volume (z-score vs 50-period mean), volume trend (slope of 10-period SMA)
- Trend: ADX (14-period), slope of 50-period SMA
- Mean reversion: RSI(14) z-score, distance from 200 SMA as % of price
- Momentum: ROC 10 and 20 period
- Range: normalized ATR (14-period ATR / close)

Standardize ALL features with rolling z-scores (252-period lookback).

3. MODEL TRAINING:
- hmmlearn.GaussianHMM, covariance_type="full"
- Minimum 2 years daily data (504 trading days)
- Expanding window retraining: retrain at configurable intervals
- Store model with pickle + metadata (n_regimes, bic, training_date, labels)
- Log: likelihood, BIC, convergence, iterations

4. REGIME DETECTION — NO LOOK-AHEAD BIAS:

*** THIS IS THE MOST IMPORTANT TECHNICAL DETAIL. ***

DO NOT use model.predict(). predict() runs the Viterbi algorithm which processes the ENTIRE sequence and revises past states using future data. This is look-ahead bias that makes backtests unrealistically good.

INSTEAD implement FORWARD ALGORITHM ONLY (filtered inference):

```python
def predict_regime_filtered(self, features_up_to_now):
    """
    Compute P(state_t | observations_1:t) using forward algorithm.
    Uses ONLY past and present data. No future data.
    """
    # Use model's startprob_, transmat_, means_, covars_
    # Implement forward pass manually:
    # 1. alpha_0 = startprob * emission_prob(obs_0)
    # 2. alpha_t = (alpha_{t-1} @ transmat) * emission_prob(obs_t)
    # 3. Normalize at each step (work in log space)
    # 4. alpha_T = filtered distribution at current time
    # Cache previous alpha for efficiency in live/backtest loop
```

MANDATORY TEST — tests/test_look_ahead.py:

```python
def test_no_look_ahead_bias():
    """Regime at T must be identical with data[0:T] vs data[0:T+100]."""
    model = train_hmm(full_data)
    regime_short = predict_regime_filtered(data[0:400])[-1]
    regime_long = predict_regime_filtered(data[0:500])[400]
    assert regime_short == regime_long, "LOOK-AHEAD BIAS DETECTED"
```

5. REGIME STABILITY FILTER:
- Regime change only "confirmed" after persisting N bars (default 3)
- During transition: keep previous regime, reduce sizes by 25%
- Track flicker rate (changes per 20 bars)
- If flicker rate > threshold (default 4): force uncertainty mode

6. ADDITIONAL METHODS:
- predict_regime_proba() -> probability distribution
- get_regime_stability() -> consecutive bars in current regime
- get_transition_matrix() -> learned transition probabilities
- detect_regime_change() -> True only if confirmed
- get_regime_flicker_rate() -> changes per window
- is_flickering() -> True if flicker rate exceeds threshold

7. REGIME METADATA:

RegimeInfo dataclass:
- regime_id, regime_name, expected_return, expected_volatility
- recommended_strategy_type, max_leverage_allowed
- max_position_size_pct, min_confidence_to_act

RegimeState dataclass:
- label, state_id, probability, state_probabilities
- timestamp, is_confirmed, consecutive_bars

Log regime changes as WARNING. Log confirmations as INFO.

---

## PHASE 3: Volatility-Based Allocation Strategy

Implement core/regime_strategies.py — the allocation layer that sizes positions based on the HMM's volatility regime detection.

DESIGN INSIGHT: The HMM excels at detecting VOLATILITY ENVIRONMENTS, not market direction. Stocks trend upward roughly 70% of the time in low-volatility periods.
