# regime-trader

HMM regime-detection trading bot. Classifies market volatility regimes with a
Gaussian HMM and drives portfolio allocation off that classification (the HMM
is a volatility classifier, not a direction predictor).

## Status

- [x] Phase 1 — project scaffolding
- [x] Phase 2 — HMM regime detection engine
- [ ] Phase 3+ — see `ultimate_hmm_regime_trading_bot_guide.md`

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # fill in Alpaca keys
cp config/credentials.yaml.example config/credentials.yaml
```

Edit `config/settings.yaml` to tune parameters.

## Running tests

```bash
pytest tests/
```

`tests/test_look_ahead.py` is the critical test — it verifies the regime
detector never uses future data to label the present.
