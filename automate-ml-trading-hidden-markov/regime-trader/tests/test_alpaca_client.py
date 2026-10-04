"""Tests for broker.alpaca_client. Uses a mocked alpaca-py TradingClient —
no real network access or credentials required.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from alpaca.common.exceptions import APIError

from broker.alpaca_client import (
    AlpacaAuthError,
    AlpacaClient,
    AlpacaConnectionError,
    LiveTradingNotConfirmedError,
    LIVE_TRADING_CONFIRMATION_PHRASE,
    confirm_live_trading,
    load_alpaca_credentials,
)


def _auth_error(status_code: int) -> APIError:
    http_error = SimpleNamespace(response=SimpleNamespace(status_code=status_code))
    return APIError('{"message": "unauthorized"}', http_error=http_error)


def _fake_account(**overrides) -> SimpleNamespace:
    defaults = dict(
        equity=100_000.0, cash=50_000.0, buying_power=150_000.0, portfolio_value=100_000.0,
        pattern_day_trader=False, trading_blocked=False, account_blocked=False, daytrade_count=0,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _make_client(**kwargs) -> tuple[AlpacaClient, MagicMock]:
    mock_trading = MagicMock()
    mock_trading.get_account.return_value = _fake_account()
    client = AlpacaClient(api_key="k", secret_key="s", paper=True, trading_client=mock_trading, **kwargs)
    return client, mock_trading


# ---------------------------------------------------------------------------
# Credentials / live-trading confirmation
# ---------------------------------------------------------------------------

def test_load_credentials_raises_without_keys(monkeypatch):
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)
    with pytest.raises(ValueError):
        load_alpaca_credentials()


def test_load_credentials_defaults_paper_true(monkeypatch):
    monkeypatch.delenv("ALPACA_PAPER", raising=False)
    _, _, paper = load_alpaca_credentials("k", "s")
    assert paper is True


def test_load_credentials_respects_env_paper_false(monkeypatch):
    monkeypatch.setenv("ALPACA_PAPER", "false")
    _, _, paper = load_alpaca_credentials("k", "s")
    assert paper is False


def test_confirm_live_trading_raises_on_wrong_phrase():
    with pytest.raises(LiveTradingNotConfirmedError):
        confirm_live_trading(input_func=lambda _: "nope")


def test_confirm_live_trading_passes_on_exact_phrase():
    confirm_live_trading(input_func=lambda _: LIVE_TRADING_CONFIRMATION_PHRASE)  # should not raise


def test_live_mode_without_confirmation_raises():
    mock_trading = MagicMock()
    mock_trading.get_account.return_value = _fake_account()
    with pytest.raises(LiveTradingNotConfirmedError):
        AlpacaClient(
            api_key="k", secret_key="s", paper=False, trading_client=mock_trading,
            confirm_live=True, input_func=lambda _: "nope",
        )


# ---------------------------------------------------------------------------
# Account / positions
# ---------------------------------------------------------------------------

def test_get_account_converts_fields():
    client, mock_trading = _make_client()
    mock_trading.get_account.return_value = _fake_account(equity=250_000.0, daytrade_count=2)
    account = client.get_account()
    assert account.equity == 250_000.0
    assert account.daytrade_count == 2


def test_get_account_treats_none_daytrade_count_as_zero():
    """A brand-new paper account (zero trades ever) returns daytrade_count=None,
    not 0 -- int(None) crashes TradingBot.startup() on the very first run.
    """
    client, mock_trading = _make_client()
    mock_trading.get_account.return_value = _fake_account(daytrade_count=None)
    account = client.get_account()
    assert account.daytrade_count == 0


def test_get_available_margin():
    client, mock_trading = _make_client()
    mock_trading.get_account.return_value = _fake_account(cash=30_000.0, buying_power=100_000.0)
    assert client.get_available_margin() == pytest.approx(70_000.0)


def test_get_positions_converts_fields():
    client, mock_trading = _make_client()
    mock_trading.get_all_positions.return_value = [
        SimpleNamespace(
            symbol="AAPL", qty=10, avg_entry_price=150.0, current_price=155.0,
            market_value=1550.0, unrealized_pl=50.0, side=SimpleNamespace(value="long"),
        )
    ]
    positions = client.get_positions()
    assert positions == [
        {
            "symbol": "AAPL", "qty": 10.0, "avg_entry_price": 150.0, "current_price": 155.0,
            "market_value": 1550.0, "unrealized_pl": 50.0, "side": "long",
        }
    ]


def test_is_market_open():
    client, mock_trading = _make_client()
    mock_trading.get_clock.return_value = SimpleNamespace(
        is_open=True, next_open=None, next_close=None, timestamp=None,
    )
    assert client.is_market_open() is True


# ---------------------------------------------------------------------------
# Retry / reconnect
# ---------------------------------------------------------------------------

def test_retries_then_succeeds(monkeypatch):
    client, mock_trading = _make_client(max_retries=3)
    monkeypatch.setattr("broker.alpaca_client.time.sleep", lambda _: None)
    mock_trading.get_account.side_effect = [APIError("boom"), _fake_account(equity=42.0)]
    account = client.get_account()
    assert account.equity == 42.0


def test_retries_exhausted_raises_connection_error(monkeypatch):
    client, mock_trading = _make_client(max_retries=2)
    monkeypatch.setattr("broker.alpaca_client.time.sleep", lambda _: None)
    mock_trading.get_account.reset_mock()  # drop the constructor's own health-check call
    mock_trading.get_account.side_effect = APIError("still down")
    with pytest.raises(AlpacaConnectionError):
        client.get_account()
    assert mock_trading.get_account.call_count == 2


@pytest.mark.parametrize("status_code", [401, 403])
def test_auth_error_fails_fast_without_retrying(monkeypatch, status_code):
    client, mock_trading = _make_client(max_retries=3)
    sleeps = []
    monkeypatch.setattr("broker.alpaca_client.time.sleep", lambda d: sleeps.append(d))
    mock_trading.get_account.reset_mock()
    mock_trading.get_account.side_effect = _auth_error(status_code)

    with pytest.raises(AlpacaAuthError):
        client.get_account()

    assert mock_trading.get_account.call_count == 1  # no retries on a hard auth failure
    assert sleeps == []  # and no backoff delay burned either
