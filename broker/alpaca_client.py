"""Thin wrapper around Alpaca's trading API (the alpaca-py SDK).

Paper trading is the default and should be used for everything except a
deliberate, explicitly-confirmed live run. Credentials are read from the
environment (.env, loaded via python-dotenv) — never hardcoded — and .env
is gitignored.

This module owns ACCOUNT / ORDER / POSITION operations only. Market data
(bars, quotes, streaming) lives in data/market_data.py — a deliberate split
matching the project layout (broker/ = trading, data/ = market data).
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Callable, Optional

from alpaca.common.exceptions import APIError
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide
from alpaca.trading.models import Order
from alpaca.trading.requests import (
    ClosePositionRequest,
    GetOrdersRequest,
    OrderRequest,
    ReplaceOrderRequest,
)

logger = logging.getLogger(__name__)

PAPER_BASE_URL = "https://paper-api.alpaca.markets"
LIVE_BASE_URL = "https://api.alpaca.markets"
LIVE_TRADING_CONFIRMATION_PHRASE = "YES I UNDERSTAND THE RISKS"


class AlpacaConnectionError(RuntimeError):
    """Raised when the Alpaca API is unreachable after all retries."""


class AlpacaAuthError(RuntimeError):
    """Raised immediately on a 401/403 -- bad/placeholder API keys never
    succeed on retry, so this skips the retry/backoff entirely instead of
    wasting several seconds reproducing the same failure.
    """


class LiveTradingNotConfirmedError(RuntimeError):
    """Raised when live (non-paper) trading was requested but not confirmed."""


def load_alpaca_credentials(
    api_key: Optional[str] = None, secret_key: Optional[str] = None, paper: Optional[bool] = None,
) -> tuple[str, str, bool]:
    """Resolve credentials from arguments, falling back to the environment."""
    api_key = api_key or os.environ.get("ALPACA_API_KEY")
    secret_key = secret_key or os.environ.get("ALPACA_SECRET_KEY")
    if paper is None:
        paper = os.environ.get("ALPACA_PAPER", "true").strip().lower() != "false"

    if not api_key or not secret_key:
        raise ValueError(
            "ALPACA_API_KEY / ALPACA_SECRET_KEY not set. Copy .env.example to .env and fill them in."
        )
    return api_key, secret_key, paper


def confirm_live_trading(input_func: Callable[[str], str] = input) -> None:
    """Blocks for operator confirmation before live trading proceeds.

    Raises LiveTradingNotConfirmedError if the exact phrase isn't typed.
    Never call this for paper trading.
    """
    print("\n" + "=" * 70)
    print("WARNING: LIVE TRADING MODE -- this will place REAL orders with REAL money.")
    print("=" * 70)
    response = input_func(f"Type '{LIVE_TRADING_CONFIRMATION_PHRASE}' to confirm: ")
    if response.strip() != LIVE_TRADING_CONFIRMATION_PHRASE:
        raise LiveTradingNotConfirmedError("Live trading not confirmed -- aborting startup.")


@dataclass
class AccountSnapshot:
    """A point-in-time read of account state."""

    equity: float
    cash: float
    buying_power: float
    portfolio_value: float
    pattern_day_trader: bool
    trading_blocked: bool
    account_blocked: bool
    daytrade_count: int


def _order_to_dict(order: Order) -> dict:
    return {
        "id": str(order.id),
        "client_order_id": order.client_order_id,
        "symbol": order.symbol,
        "qty": float(order.qty) if order.qty is not None else None,
        "side": order.side.value if hasattr(order.side, "value") else order.side,
        "order_type": order.order_type.value if hasattr(order.order_type, "value") else order.order_type,
        "status": order.status.value if hasattr(order.status, "value") else order.status,
        "filled_qty": float(order.filled_qty) if order.filled_qty is not None else 0.0,
        "filled_avg_price": float(order.filled_avg_price) if order.filled_avg_price is not None else None,
        "limit_price": float(order.limit_price) if order.limit_price is not None else None,
        "stop_price": float(order.stop_price) if order.stop_price is not None else None,
        "submitted_at": order.submitted_at,
        "filled_at": order.filled_at,
        "canceled_at": order.canceled_at,
        "legs": [_order_to_dict(leg) for leg in order.legs] if getattr(order, "legs", None) else [],
    }


class AlpacaClient:
    """Wraps alpaca-py's TradingClient behind a small, retry-hardened, testable interface."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        paper: Optional[bool] = None,
        max_retries: int = 3,
        confirm_live: bool = True,
        trading_client: Optional[TradingClient] = None,
        input_func: Callable[[str], str] = input,
    ) -> None:
        api_key, secret_key, paper = load_alpaca_credentials(api_key, secret_key, paper)

        if not paper and confirm_live:
            confirm_live_trading(input_func=input_func)

        self.paper = paper
        self.max_retries = max_retries
        # trading_client injection point lets tests substitute a fake/mock client.
        self.trading_client = trading_client or TradingClient(api_key, secret_key, paper=paper)
        self.health_check()

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def _call_with_retry(self, func: Callable, *args, **kwargs):
        delay = 1.0
        last_exc: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                return func(*args, **kwargs)
            except APIError as exc:
                if exc.status_code in (401, 403):
                    raise AlpacaAuthError(
                        "Alpaca rejected your API key/secret (401/403). Check ALPACA_API_KEY and "
                        "ALPACA_SECRET_KEY in .env -- these still look like the .env.example "
                        "placeholders if you haven't replaced them yet."
                    ) from exc
                last_exc = exc
                logger.warning("Alpaca API error (attempt %d/%d): %s", attempt, self.max_retries, exc)
                if attempt < self.max_retries:
                    time.sleep(delay)
                    delay *= 2
        raise AlpacaConnectionError(f"Alpaca API unreachable after {self.max_retries} attempts") from last_exc

    def health_check(self) -> bool:
        """Verify connectivity by fetching the account once."""
        self._call_with_retry(self.trading_client.get_account)
        logger.info("Alpaca connection OK (paper=%s)", self.paper)
        return True

    # ------------------------------------------------------------------
    # Account / market state
    # ------------------------------------------------------------------

    def get_account(self) -> AccountSnapshot:
        account = self._call_with_retry(self.trading_client.get_account)
        return AccountSnapshot(
            equity=float(account.equity),
            cash=float(account.cash),
            buying_power=float(account.buying_power),
            portfolio_value=float(account.portfolio_value),
            pattern_day_trader=bool(account.pattern_day_trader),
            trading_blocked=bool(account.trading_blocked),
            account_blocked=bool(account.account_blocked),
            daytrade_count=int(account.daytrade_count or 0),  # Alpaca returns None, not 0, on a brand-new account
        )

    def get_available_margin(self) -> float:
        """Buying power extended by the broker beyond the account's own cash."""
        account = self._call_with_retry(self.trading_client.get_account)
        return float(account.buying_power) - float(account.cash)

    def get_positions(self) -> list[dict]:
        positions = self._call_with_retry(self.trading_client.get_all_positions)
        return [
            {
                "symbol": p.symbol,
                "qty": float(p.qty),
                "avg_entry_price": float(p.avg_entry_price),
                "current_price": float(p.current_price),
                "market_value": float(p.market_value),
                "unrealized_pl": float(p.unrealized_pl),
                "side": p.side.value if hasattr(p.side, "value") else p.side,
            }
            for p in positions
        ]

    def get_order_history(self, status: str = "all", limit: int = 100) -> list[dict]:
        request = GetOrdersRequest(status=status, limit=limit)
        orders = self._call_with_retry(self.trading_client.get_orders, request)
        return [_order_to_dict(o) for o in orders]

    def get_clock(self) -> dict:
        clock = self._call_with_retry(self.trading_client.get_clock)
        return {
            "is_open": bool(clock.is_open),
            "next_open": clock.next_open,
            "next_close": clock.next_close,
            "timestamp": clock.timestamp,
        }

    def is_market_open(self) -> bool:
        return self.get_clock()["is_open"]

    # ------------------------------------------------------------------
    # Orders / positions (raw passthroughs used by broker.order_executor)
    # ------------------------------------------------------------------

    def submit_order(self, order_request: OrderRequest) -> dict:
        order = self._call_with_retry(self.trading_client.submit_order, order_request)
        return _order_to_dict(order)

    def get_order(self, order_id: str) -> dict:
        order = self._call_with_retry(self.trading_client.get_order_by_id, order_id)
        return _order_to_dict(order)

    def cancel_order(self, order_id: str) -> bool:
        try:
            self._call_with_retry(self.trading_client.cancel_order_by_id, order_id)
            return True
        except AlpacaConnectionError:
            return False

    def replace_order(self, order_id: str, replace_request: ReplaceOrderRequest) -> dict:
        order = self._call_with_retry(self.trading_client.replace_order_by_id, order_id, replace_request)
        return _order_to_dict(order)

    def close_position(self, symbol: str, request: Optional[ClosePositionRequest] = None) -> dict:
        order = self._call_with_retry(self.trading_client.close_position, symbol, request)
        return _order_to_dict(order)

    def close_all_positions(self, cancel_orders: bool = True) -> list[dict]:
        """Liquidate every open position. Returns one {symbol, order_id, status} dict per position.

        (`.body`'s shape varies by response, so we surface only the fields
        ClosePositionResponse itself guarantees rather than reusing _order_to_dict.)
        """
        results = self._call_with_retry(self.trading_client.close_all_positions, cancel_orders)
        return [{"symbol": r.symbol, "order_id": r.order_id, "status": r.status} for r in results]
