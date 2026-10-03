"""Tracks open positions with regime/P&L context, fed by Alpaca fills.

Two update paths keep this in sync:
  - `sync_with_broker()` — a full reconciliation against Alpaca's own view,
    meant to run once at startup (recovering from a restart).
  - `on_fill()` — incremental updates from the trade-updates WebSocket
    stream, called once per fill event while the bot is running.

`start_streaming()` only constructs the stream and registers the callback;
it does not block. Running the stream's event loop (`.run()`, which never
returns) is left to the caller (Phase 7's main loop), typically in its own
thread or asyncio task, so this class stays synchronous and unit-testable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

from broker.alpaca_client import AlpacaClient
from core.risk_manager import PortfolioState, PositionInfo

logger = logging.getLogger(__name__)


@dataclass
class TrackedPosition:
    """One open position, enriched with the context core.risk_manager and
    performance reporting need that a raw broker position doesn't carry.
    """

    symbol: str
    qty: float
    entry_price: float
    entry_time: datetime
    current_price: float
    stop_level: Optional[float] = None
    regime_at_entry: Optional[str] = None
    current_regime: Optional[str] = None
    trade_id: Optional[str] = None
    sector: Optional[str] = None

    @property
    def market_value(self) -> float:
        return self.qty * self.current_price

    @property
    def unrealized_pnl(self) -> float:
        return (self.current_price - self.entry_price) * self.qty

    @property
    def unrealized_pnl_pct(self) -> float:
        if self.entry_price == 0:
            return 0.0
        return self.current_price / self.entry_price - 1.0

    def holding_period(self, as_of: Optional[datetime] = None) -> timedelta:
        return (as_of or datetime.now()) - self.entry_time


class PositionTracker:
    """Maintains the book of open positions, regime context, and P&L."""

    def __init__(self, alpaca_client: AlpacaClient) -> None:
        self.alpaca_client = alpaca_client
        self.positions: dict[str, TrackedPosition] = {}
        self._stream = None  # alpaca.trading.stream.TradingStream, set by start_streaming()

    # ------------------------------------------------------------------
    # Startup recovery
    # ------------------------------------------------------------------

    def sync_with_broker(self) -> None:
        """Reconcile tracked positions against Alpaca's actual open positions.

        Positions the broker no longer shows are dropped (closed outside
        this process, e.g. manually or by a stop). Positions the broker
        shows that we aren't tracking are adopted with regime context
        unknown — better to track an unexplained position than to ignore it.
        """
        broker_positions = {p["symbol"]: p for p in self.alpaca_client.get_positions()}

        for symbol in list(self.positions):
            if symbol not in broker_positions:
                logger.warning("Position %s tracked locally but not open at broker -- dropping", symbol)
                del self.positions[symbol]

        for symbol, p in broker_positions.items():
            if symbol in self.positions:
                tracked = self.positions[symbol]
                tracked.qty = p["qty"]
                tracked.current_price = p["current_price"]
            else:
                logger.warning("Position %s open at broker but not tracked locally -- adopting, regime unknown", symbol)
                self.positions[symbol] = TrackedPosition(
                    symbol=symbol,
                    qty=p["qty"],
                    entry_price=p["avg_entry_price"],
                    entry_time=datetime.now(),
                    current_price=p["current_price"],
                )

    # ------------------------------------------------------------------
    # Live updates
    # ------------------------------------------------------------------

    def open_position(
        self, symbol: str, qty: float, entry_price: float, stop_level: Optional[float] = None,
        regime_at_entry: Optional[str] = None, trade_id: Optional[str] = None, sector: Optional[str] = None,
        entry_time: Optional[datetime] = None,
    ) -> TrackedPosition:
        position = TrackedPosition(
            symbol=symbol, qty=qty, entry_price=entry_price, entry_time=entry_time or datetime.now(),
            current_price=entry_price, stop_level=stop_level, regime_at_entry=regime_at_entry,
            current_regime=regime_at_entry, trade_id=trade_id, sector=sector,
        )
        self.positions[symbol] = position
        return position

    def on_fill(self, fill_event: dict) -> None:
        """Handle one trade-update event from the WebSocket stream.

        `fill_event` is the parsed payload: {event, symbol, qty, price, side, ...}
        matching alpaca-py's TradeUpdate.
        """
        event = fill_event.get("event")
        symbol = fill_event["symbol"]

        if event in ("fill", "partial_fill"):
            qty = float(fill_event["qty"])
            price = float(fill_event["price"])
            side = fill_event.get("side", "buy")

            if symbol not in self.positions:
                if side == "buy":
                    self.open_position(symbol, qty=qty, entry_price=price, trade_id=fill_event.get("client_order_id"))
                return

            position = self.positions[symbol]
            if side == "sell":
                position.qty -= qty
                if position.qty <= 0:
                    logger.info("Position %s closed (sold %s)", symbol, qty)
                    del self.positions[symbol]
                return

            # Additional buy fill: blend into a new average entry price.
            total_qty = position.qty + qty
            if total_qty > 0:
                position.entry_price = (position.entry_price * position.qty + price * qty) / total_qty
            position.qty = total_qty

        elif event in ("canceled", "rejected", "expired"):
            logger.info("Order for %s ended with event=%s, no position change", symbol, event)

    def update_price(self, symbol: str, price: float) -> None:
        if symbol in self.positions:
            self.positions[symbol].current_price = price

    def update_current_regime(self, symbol: str, regime_label: str) -> None:
        if symbol in self.positions:
            self.positions[symbol].current_regime = regime_label

    def update_stop(self, symbol: str, stop_level: float) -> None:
        if symbol in self.positions:
            self.positions[symbol].stop_level = stop_level

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_position(self, symbol: str) -> Optional[TrackedPosition]:
        return self.positions.get(symbol)

    def get_all_positions(self) -> list[TrackedPosition]:
        return list(self.positions.values())

    def total_exposure(self) -> float:
        return sum(abs(p.market_value) for p in self.positions.values())

    def total_unrealized_pnl(self) -> float:
        return sum(p.unrealized_pnl for p in self.positions.values())

    def to_portfolio_state(self, equity: float, cash: float, buying_power: float, **kwargs) -> PortfolioState:
        """Build a core.risk_manager.PortfolioState snapshot from what's tracked here."""
        positions = {
            symbol: PositionInfo(
                symbol=p.symbol, qty=p.qty, entry_price=p.entry_price, current_price=p.current_price,
                sector=p.sector,
            )
            for symbol, p in self.positions.items()
        }
        return PortfolioState(equity=equity, cash=cash, buying_power=buying_power, positions=positions, **kwargs)

    # ------------------------------------------------------------------
    # Streaming (construction only — caller runs the event loop)
    # ------------------------------------------------------------------

    def start_streaming(self, api_key: str, secret_key: str, paper: bool = True):
        """Construct a TradingStream and register on_fill as its callback.

        Returns the stream object; the caller is responsible for running
        its event loop (`stream.run()`, which blocks) in its own
        thread/task, and for calling `.stop()` on shutdown.
        """
        from alpaca.trading.stream import TradingStream  # imported lazily: not needed outside live trading

        self._stream = TradingStream(api_key, secret_key, paper=paper)

        async def _handle(data) -> None:
            self.on_fill(
                {
                    "event": data.event,
                    "symbol": data.order.symbol,
                    "qty": data.order.filled_qty,
                    "price": data.order.filled_avg_price,
                    "side": data.order.side.value if hasattr(data.order.side, "value") else data.order.side,
                    "client_order_id": data.order.client_order_id,
                }
            )

        self._stream.subscribe_trade_updates(_handle)
        return self._stream

    def stop_streaming(self) -> None:
        if self._stream is not None:
            self._stream.stop()
