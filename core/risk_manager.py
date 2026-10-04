"""Independent risk management layer with absolute veto power over signals.

This module does NOT trust the HMM. Even if regime detection is completely
wrong, the circuit breakers here fire off ACTUAL portfolio P&L — defense in
depth. A Signal from core.regime_strategies.StrategyOrchestrator is a
*proposal*; this layer can shrink it, force its leverage to 1.0x, or refuse
it outright, and nothing downstream may override that.

DESIGN NOTE on position_size_pct: core.regime_strategies.Signal documents
position_size_pct as "target fraction of PORTFOLIO equity" — this module
takes that literally, against the TOTAL account, not a per-symbol capital
sleeve. That matters: it's why max_single_position (15%) routinely caps a
LowVolBullStrategy signal's 95% request down hard. That is intentional, not
a bug — the system is designed to run one shared regime signal across a
basket of symbols (see config/settings.yaml's broker.symbols), and this cap
is what keeps any single name from dominating the book. (backtest.backtester
instead gives each symbol its own independent capital sleeve for backtesting
convenience — that is a documented backtesting-only simplification, not how
a live, single shared account behaves, which is what this module models.)
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from enum import Enum
from typing import Optional

import pandas as pd

from core.regime_strategies import Signal

logger = logging.getLogger(__name__)

DEFAULT_LOCK_FILE = "trading_halted.lock"


class CircuitBreakerStatus(Enum):
    """Current drawdown-driven risk posture. Independent of regime detection."""

    NORMAL = "normal"
    DAILY_REDUCE = "daily_reduce"
    DAILY_HALT = "daily_halt"
    WEEKLY_REDUCE = "weekly_reduce"
    WEEKLY_HALT = "weekly_halt"
    PEAK_HALT = "peak_halt"


_HALT_STATUSES = {CircuitBreakerStatus.DAILY_HALT, CircuitBreakerStatus.WEEKLY_HALT, CircuitBreakerStatus.PEAK_HALT}
_REDUCE_STATUSES = {CircuitBreakerStatus.DAILY_REDUCE, CircuitBreakerStatus.WEEKLY_REDUCE}


@dataclass
class PositionInfo:
    """One currently-open position, as the risk manager needs to see it."""

    symbol: str
    qty: float
    entry_price: float
    current_price: float
    sector: Optional[str] = None

    @property
    def notional(self) -> float:
        return abs(self.qty) * self.current_price


@dataclass
class PortfolioState:
    """Live account/portfolio snapshot the risk manager reasons about."""

    equity: float
    cash: float
    buying_power: float
    positions: dict[str, PositionInfo] = field(default_factory=dict)
    daily_trades_count: int = 0
    recent_orders: list[tuple] = field(default_factory=list)  # (symbol, Direction, datetime)
    price_history: dict[str, pd.Series] = field(default_factory=dict)  # recent daily closes, for correlation
    tradeable: dict[str, bool] = field(default_factory=dict)
    bid_ask_spread_pct: dict[str, float] = field(default_factory=dict)

    @property
    def open_position_count(self) -> int:
        return len(self.positions)

    @property
    def total_exposure_pct(self) -> float:
        if self.equity <= 0:
            return 0.0
        return sum(p.notional for p in self.positions.values()) / self.equity


@dataclass
class CircuitBreakerEvent:
    """One logged circuit-breaker trigger — for post-mortems on whether the HMM was wrong."""

    timestamp: datetime
    breaker_type: CircuitBreakerStatus
    drawdown: float
    equity: float
    positions_to_close: list[str]
    regime_label: Optional[str]


@dataclass
class RiskDecision:
    """The risk manager's final word on a proposed signal."""

    approved: bool
    modified_signal: Optional[Signal]
    rejection_reason: Optional[str]
    modifications: list[str]


class CircuitBreaker:
    """Tracks daily/weekly/peak drawdown and fires independent of regime detection."""

    def __init__(self, config: dict, lock_file_path: str = DEFAULT_LOCK_FILE) -> None:
        self.config = config
        self.lock_file_path = lock_file_path
        self.day_start_equity: Optional[float] = None
        self.week_start_equity: Optional[float] = None
        self.peak_equity: Optional[float] = None
        self._current_day: Optional[date] = None
        self._current_week: Optional[tuple] = None
        self._status = CircuitBreakerStatus.NORMAL
        self._history: list[CircuitBreakerEvent] = []

    def reset_daily(self, equity: float, today: date) -> None:
        self.day_start_equity = equity
        self._current_day = today

    def reset_weekly(self, equity: float, today: date) -> None:
        self.week_start_equity = equity
        self._current_week = today.isocalendar()[:2]

    def update(
        self, equity: float, now: datetime, open_symbols: list[str], regime_label: Optional[str] = None,
    ) -> CircuitBreakerStatus:
        """Recompute daily/weekly/peak drawdown from the latest equity and return the status.

        Rolls day/week boundaries forward automatically based on `now`.
        """
        today = now.date()
        if self._current_day != today:
            self.reset_daily(equity, today)
        if self._current_week != today.isocalendar()[:2]:
            self.reset_weekly(equity, today)
        if self.peak_equity is None or equity > self.peak_equity:
            self.peak_equity = equity

        daily_dd = equity / self.day_start_equity - 1.0 if self.day_start_equity else 0.0
        weekly_dd = equity / self.week_start_equity - 1.0 if self.week_start_equity else 0.0
        peak_dd = equity / self.peak_equity - 1.0 if self.peak_equity else 0.0

        if peak_dd <= -self.config["max_dd_from_peak"]:
            status, worst_dd = CircuitBreakerStatus.PEAK_HALT, peak_dd
        elif weekly_dd <= -self.config["weekly_dd_halt"]:
            status, worst_dd = CircuitBreakerStatus.WEEKLY_HALT, weekly_dd
        elif daily_dd <= -self.config["daily_dd_halt"]:
            status, worst_dd = CircuitBreakerStatus.DAILY_HALT, daily_dd
        elif weekly_dd <= -self.config["weekly_dd_reduce"]:
            status, worst_dd = CircuitBreakerStatus.WEEKLY_REDUCE, weekly_dd
        elif daily_dd <= -self.config["daily_dd_reduce"]:
            status, worst_dd = CircuitBreakerStatus.DAILY_REDUCE, daily_dd
        else:
            status, worst_dd = CircuitBreakerStatus.NORMAL, 0.0

        if status != CircuitBreakerStatus.NORMAL and status != self._status:
            event = CircuitBreakerEvent(
                timestamp=now, breaker_type=status, drawdown=worst_dd, equity=equity,
                positions_to_close=list(open_symbols) if status in _HALT_STATUSES else [],
                regime_label=regime_label,
            )
            self._history.append(event)
            logger.warning(
                "CIRCUIT BREAKER %s: drawdown=%.2f%%, equity=%.2f, regime=%s, positions_to_close=%s",
                status.value, worst_dd * 100, equity, regime_label, event.positions_to_close,
            )
            if status == CircuitBreakerStatus.PEAK_HALT:
                self._write_lock_file(event)

        self._status = status
        return status

    def check(self) -> CircuitBreakerStatus:
        """Current status without recomputing it (does not advance day/week tracking).

        A manually-restored lock file always wins, even if in-memory state
        would otherwise say NORMAL (e.g. after a process restart).
        """
        if os.path.exists(self.lock_file_path):
            return CircuitBreakerStatus.PEAK_HALT
        return self._status

    def get_history(self) -> list[CircuitBreakerEvent]:
        return list(self._history)

    def _write_lock_file(self, event: CircuitBreakerEvent) -> None:
        payload = {
            "triggered_at": event.timestamp.isoformat(),
            "drawdown": event.drawdown,
            "equity": event.equity,
            "regime_label": event.regime_label,
            "positions_to_close": event.positions_to_close,
            "note": "Peak drawdown limit breached. Delete this file manually to resume trading.",
        }
        with open(self.lock_file_path, "w") as f:
            json.dump(payload, f, indent=2)


class RiskManager:
    """Independent risk layer with absolute veto power. See module docstring.

    `config` is the `risk:` section of settings.yaml.
    """

    def __init__(self, config: dict, lock_file_path: str = DEFAULT_LOCK_FILE) -> None:
        self.config = config
        self.circuit_breaker = CircuitBreaker(config, lock_file_path=lock_file_path)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def validate_signal(
        self,
        signal: Signal,
        portfolio_state: PortfolioState,
        *,
        regime_max_position_size_pct: float = 1.0,
        regime_uncertain: bool = False,
        is_flickering: bool = False,
        regime_label: Optional[str] = None,
        now: Optional[datetime] = None,
    ) -> RiskDecision:
        """Approve, shrink, or reject a proposed signal. Never increases its size or leverage."""
        now = now or datetime.now()
        modifications: list[str] = []

        breaker_status = self.circuit_breaker.update(
            portfolio_state.equity, now, list(portfolio_state.positions), regime_label,
        )
        if breaker_status in _HALT_STATUSES:
            return RiskDecision(False, None, f"circuit breaker active: {breaker_status.value}", [])

        if signal.stop_loss is None or signal.stop_loss == signal.entry_price:
            return RiskDecision(False, None, "signal has no valid stop_loss", [])

        if self._is_duplicate_order(signal, portfolio_state, now):
            return RiskDecision(False, None, "duplicate order: same symbol+direction within duplicate window", [])

        reason = self._check_tradeable(signal, portfolio_state)
        if reason:
            return RiskDecision(False, None, reason, [])

        reason = self._check_trade_limits(portfolio_state)
        if reason:
            return RiskDecision(False, None, reason, [])

        leverage, leverage_mods = self._decide_leverage(
            signal, portfolio_state, breaker_status, regime_uncertain, is_flickering,
        )
        modifications.extend(leverage_mods)

        notional_pct, size_mods = self._size_position(signal, regime_max_position_size_pct)
        modifications.extend(size_mods)

        notional_pct, corr_reason, corr_mods = self._apply_correlation_check(signal, portfolio_state, notional_pct)
        if corr_reason:
            return RiskDecision(False, None, corr_reason, modifications)
        modifications.extend(corr_mods)

        notional_pct, exposure_mods = self._apply_exposure_caps(portfolio_state, notional_pct, leverage)
        modifications.extend(exposure_mods)

        sector_reason, notional_pct, sector_mods = self._check_sector_concentration(signal, portfolio_state, notional_pct)
        if sector_reason:
            return RiskDecision(False, None, sector_reason, modifications)
        modifications.extend(sector_mods)

        if breaker_status in _REDUCE_STATUSES:
            notional_pct *= 0.5
            modifications.append(f"size halved: {breaker_status.value}")

        position_pct = min(notional_pct / leverage, signal.position_size_pct)
        notional_dollars = position_pct * leverage * portfolio_state.equity

        if notional_dollars < self.config["min_position_dollars"]:
            return RiskDecision(
                False, None,
                f"sized position (${notional_dollars:,.2f}) below ${self.config['min_position_dollars']} minimum",
                modifications,
            )
        if notional_dollars > portfolio_state.buying_power:
            return RiskDecision(False, None, "insufficient buying power for sized position", modifications)

        modified = Signal(
            symbol=signal.symbol, direction=signal.direction, confidence=signal.confidence,
            entry_price=signal.entry_price, stop_loss=signal.stop_loss, take_profit=signal.take_profit,
            position_size_pct=position_pct, leverage=leverage, regime_id=signal.regime_id,
            regime_name=signal.regime_name, regime_probability=signal.regime_probability,
            timestamp=signal.timestamp, reasoning=signal.reasoning, strategy_name=signal.strategy_name,
            metadata=signal.metadata,
        )
        return RiskDecision(True, modified, None, modifications)

    def check_circuit_breaker(
        self, portfolio_state: PortfolioState, now: Optional[datetime] = None, regime_label: Optional[str] = None,
    ) -> CircuitBreakerStatus:
        """Update + return circuit-breaker status without evaluating any particular signal.

        Useful for deciding whether open positions need to be force-liquidated
        even when there's no new candidate trade this bar.
        """
        now = now or datetime.now()
        return self.circuit_breaker.update(portfolio_state.equity, now, list(portfolio_state.positions), regime_label)

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    def _is_duplicate_order(self, signal: Signal, portfolio_state: PortfolioState, now: datetime) -> bool:
        window = timedelta(seconds=self.config["duplicate_order_window_seconds"])
        for symbol, direction, ts in portfolio_state.recent_orders:
            if symbol == signal.symbol and direction == signal.direction and now - ts < window:
                return True
        return False

    def _check_tradeable(self, signal: Signal, portfolio_state: PortfolioState) -> Optional[str]:
        if not portfolio_state.tradeable.get(signal.symbol, True):
            return f"{signal.symbol} is not tradeable"
        spread = portfolio_state.bid_ask_spread_pct.get(signal.symbol, 0.0)
        if spread > self.config["max_bid_ask_spread_pct"]:
            return f"{signal.symbol} bid-ask spread {spread:.2%} exceeds {self.config['max_bid_ask_spread_pct']:.2%} limit"
        return None

    def _check_trade_limits(self, portfolio_state: PortfolioState) -> Optional[str]:
        if portfolio_state.daily_trades_count >= self.config["max_daily_trades"]:
            return "max_daily_trades limit reached"
        if portfolio_state.open_position_count >= self.config["max_concurrent"]:
            return "max_concurrent positions limit reached"
        return None

    def _decide_leverage(
        self, signal: Signal, portfolio_state: PortfolioState, breaker_status: CircuitBreakerStatus,
        regime_uncertain: bool, is_flickering: bool,
    ) -> tuple[float, list[str]]:
        """Leverage can only ever be forced DOWN from what the strategy requested."""
        force_reasons = []
        if regime_uncertain:
            force_reasons.append("regime uncertain")
        if breaker_status != CircuitBreakerStatus.NORMAL:
            force_reasons.append(f"circuit breaker {breaker_status.value}")
        if portfolio_state.open_position_count >= 3:
            force_reasons.append("3+ positions open")
        if is_flickering:
            force_reasons.append("high flicker rate")

        leverage = signal.leverage
        modifications = []
        if force_reasons and leverage > 1.0:
            leverage = 1.0
            modifications.append(f"leverage forced to 1.0x: {', '.join(force_reasons)}")

        leverage = min(leverage, self.config["max_leverage"])
        return leverage, modifications

    def _size_position(self, signal: Signal, regime_max_position_size_pct: float) -> tuple[float, list[str]]:
        """Classic 1%-risk position sizing, independently capped by gap risk,
        the regime's own max, and the portfolio's max_single_position — never
        larger than what the strategy itself requested.
        """
        modifications = []
        entry, stop = signal.entry_price, signal.stop_loss
        stop_distance_pct = abs(entry - stop) / entry

        risk_based_pct = self.config["max_risk_per_trade"] / stop_distance_pct
        gap_based_pct = self.config["overnight_max_pct"] / (self.config["gap_risk_atr_mult"] * stop_distance_pct)
        requested = signal.position_size_pct * signal.leverage

        capped = min(requested, risk_based_pct, gap_based_pct, regime_max_position_size_pct, self.config["max_single_position"])
        if capped < requested:
            modifications.append(f"size capped from {requested:.1%} to {capped:.1%} (1%-risk/gap/regime/position caps)")
        return capped, modifications

    def _apply_correlation_check(
        self, signal: Signal, portfolio_state: PortfolioState, notional_pct: float,
    ) -> tuple[float, Optional[str], list[str]]:
        history = portfolio_state.price_history.get(signal.symbol)
        if history is None or not portfolio_state.positions:
            return notional_pct, None, []

        max_corr = 0.0
        for existing_symbol in portfolio_state.positions:
            if existing_symbol == signal.symbol:
                continue
            other = portfolio_state.price_history.get(existing_symbol)
            if other is None:
                continue
            aligned = pd.concat([history, other], axis=1).dropna().tail(60)
            if len(aligned) < 20:
                continue
            corr = aligned.iloc[:, 0].pct_change().corr(aligned.iloc[:, 1].pct_change())
            if pd.notna(corr):
                max_corr = max(max_corr, abs(float(corr)))

        if max_corr > self.config["correlation_reject_threshold"]:
            return notional_pct, f"correlation {max_corr:.2f} with an existing position exceeds reject threshold", []
        if max_corr > self.config["correlation_reduce_threshold"]:
            return notional_pct * 0.5, None, [f"size halved: correlation {max_corr:.2f} with an existing position"]
        return notional_pct, None, []

    def _apply_exposure_caps(
        self, portfolio_state: PortfolioState, notional_pct: float, leverage: float,
    ) -> tuple[float, list[str]]:
        existing = portfolio_state.total_exposure_pct
        # A low-vol leveraged position is allowed to push total notional past
        # max_exposure (the guide's own note) — but max_leverage is the
        # absolute ceiling regardless of which strategy proposed the trade.
        soft_ceiling = self.config["max_leverage"] if leverage > 1.0 else self.config["max_exposure"]

        room = max(soft_ceiling - existing, 0.0)
        hard_room = max(self.config["max_leverage"] - existing, 0.0)
        capped = min(notional_pct, room, hard_room)

        modifications = []
        if capped < notional_pct:
            modifications.append(f"size reduced from {notional_pct:.1%} to {capped:.1%} by portfolio exposure cap")
        return capped, modifications

    def _check_sector_concentration(
        self, signal: Signal, portfolio_state: PortfolioState, notional_pct: float,
    ) -> tuple[Optional[str], float, list[str]]:
        sector = signal.metadata.get("sector") if isinstance(signal.metadata, dict) else None
        if not sector:
            return None, notional_pct, []  # sector unknown: skip gracefully (no data source wired up yet)

        same_sector_exposure = sum(
            p.notional / portfolio_state.equity for p in portfolio_state.positions.values() if p.sector == sector
        )
        projected = same_sector_exposure + notional_pct
        if projected > self.config["max_correlated_exposure"]:
            return (
                f"sector '{sector}' exposure would reach {projected:.1%}, over the "
                f"{self.config['max_correlated_exposure']:.0%} limit",
                notional_pct,
                [],
            )
        return None, notional_pct, []
