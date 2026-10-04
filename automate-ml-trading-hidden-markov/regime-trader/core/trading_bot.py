"""Main orchestration: wires the HMM, strategy, risk, and broker layers
into one continuously-running (or dry-run / one-shot) trading loop.

DESIGN SPLIT FOR TESTABILITY: everything that can run without a live
network connection — the per-bar decision pipeline, HMM retrain
scheduling, state-snapshot save/load, trailing-stop updates, forced
liquidation on a circuit-breaker halt — is a plain synchronous method on
TradingBot, unit tested directly against mocked broker/data clients.

Only `run()` itself — "subscribe to the live WebSocket bar feed and loop
forever" — can't be meaningfully unit tested without a real (paper) Alpaca
connection. Everything `run()` calls internally is tested in isolation;
`run()` is a thin wrapper gluing them to the live feed. Same caveat as
broker.position_tracker's streaming and data.market_data's run_stream().
"""

from __future__ import annotations

import json
import logging
import os
import signal
import time
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Optional

import pandas as pd

from broker.alpaca_client import AlpacaClient
from broker.order_executor import OrderExecutor
from broker.position_tracker import PositionTracker
from core.hmm_engine import HMMRegimeEngine, RegimeState
from core.regime_strategies import Signal, StrategyOrchestrator
from core.risk_manager import CircuitBreakerStatus, PortfolioState, RiskManager
from data.feature_engineering import compute_features, log_returns
from data.market_data import MarketDataFeed
from monitoring.alerts import AlertManager, AlertSeverity, AlertType

logger = logging.getLogger("regime_trader.main")

DEFAULT_STATE_PATH = "state_snapshot.json"
DEFAULT_MODEL_PATH = "hmm_model.pkl"
DEFAULT_DASHBOARD_JSON_PATH = "dashboard_state.json"
HMM_RETRAIN_MAX_AGE_DAYS = 7


@dataclass
class BarDecision:
    """Everything that happened for one symbol on one bar — for logging/dashboard."""

    symbol: str
    regime_state: Optional[RegimeState]
    signal: Optional[Signal]
    approved: bool
    modifications: list[str] = field(default_factory=list)
    rejection_reason: Optional[str] = None
    order_result: Optional[object] = None  # broker.order_executor.OrderResult, when an order was placed


class TradingBot:
    """Ties HMM + strategy + risk + broker together into one trading loop."""

    def __init__(
        self,
        config: dict,
        alpaca_client: AlpacaClient,
        market_data: MarketDataFeed,
        hmm_engine: Optional[HMMRegimeEngine] = None,
        risk_manager: Optional[RiskManager] = None,
        order_executor: Optional[OrderExecutor] = None,
        position_tracker: Optional[PositionTracker] = None,
        alert_manager: Optional[AlertManager] = None,
        dry_run: bool = False,
        state_path: str = DEFAULT_STATE_PATH,
        model_path: str = DEFAULT_MODEL_PATH,
    ) -> None:
        self.config = config
        self.alpaca_client = alpaca_client
        self.market_data = market_data
        self.hmm_engine = hmm_engine
        self.orchestrator: Optional[StrategyOrchestrator] = None
        self.risk_manager = risk_manager or RiskManager(config["risk"])
        self.order_executor = order_executor or OrderExecutor(alpaca_client)
        self.position_tracker = position_tracker or PositionTracker(alpaca_client)
        self.alert_manager = alert_manager or AlertManager(config["monitoring"])
        self.dry_run = dry_run
        self.state_path = state_path
        self.model_path = model_path

        self.last_train_date: Optional[datetime] = None
        self.last_regime_state: dict[str, RegimeState] = {}
        self.recent_signals: list[dict] = []
        self.recent_orders: list[tuple] = []  # (symbol, Direction, datetime) -- feeds RiskManager's duplicate-order check
        self.daily_trades_count = 0
        self._trade_day: Optional[date] = None
        self._running = False

        if self.hmm_engine is not None and self.hmm_engine.regime_info:
            self.orchestrator = StrategyOrchestrator(config["strategy"], self.hmm_engine.regime_info)

    # ------------------------------------------------------------------
    # Startup
    # ------------------------------------------------------------------

    def startup(
        self,
        features_by_symbol: Optional[dict[str, pd.DataFrame]] = None,
        returns_by_symbol: Optional[dict[str, pd.Series]] = None,
    ) -> None:
        """Steps 1, 3-6, 8 of the guide's STARTUP sequence.

        Market-hours waiting (step 2) and WebSocket connection (step 7) are
        handled by run(), not here — both involve real blocking/network I/O
        that doesn't belong in a unit-testable method.
        """
        account = self.alpaca_client.get_account()
        logger.info("Connected to Alpaca (paper=%s), equity=$%.2f", self.alpaca_client.paper, account.equity)

        if self._model_is_stale() and features_by_symbol and returns_by_symbol:
            primary_symbol = next(iter(features_by_symbol))
            self.train_hmm(features_by_symbol[primary_symbol], returns_by_symbol[primary_symbol])
        elif self.hmm_engine is None:
            logger.warning("No HMM model available and no training data provided at startup")

        self.position_tracker.sync_with_broker()
        self._load_state_snapshot()

        logger.info("System online (dry_run=%s)", self.dry_run)

    def _model_is_stale(self) -> bool:
        if self.hmm_engine is None or self.last_train_date is None:
            return True
        return datetime.now() - self.last_train_date > timedelta(days=HMM_RETRAIN_MAX_AGE_DAYS)

    # ------------------------------------------------------------------
    # HMM training / retraining
    # ------------------------------------------------------------------

    def train_hmm(self, features: pd.DataFrame, returns: pd.Series) -> None:
        engine = self.hmm_engine or HMMRegimeEngine(self.config["hmm"])
        engine.fit(features, returns)
        self.hmm_engine = engine
        self.orchestrator = StrategyOrchestrator(self.config["strategy"], engine.regime_info)
        self.last_train_date = datetime.now()
        try:
            engine.save(self.model_path)
        except OSError as exc:
            logger.warning("Could not save HMM model to %s: %s", self.model_path, exc)

        logger.info("HMM trained: n_regimes=%d", engine.n_regimes)
        self.alert_manager.send_alert(
            AlertType.HMM_RETRAINED, f"HMM (re)trained: {engine.n_regimes} regimes", AlertSeverity.INFO,
        )

    def maybe_retrain_weekly(self, features: pd.DataFrame, returns: pd.Series, now: Optional[datetime] = None) -> bool:
        now = now or datetime.now()
        if self.last_train_date is None or (now - self.last_train_date) >= timedelta(days=7):
            self.train_hmm(features, returns)
            return True
        return False

    # ------------------------------------------------------------------
    # Per-bar pipeline (MAIN LOOP steps 2-7)
    # ------------------------------------------------------------------

    def process_bar(self, symbol: str, bars: pd.DataFrame, now: Optional[datetime] = None) -> BarDecision:
        now = now or datetime.now()
        self._roll_daily_counters(now)

        if self.hmm_engine is None or self.orchestrator is None:
            return BarDecision(symbol, None, None, False, [], "no HMM model trained yet")

        regime_state = self._infer_regime(symbol, bars)
        if regime_state is None:
            return BarDecision(symbol, None, None, False, [], "no regime available (HMM error, no prior state)")

        is_flickering = self.hmm_engine.is_flickering()
        if is_flickering:
            self.alert_manager.send_alert(
                AlertType.FLICKER_EXCEEDED,
                f"{symbol}: regime flickering (rate={self.hmm_engine.get_regime_flicker_rate()})",
                AlertSeverity.WARNING, context={"symbol": symbol}, now=now,
            )

        signals = self.orchestrator.generate_signals([symbol], {symbol: bars}, regime_state, is_flickering)
        signal = signals[0] if signals else None
        if signal is None:
            self._record_signal(now, symbol, "no_action", "insufficient price history for this regime's strategy")
            return BarDecision(symbol, regime_state, None, False, [], "no signal generated")

        portfolio_state = self._build_portfolio_state()
        regime_info = self.hmm_engine.regime_info.get(regime_state.state_id)
        regime_max = regime_info.max_position_size_pct if regime_info else 1.0
        min_confidence = regime_info.min_confidence_to_act if regime_info else self.config["hmm"].get("min_confidence", 0.55)

        decision = self.risk_manager.validate_signal(
            signal, portfolio_state,
            regime_max_position_size_pct=regime_max,
            regime_uncertain=regime_state.probability < min_confidence,
            is_flickering=is_flickering,
            regime_label=regime_state.label,
            now=now,
        )

        if not decision.approved:
            self._record_signal(now, symbol, "rejected", decision.rejection_reason)
            logger.info("Signal rejected for %s: %s", symbol, decision.rejection_reason)
            return BarDecision(symbol, regime_state, signal, False, decision.modifications, decision.rejection_reason)

        final_signal = decision.modified_signal
        if decision.modifications:
            logger.info("Signal for %s modified: %s", symbol, "; ".join(decision.modifications))

        action, qty = self._decide_action(symbol, final_signal, portfolio_state.equity)
        self._record_signal(
            now, symbol, action,
            final_signal.reasoning if action != "hold" else f"already at target allocation ({final_signal.reasoning})",
        )

        order_result = None
        if action != "hold" and not self.dry_run:
            order_result = self._execute_action(symbol, final_signal, action, qty, now)

        return BarDecision(symbol, regime_state, final_signal, True, decision.modifications, None, order_result)

    def _decide_action(self, symbol: str, signal: Signal, equity: float) -> tuple[str, Optional[int]]:
        """Decide hold / enter / rebalance from the CURRENT tracked position,
        not just from the signal — this is what prevents double-entry: an
        approved signal every bar must NOT mean a new order every bar.
        """
        target_allocation = signal.position_size_pct * signal.leverage
        existing = self.position_tracker.get_position(symbol)

        if existing is None:
            qty = int(target_allocation * equity / signal.entry_price)
            return ("enter", qty) if qty > 0 else ("hold", None)

        current_allocation = (existing.qty * existing.current_price) / equity if equity else 0.0
        if not self.orchestrator.needs_rebalance(current_allocation, target_allocation):
            return "hold", None

        qty = int(target_allocation * equity / signal.entry_price)
        return ("rebalance", qty) if qty > 0 else ("hold", None)

    def _execute_action(self, symbol: str, signal: Signal, action: str, qty: int, now: datetime):
        if action == "rebalance":
            # Close-then-reopen rather than trading the delta: simpler and
            # safer to reason about than juggling two live brackets (old
            # stop + new stop) on the same symbol. Costs one extra round
            # trip on a rebalance; a future improvement could trade the
            # delta directly and just replace the stop/take-profit legs.
            self.order_executor.close_position(symbol)
            self.position_tracker.positions.pop(symbol, None)

        order_result = self.order_executor.submit_bracket_order(signal, qty, trade_id=str(uuid.uuid4()))
        self.position_tracker.open_position(
            symbol, qty=qty, entry_price=signal.entry_price, stop_level=signal.stop_loss,
            regime_at_entry=signal.regime_name, trade_id=order_result.trade_id,
        )
        self.recent_orders.append((symbol, signal.direction, now))
        self.recent_orders = self.recent_orders[-50:]
        self.daily_trades_count += 1
        return order_result

    def _infer_regime(self, symbol: str, bars: pd.DataFrame) -> Optional[RegimeState]:
        try:
            features = compute_features(bars)
            if features.empty:
                raise ValueError("not enough bars yet for a valid standardized feature row")
            regime_state = self.hmm_engine.update(features)
        except Exception as exc:  # noqa: BLE001 - HMM error: hold current regime, never crash the loop
            logger.error("HMM error for %s, holding last known regime: %s", symbol, exc)
            return self.last_regime_state.get(symbol)

        previous = self.last_regime_state.get(symbol)
        if previous is not None and previous.label != regime_state.label and regime_state.is_confirmed:
            self.alert_manager.send_alert(
                AlertType.REGIME_CHANGE,
                f"{symbol}: regime {previous.label} -> {regime_state.label} ({regime_state.probability:.0%})",
                AlertSeverity.INFO, context={"symbol": symbol, "regime": regime_state.label},
            )
        self.last_regime_state[symbol] = regime_state
        return regime_state

    def update_trailing_stops(self, symbol: str, bars: pd.DataFrame, regime_state: Optional[RegimeState]) -> None:
        """Step 8 of MAIN LOOP: re-derive each held position's stop from its
        regime's own strategy and tighten it if the new one is higher.

        Reuses BaseStrategy.generate_signal's stop computation instead of
        duplicating ATR/EMA math here — order_executor.modify_stop() already
        refuses to widen, so calling this every bar is always safe.
        """
        if self.dry_run or regime_state is None or self.orchestrator is None:
            return
        position = self.position_tracker.get_position(symbol)
        if position is None:
            return
        strategy = self.orchestrator.strategy_by_state.get(regime_state.state_id)
        if strategy is None:
            return
        fresh_signal = strategy.generate_signal(symbol, bars, regime_state)
        if fresh_signal is None:
            return
        if self.order_executor.modify_stop(symbol, fresh_signal.stop_loss):
            self.position_tracker.update_stop(symbol, fresh_signal.stop_loss)
        self.position_tracker.update_current_regime(symbol, regime_state.label)

    def force_liquidate_if_halted(self, now: Optional[datetime] = None) -> bool:
        """Step 9 of MAIN LOOP: if a HALT-level circuit breaker just fired,
        close everything, per the guide's "close ALL positions" rule.
        """
        now = now or datetime.now()
        portfolio_state = self._build_portfolio_state()
        status = self.risk_manager.check_circuit_breaker(portfolio_state, now=now)
        if status not in (CircuitBreakerStatus.DAILY_HALT, CircuitBreakerStatus.WEEKLY_HALT, CircuitBreakerStatus.PEAK_HALT):
            return False

        self.alert_manager.send_alert(
            AlertType.CIRCUIT_BREAKER, f"Circuit breaker {status.value} -- closing all positions",
            AlertSeverity.CRITICAL, context={"status": status.value, "equity": portfolio_state.equity}, now=now,
        )
        if not self.dry_run:
            self.order_executor.close_all_positions()
        return True

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _roll_daily_counters(self, now: datetime) -> None:
        today = now.date()
        if self._trade_day != today:
            self._trade_day = today
            self.daily_trades_count = 0

    def _record_signal(self, now: datetime, symbol: str, action: str, reason: Optional[str]) -> None:
        self.recent_signals.append({"time": now.strftime("%H:%M"), "symbol": symbol, "action": action, "reason": reason})
        self.recent_signals = self.recent_signals[-20:]

    def _build_portfolio_state(self) -> PortfolioState:
        account = self.alpaca_client.get_account()
        return self.position_tracker.to_portfolio_state(
            equity=account.equity, cash=account.cash, buying_power=account.buying_power,
            daily_trades_count=self.daily_trades_count, recent_orders=list(self.recent_orders),
        )

    # ------------------------------------------------------------------
    # State snapshot (recovery across restarts)
    # ------------------------------------------------------------------

    def save_state_snapshot(self) -> None:
        cb = self.risk_manager.circuit_breaker
        payload = {
            "saved_at": datetime.now().isoformat(),
            "last_train_date": self.last_train_date.isoformat() if self.last_train_date else None,
            "daily_trades_count": self.daily_trades_count,
            "trade_day": self._trade_day.isoformat() if self._trade_day else None,
            "recent_signals": self.recent_signals,
            "circuit_breaker": {
                "day_start_equity": cb.day_start_equity,
                "week_start_equity": cb.week_start_equity,
                "peak_equity": cb.peak_equity,
                "current_day": cb._current_day.isoformat() if cb._current_day else None,
                "current_week": list(cb._current_week) if cb._current_week else None,
            },
            "positions_metadata": {
                symbol: {
                    "entry_time": p.entry_time.isoformat(),
                    "stop_level": p.stop_level,
                    "regime_at_entry": p.regime_at_entry,
                    "trade_id": p.trade_id,
                    "sector": p.sector,
                }
                for symbol, p in self.position_tracker.positions.items()
            },
        }
        with open(self.state_path, "w") as f:
            json.dump(payload, f, indent=2)
        logger.info("Saved state snapshot to %s", self.state_path)

    def _load_state_snapshot(self) -> bool:
        if not os.path.exists(self.state_path):
            return False
        with open(self.state_path) as f:
            payload = json.load(f)

        if payload.get("last_train_date"):
            self.last_train_date = datetime.fromisoformat(payload["last_train_date"])
        self.daily_trades_count = payload.get("daily_trades_count", 0)
        self._trade_day = date.fromisoformat(payload["trade_day"]) if payload.get("trade_day") else None
        self.recent_signals = payload.get("recent_signals", [])

        cb_data = payload.get("circuit_breaker", {})
        cb = self.risk_manager.circuit_breaker
        cb.day_start_equity = cb_data.get("day_start_equity")
        cb.week_start_equity = cb_data.get("week_start_equity")
        cb.peak_equity = cb_data.get("peak_equity")
        cb._current_day = date.fromisoformat(cb_data["current_day"]) if cb_data.get("current_day") else None
        cb._current_week = tuple(cb_data["current_week"]) if cb_data.get("current_week") else None

        for symbol, meta in payload.get("positions_metadata", {}).items():
            position = self.position_tracker.get_position(symbol)
            if position is not None:
                position.entry_time = datetime.fromisoformat(meta["entry_time"])
                position.stop_level = meta.get("stop_level")
                position.regime_at_entry = meta.get("regime_at_entry")
                position.trade_id = meta.get("trade_id")
                position.sector = meta.get("sector")

        logger.info("Recovered state snapshot from %s (saved_at=%s)", self.state_path, payload.get("saved_at"))
        return True

    # ------------------------------------------------------------------
    # Dashboard state
    # ------------------------------------------------------------------

    def get_dashboard_state(self) -> dict:
        """Build the plain dict monitoring.dashboard.Dashboard.render() expects."""
        account = self.alpaca_client.get_account()
        cb = self.risk_manager.circuit_breaker
        positions = self.position_tracker.get_all_positions()
        any_regime = next(iter(self.last_regime_state.values()), None)

        daily_dd = 1 - account.equity / cb.day_start_equity if cb.day_start_equity else 0.0
        peak_dd = 1 - account.equity / cb.peak_equity if cb.peak_equity else 0.0

        return {
            "regime": {
                "label": any_regime.label if any_regime else "?",
                "probability": any_regime.probability if any_regime else 0.0,
                "stability_bars": any_regime.consecutive_bars if any_regime else 0,
                "flicker_rate": self.hmm_engine.get_regime_flicker_rate() if self.hmm_engine else 0,
                "flicker_window": self.config["hmm"].get("flicker_window", 20),
            },
            "portfolio": {"equity": account.equity, "daily_pnl": 0.0, "daily_pnl_pct": -daily_dd, "allocation": 0.0, "leverage": 1.0},
            "positions": [
                {
                    "symbol": p.symbol, "direction": "LONG", "price": p.current_price,
                    "pnl_pct": p.unrealized_pnl_pct, "stop": p.stop_level or 0.0,
                    "held": str(p.holding_period()).split(".")[0],
                }
                for p in positions
            ],
            "recent_signals": self.recent_signals,
            "risk": {
                "daily_dd": max(daily_dd, 0.0), "daily_dd_limit": self.config["risk"]["daily_dd_halt"],
                "peak_dd": max(peak_dd, 0.0), "peak_dd_limit": self.config["risk"]["max_dd_from_peak"],
            },
            "system": {
                "data_ok": True, "api_ok": True, "api_latency_ms": 0,
                "hmm_age": f"{(datetime.now() - self.last_train_date).days}d ago" if self.last_train_date else "never",
                "paper": self.alpaca_client.paper,
            },
        }

    # ------------------------------------------------------------------
    # Live dashboard
    # ------------------------------------------------------------------

    def start_dashboard(self, refresh_seconds: Optional[float] = None):
        """Launch the live terminal dashboard in a background daemon thread,
        repeatedly rendering get_dashboard_state(). Returns the started
        Thread; it exits automatically when the main process does (daemon
        thread), and run()/shutdown() don't need to manage it explicitly.

        Runs in its OWN thread rather than the main one because both this
        and run() block forever in their own way (rich.live.Live's refresh
        loop here, market_data.run_stream()'s WebSocket loop there) — they
        have to share the process, not take turns.
        """
        import threading

        from monitoring.dashboard import Dashboard

        dashboard = Dashboard(self.config.get("monitoring"))
        thread = threading.Thread(
            target=dashboard.run, args=(self.get_dashboard_state,), kwargs={"refresh_seconds": refresh_seconds},
            daemon=True, name="regime-trader-dashboard",
        )
        thread.start()
        return thread

    def write_dashboard_snapshot(self, path: str = DEFAULT_DASHBOARD_JSON_PATH) -> None:
        """Dump get_dashboard_state() to disk, atomically, for an out-of-process
        UI (streamlit_app.py) that has no access to this object's in-memory
        state. Same rationale as save_state_snapshot(), but for display rather
        than crash recovery, so it's written much more frequently.
        """
        tmp_path = f"{path}.tmp"
        with open(tmp_path, "w") as f:
            json.dump(self.get_dashboard_state(), f, indent=2, default=str)
        os.replace(tmp_path, path)

    def start_json_dashboard_feed(
        self, path: str = DEFAULT_DASHBOARD_JSON_PATH, refresh_seconds: Optional[float] = None
    ):
        """Launch a background daemon thread that periodically writes
        write_dashboard_snapshot(). This is how streamlit_app.py (a separate
        process, polling the file) sees near-live state without any direct
        IPC channel into this running bot -- same constraint start_dashboard()
        works around for the in-terminal rich UI, just via a file instead of
        a shared Python object.
        """
        import threading

        refresh_seconds = refresh_seconds or self.config.get("monitoring", {}).get("dashboard_refresh_seconds", 5)

        def _loop() -> None:
            while True:
                self.write_dashboard_snapshot(path)
                time.sleep(refresh_seconds)

        thread = threading.Thread(target=_loop, daemon=True, name="regime-trader-json-feed")
        thread.start()
        return thread

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        logger.info("Shutting down -- positions are NOT closed, stops remain in place")
        self.market_data.stop_stream()
        self.position_tracker.stop_streaming()
        self.save_state_snapshot()
        self._print_session_summary()

    def _print_session_summary(self) -> None:
        positions = self.position_tracker.get_all_positions()
        print("\n=== Session summary ===")
        print(f"Open positions: {len(positions)}")
        print(f"Total unrealized P&L: ${self.position_tracker.total_unrealized_pnl():,.2f}")
        print(f"Signals processed this session: {len(self.recent_signals)}")

    # ------------------------------------------------------------------
    # Live run — NOT unit-testable without a real/live feed (see module docstring)
    # ------------------------------------------------------------------

    def run(self, symbols: list[str], timeframe: str = "5Min", market_check_interval_seconds: int = 30) -> None:
        """Blocking: waits for market hours, starts streaming, and processes
        bars as they arrive via WebSocket until stop() is called or the
        process receives SIGINT/SIGTERM.
        """
        self._running = True
        signal.signal(signal.SIGINT, lambda *_: self.stop())
        signal.signal(signal.SIGTERM, lambda *_: self.stop())

        while self._running and not self.alpaca_client.is_market_open():
            logger.info("Market closed, waiting %ds...", market_check_interval_seconds)
            time.sleep(market_check_interval_seconds)
        if not self._running:
            return

        self.startup()

        async def _on_bar(bar) -> None:
            symbol = bar.symbol
            try:
                bars = self.market_data.get_historical_bars(symbol, timeframe=timeframe, limit=500)
                decision = self.process_bar(symbol, bars)
                self.update_trailing_stops(symbol, bars, decision.regime_state)
                self.force_liquidate_if_halted()
                self.maybe_retrain_weekly(compute_features(bars), log_returns(bars["close"], 1))
            except Exception:  # noqa: BLE001 - never let one bad bar kill the loop
                logger.error("Unhandled error processing bar for %s:\n%s", symbol, traceback.format_exc())
                self.save_state_snapshot()
                self.alert_manager.send_alert(
                    AlertType.API_LOST, f"Unhandled error processing {symbol} (see main.log)", AlertSeverity.CRITICAL,
                )

        self.market_data.subscribe_bars(symbols, _on_bar, timeframe=timeframe)
        try:
            self.market_data.run_stream()
        finally:
            self.shutdown()

    def stop(self) -> None:
        self._running = False
        self.market_data.stop_stream()
