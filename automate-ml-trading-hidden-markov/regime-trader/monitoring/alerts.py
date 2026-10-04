"""Alerts for critical events: regime change, circuit breaker, large P&L,
data feed down, API lost, HMM retrained, flicker exceeded.

Delivery: console + log file always; email/webhook only if the caller passes
`email_config`/`webhook_config` to AlertManager (e.g. built from `.env`
variables, never hardcoded). Each event TYPE is rate-limited independently
to at most one delivery per
`alert_rate_limit_minutes` (default 15, from settings.yaml's monitoring
section), so a storm of one kind of alert can't bury a different kind.
"""

from __future__ import annotations

import logging
import smtplib
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from enum import Enum
from typing import Optional

logger = logging.getLogger("regime_trader.alerts")


class AlertSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertType(Enum):
    REGIME_CHANGE = "regime_change"
    CIRCUIT_BREAKER = "circuit_breaker"
    LARGE_PNL = "large_pnl"
    DATA_FEED_DOWN = "data_feed_down"
    API_LOST = "api_lost"
    HMM_RETRAINED = "hmm_retrained"
    FLICKER_EXCEEDED = "flicker_exceeded"


@dataclass
class Alert:
    alert_type: AlertType
    severity: AlertSeverity
    message: str
    timestamp: datetime
    context: dict = field(default_factory=dict)


class AlertManager:
    """Sends rate-limited alerts via console, log, and optionally email/webhook.

    `config` is the `monitoring:` section of settings.yaml. `email_config` /
    `webhook_config` are plain dicts the caller builds (e.g. from `.env`
    variables) — left None to skip that channel; see _deliver_email() /
    _deliver_webhook() below for the keys each one expects.
    """

    def __init__(
        self,
        config: dict,
        email_config: Optional[dict] = None,
        webhook_config: Optional[dict] = None,
        logger_: Optional[logging.Logger] = None,
    ) -> None:
        self.config = config
        self.email_config = email_config
        self.webhook_config = webhook_config
        self.logger = logger_ or logger
        self._last_sent: dict[AlertType, datetime] = {}
        self.history: list[Alert] = []

    def _is_rate_limited(self, alert_type: AlertType, now: datetime) -> bool:
        last = self._last_sent.get(alert_type)
        if last is None:
            return False
        window = timedelta(minutes=self.config.get("alert_rate_limit_minutes", 15))
        return now - last < window

    def send_alert(
        self,
        alert_type: AlertType,
        message: str,
        severity: AlertSeverity = AlertSeverity.WARNING,
        context: Optional[dict] = None,
        now: Optional[datetime] = None,
    ) -> bool:
        """Deliver an alert unless this alert_type was sent too recently. Returns whether it was sent."""
        now = now or datetime.now()
        if self._is_rate_limited(alert_type, now):
            return False

        alert = Alert(alert_type=alert_type, severity=severity, message=message, timestamp=now, context=context or {})
        self.history.append(alert)
        self._last_sent[alert_type] = now

        self._deliver_console(alert)
        self._deliver_log(alert)
        if self.email_config:
            self._deliver_email(alert)
        if self.webhook_config:
            self._deliver_webhook(alert)
        return True

    def get_history(self) -> list[Alert]:
        return list(self.history)

    # ------------------------------------------------------------------
    # Delivery channels
    # ------------------------------------------------------------------

    def _deliver_console(self, alert: Alert) -> None:
        print(f"[{alert.severity.value.upper()}] {alert.alert_type.value}: {alert.message}")

    def _deliver_log(self, alert: Alert) -> None:
        log_fn = {
            AlertSeverity.INFO: self.logger.info,
            AlertSeverity.WARNING: self.logger.warning,
            AlertSeverity.CRITICAL: self.logger.critical,
        }[alert.severity]
        log_fn(alert.message, extra={"alert_type": alert.alert_type.value, **alert.context})

    def _deliver_email(self, alert: Alert) -> None:
        try:
            msg = MIMEText(alert.message)
            msg["Subject"] = f"[regime-trader] {alert.alert_type.value}"
            msg["From"] = self.email_config["username"]
            msg["To"] = ", ".join(self.email_config["to_addresses"])
            with smtplib.SMTP(self.email_config["smtp_host"], self.email_config["smtp_port"]) as server:
                server.starttls()
                server.login(self.email_config["username"], self.email_config["password"])
                server.send_message(msg)
        except Exception as exc:  # noqa: BLE001 - an alert channel failing must never crash the bot
            self.logger.warning("Failed to send alert email: %s", exc)

    def _deliver_webhook(self, alert: Alert) -> None:
        try:
            import json as json_module

            data = json_module.dumps(
                {
                    "alert_type": alert.alert_type.value,
                    "severity": alert.severity.value,
                    "message": alert.message,
                    "timestamp": alert.timestamp.isoformat(),
                    "context": alert.context,
                }
            ).encode()
            request = urllib.request.Request(
                self.webhook_config["url"], data=data, headers={"Content-Type": "application/json"},
            )
            urllib.request.urlopen(request, timeout=5)
        except Exception as exc:  # noqa: BLE001 - an alert channel failing must never crash the bot
            self.logger.warning("Failed to send alert webhook: %s", exc)
