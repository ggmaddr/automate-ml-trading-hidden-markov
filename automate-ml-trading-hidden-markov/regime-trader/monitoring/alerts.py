"""Email/webhook alerts for critical events (regime flips, drawdown halts)."""

from __future__ import annotations

from enum import Enum


class AlertSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertManager:
    """Sends rate-limited alerts via email and/or webhook."""

    def __init__(self, config: dict) -> None:
        raise NotImplementedError

    def send_alert(self, message: str, severity: AlertSeverity) -> bool:
        """Send an alert, respecting alert_rate_limit_minutes per alert kind."""
        raise NotImplementedError

    def send_email(self, subject: str, body: str) -> bool:
        raise NotImplementedError

    def send_webhook(self, payload: dict) -> bool:
        raise NotImplementedError
