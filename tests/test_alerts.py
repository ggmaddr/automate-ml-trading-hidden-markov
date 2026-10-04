"""Tests for monitoring.alerts. Email/webhook channels are tested via
mocked smtplib/urllib — no real network access required.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import patch

from monitoring.alerts import AlertManager, AlertSeverity, AlertType

CONFIG = {"alert_rate_limit_minutes": 15}


def test_send_alert_delivers_and_records_history(capsys):
    manager = AlertManager(CONFIG)
    sent = manager.send_alert(AlertType.REGIME_CHANGE, "SPY regime changed", AlertSeverity.INFO)
    assert sent is True
    assert len(manager.get_history()) == 1
    assert "regime_change" in capsys.readouterr().out


def test_rate_limiting_blocks_repeat_within_window():
    manager = AlertManager(CONFIG)
    now = datetime(2024, 1, 1, 10, 0)
    assert manager.send_alert(AlertType.CIRCUIT_BREAKER, "a", now=now) is True
    assert manager.send_alert(AlertType.CIRCUIT_BREAKER, "b", now=now + timedelta(minutes=5)) is False
    assert manager.send_alert(AlertType.CIRCUIT_BREAKER, "c", now=now + timedelta(minutes=16)) is True


def test_rate_limiting_is_independent_per_event_type():
    manager = AlertManager(CONFIG)
    now = datetime(2024, 1, 1, 10, 0)
    assert manager.send_alert(AlertType.CIRCUIT_BREAKER, "a", now=now) is True
    assert manager.send_alert(AlertType.REGIME_CHANGE, "b", now=now) is True


def test_email_not_attempted_when_not_configured():
    manager = AlertManager(CONFIG)
    with patch("monitoring.alerts.smtplib.SMTP") as mock_smtp:
        manager.send_alert(AlertType.LARGE_PNL, "big move")
        mock_smtp.assert_not_called()


def test_email_delivery_attempted_when_configured():
    email_config = {
        "smtp_host": "smtp.example.com", "smtp_port": 587, "username": "bot@example.com",
        "password": "pw", "to_addresses": ["me@example.com"],
    }
    manager = AlertManager(CONFIG, email_config=email_config)
    with patch("monitoring.alerts.smtplib.SMTP") as mock_smtp:
        instance = mock_smtp.return_value.__enter__.return_value
        manager.send_alert(AlertType.LARGE_PNL, "big move")
        mock_smtp.assert_called_once_with("smtp.example.com", 587)
        instance.login.assert_called_once()
        instance.send_message.assert_called_once()


def test_email_failure_does_not_raise():
    email_config = {"smtp_host": "x", "smtp_port": 1, "username": "a", "password": "b", "to_addresses": ["c"]}
    manager = AlertManager(CONFIG, email_config=email_config)
    with patch("monitoring.alerts.smtplib.SMTP", side_effect=OSError("down")):
        sent = manager.send_alert(AlertType.LARGE_PNL, "big move")
    assert sent is True  # console+log still succeed even though the email channel failed


def test_webhook_delivery_attempted_when_configured():
    webhook_config = {"url": "https://hooks.example.com/x"}
    manager = AlertManager(CONFIG, webhook_config=webhook_config)
    with patch("monitoring.alerts.urllib.request.urlopen") as mock_urlopen:
        manager.send_alert(AlertType.DATA_FEED_DOWN, "feed down")
        mock_urlopen.assert_called_once()


def test_webhook_failure_does_not_raise():
    webhook_config = {"url": "https://hooks.example.com/x"}
    manager = AlertManager(CONFIG, webhook_config=webhook_config)
    with patch("monitoring.alerts.urllib.request.urlopen", side_effect=OSError("down")):
        sent = manager.send_alert(AlertType.DATA_FEED_DOWN, "feed down")
    assert sent is True
