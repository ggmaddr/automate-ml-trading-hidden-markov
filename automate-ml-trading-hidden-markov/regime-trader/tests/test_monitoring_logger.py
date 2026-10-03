"""Tests for monitoring.logger. Always points log_dir at tmp_path so the
real project logs/ directory is never touched or polluted by tests.
"""

from __future__ import annotations

import json

from monitoring.logger import get_alerts_logger, get_logger, get_main_logger, get_regime_logger, get_trades_logger


def _flush(logger):
    for handler in logger.handlers:
        handler.flush()


def test_get_logger_writes_a_json_line(tmp_path):
    logger = get_logger("test.logger.json_line", "main.log", log_dir=str(tmp_path))
    logger.info("hello", extra={"regime": "BULL", "equity": 1000})
    _flush(logger)

    lines = (tmp_path / "main.log").read_text().strip().splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["message"] == "hello"
    assert payload["regime"] == "BULL"
    assert payload["equity"] == 1000
    assert payload["level"] == "INFO"
    assert "timestamp" in payload


def test_get_logger_is_idempotent_for_same_name_and_file(tmp_path):
    logger1 = get_logger("test.logger.idempotent", "main.log", log_dir=str(tmp_path))
    logger2 = get_logger("test.logger.idempotent", "main.log", log_dir=str(tmp_path))
    assert logger1 is logger2
    file_handlers = [h for h in logger1.handlers if hasattr(h, "baseFilename")]
    assert len(file_handlers) == 1  # no duplicate handler stacking


def test_exception_info_is_captured(tmp_path):
    logger = get_logger("test.logger.exception", "main.log", log_dir=str(tmp_path))
    try:
        raise ValueError("boom")
    except ValueError:
        logger.exception("something broke")
    _flush(logger)

    payload = json.loads((tmp_path / "main.log").read_text().strip().splitlines()[-1])
    assert "exception" in payload
    assert "ValueError" in payload["exception"]


def test_convenience_loggers_write_to_expected_files(tmp_path):
    main = get_main_logger(log_dir=str(tmp_path), console=False)
    trades = get_trades_logger(log_dir=str(tmp_path))
    alerts = get_alerts_logger(log_dir=str(tmp_path))
    regime = get_regime_logger(log_dir=str(tmp_path))

    main.info("m")
    trades.info("t")
    alerts.info("a")
    regime.info("r")
    for lg in (main, trades, alerts, regime):
        _flush(lg)

    assert (tmp_path / "main.log").exists()
    assert (tmp_path / "trades.log").exists()
    assert (tmp_path / "alerts.log").exists()
    assert (tmp_path / "regime.log").exists()
