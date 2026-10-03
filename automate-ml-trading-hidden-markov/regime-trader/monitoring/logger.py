"""Structured JSON logging with rotating files.

Four separate log streams, each its own rotating file:
    main.log    - general system/lifecycle events
    trades.log  - every order submission/fill/rejection
    alerts.log  - every triggered alert
    regime.log  - every regime read/change

Rotation: 10MB per file, up to 30 rotated backups. ("10MB, 30 days" in the
guide mixes a size-based and a time-based rotation scheme; this picks
size-based rotation with enough backups to cover roughly a month of
typical use — a pragmatic reading, documented here rather than silently
picked.)

Any extra context (regime, probability, equity, positions, daily_pnl, ...)
is passed via the standard `logger.info(msg, extra={...})` mechanism and
merged into the JSON line — callers attach whatever's relevant to THAT
event rather than every log call being forced to populate every field.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import os
from datetime import datetime
from typing import Optional

DEFAULT_LOG_DIR = "logs"
MAX_BYTES = 10 * 1024 * 1024  # 10MB
BACKUP_COUNT = 30

_RESERVED_RECORD_ATTRS = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__.keys()) | {"message", "asctime"}


class JsonFormatter(logging.Formatter):
    """Renders each log record as one JSON line."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        for key, value in record.__dict__.items():
            if key not in _RESERVED_RECORD_ATTRS and key not in payload:
                payload[key] = value
        return json.dumps(payload, default=str)


def get_logger(
    name: str,
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    console: bool = False,
    log_dir: str = DEFAULT_LOG_DIR,
) -> logging.Logger:
    """Return a logger writing JSON lines to `<log_dir>/<log_file>`, rotated at 10MB.

    Safe to call repeatedly with the same (name, log_file) — won't stack
    duplicate handlers.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    if log_file:
        os.makedirs(log_dir, exist_ok=True)
        path = os.path.abspath(os.path.join(log_dir, log_file))
        already_attached = any(
            isinstance(h, logging.handlers.RotatingFileHandler) and h.baseFilename == path for h in logger.handlers
        )
        if not already_attached:
            handler = logging.handlers.RotatingFileHandler(path, maxBytes=MAX_BYTES, backupCount=BACKUP_COUNT)
            handler.setFormatter(JsonFormatter())
            logger.addHandler(handler)

    if console:
        has_console = any(
            isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler) for h in logger.handlers
        )
        if not has_console:
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
            logger.addHandler(stream_handler)

    return logger


def get_main_logger(log_dir: str = DEFAULT_LOG_DIR, console: bool = True) -> logging.Logger:
    return get_logger("regime_trader.main", "main.log", console=console, log_dir=log_dir)


def get_trades_logger(log_dir: str = DEFAULT_LOG_DIR) -> logging.Logger:
    return get_logger("regime_trader.trades", "trades.log", log_dir=log_dir)


def get_alerts_logger(log_dir: str = DEFAULT_LOG_DIR) -> logging.Logger:
    return get_logger("regime_trader.alerts", "alerts.log", log_dir=log_dir)


def get_regime_logger(log_dir: str = DEFAULT_LOG_DIR) -> logging.Logger:
    return get_logger("regime_trader.regime", "regime.log", log_dir=log_dir)
