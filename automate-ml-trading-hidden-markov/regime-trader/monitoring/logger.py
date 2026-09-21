"""Structured logging configuration for regime-trader."""

from __future__ import annotations

import logging
from typing import Optional


def get_logger(name: str, level: int = logging.INFO, log_file: Optional[str] = None) -> logging.Logger:
    """Return a configured structured logger.

    Regime changes should be logged as WARNING; confirmations as INFO.
    """
    raise NotImplementedError
