"""Structured event logging + secret redaction for the research layer.

Events are emitted on the `regime_trader.research` logger. File output is
opt-in: entrypoints (CLI / Streamlit) call `monitoring.logger.get_research_logger()`
to attach the rotating JSON handler; importing this module never creates
files.
"""

from __future__ import annotations

import logging
import os
import re

LOGGER_NAME = "regime_trader.research"
_logger = logging.getLogger(LOGGER_NAME)

# Env vars whose VALUES must never reach a log line or a prompt.
_SECRET_ENV_VARS = (
    "GEMINI_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "SERPAPI_API_KEY", "ALPACA_API_KEY", "ALPACA_SECRET_KEY",
)
_KEY_PATTERN = re.compile(r"(?:AIza[0-9A-Za-z_\-]{20,}|sk-(?:ant-)?[0-9A-Za-z_\-]{20,})")


def redact(text: object) -> str:
    """Replace known secret values (and anything shaped like a Google API key) with ***."""
    out = str(text)
    for var in _SECRET_ENV_VARS:
        value = os.environ.get(var)
        if value and len(value) >= 6:
            out = out.replace(value, "***")
    return _KEY_PATTERN.sub("***", out)


def log_event(event: str, level: int = logging.INFO, **fields: object) -> None:
    """Emit one structured research event. String fields are redacted."""
    safe = {k: (redact(v) if isinstance(v, str) else v) for k, v in fields.items()}
    _logger.log(level, event, extra={"event": event, **safe})
