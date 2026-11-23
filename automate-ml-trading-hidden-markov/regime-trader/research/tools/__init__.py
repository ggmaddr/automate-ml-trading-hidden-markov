"""Read-only internal tools + the allowlisted dispatcher.

SECURITY BOUNDARY: this package must never import broker, order-execution,
risk-manager, strategy, or trading-bot modules. `ALLOWED_RESEARCH_TOOLS` is
the complete list of callable tools; the registry refuses anything else by
name and never resolves a function dynamically (no getattr/eval). The
guarantees are enforced by tests/test_research_permissions.py.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Optional

from research.exceptions import ToolBudgetExceededError, ToolError, ToolRejectedError
from research.logging_util import log_event, redact
from research.schemas import ToolCallRecord
from research.tools.backtest_tools import BacktestReader
from research.tools.hmm_tools import HMMRuntimeReader

ALLOWED_RESEARCH_TOOLS = frozenset({
    "get_hmm_snapshot",
    "get_regime_history",
    "get_backtest_summary",
})


class ResearchToolRegistry:
    """Executes allowlisted tools on the application's behalf, with a call budget.

    One instance per research request: it owns that request's budget and
    its trace (`calls`).
    """

    def __init__(
        self,
        hmm_reader: HMMRuntimeReader,
        backtest_reader: BacktestReader,
        max_calls: int = 5,
        request_id: str = "",
    ) -> None:
        # Explicit name -> bound method table. This is the ONLY place tools are bound.
        self._table: dict[str, Callable[..., Any]] = {
            "get_hmm_snapshot": hmm_reader.get_hmm_snapshot,
            "get_regime_history": hmm_reader.get_regime_history,
            "get_backtest_summary": backtest_reader.get_backtest_summary,
        }
        assert set(self._table) <= ALLOWED_RESEARCH_TOOLS
        self.max_calls = max_calls
        self.request_id = request_id
        self.calls: list[ToolCallRecord] = []

    @property
    def tool_names(self) -> frozenset[str]:
        return frozenset(self._table)

    def call(self, name: str, **kwargs: Any) -> Any:
        """Run tool `name`. Raises ToolRejectedError for anything off-allowlist
        (including names Gemini or a caller might invent), ToolBudgetExceededError
        past the budget, ToolError for tool failures."""
        if name not in ALLOWED_RESEARCH_TOOLS or name not in self._table:
            log_event("research.tool.rejected", level=30, request_id=self.request_id, tool_name=str(name)[:80])
            raise ToolRejectedError(f"tool {str(name)[:80]!r} is not allowed")
        if len(self.calls) >= self.max_calls:
            raise ToolBudgetExceededError(f"max_internal_tool_calls={self.max_calls} reached")

        started = time.monotonic()
        log_event("research.tool.called", request_id=self.request_id, tool_name=name)
        try:
            result = self._table[name](**kwargs)
        except Exception as exc:  # noqa: BLE001 - record then re-raise as a controlled error
            self.calls.append(ToolCallRecord(
                tool=name, args=kwargs, ok=False, error=redact(f"{type(exc).__name__}: {exc}")[:300],
                duration_ms=int((time.monotonic() - started) * 1000),
            ))
            if isinstance(exc, ToolError) or isinstance(getattr(exc, "__cause__", None), ToolError):
                raise
            raise ToolError(f"{name} failed: {type(exc).__name__}") from exc
        self.calls.append(ToolCallRecord(
            tool=name, args=kwargs, ok=True, duration_ms=int((time.monotonic() - started) * 1000),
        ))
        return result
