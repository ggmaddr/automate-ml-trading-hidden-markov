"""Typed errors for the research layer.

Every failure the research sidecar can hit maps to one of these, each with
a stable `code` and a user-facing `user_message`. The orchestrator converts
them into a `ResearchOutcome` error state — they never propagate into the
trading system.
"""

from __future__ import annotations


class ResearchError(Exception):
    """Base class. `code` is stable and machine-readable; `retryable` tells
    the UI whether a retry/refresh is worth offering."""

    code = "research_error"
    retryable = False
    fail_fast = False            # True: never retry inside a call (e.g. server is simply down)
    show_detail = False          # True: the (redacted) detail is actionable and shown to the user
    user_message = "Research failed."

    def __init__(self, detail: str = "") -> None:
        self.detail = detail
        super().__init__(detail or self.user_message)

    @property
    def message(self) -> str:
        """User-facing text: the stable message, plus the detail when it tells the user what to fix."""
        if self.show_detail and self.detail:
            return f"{self.user_message} [{self.detail}]"
        return self.user_message


class ResearchDisabledError(ResearchError):
    code = "disabled"
    user_message = "AI research is disabled (research.enabled is false in config/settings.yaml)."


class ResearchNotConfiguredError(ResearchError):
    show_detail = True
    code = "not_configured"
    user_message = "Research is not configured."


class ResearchAuthError(ResearchError):
    code = "auth_failed"
    user_message = "Gemini rejected the API key (authentication/permission error)."


class ResearchModelError(ResearchError):
    show_detail = True
    code = "bad_request"
    user_message = "Gemini rejected the request (check research.model / GEMINI_MODEL)."


class ResearchQuotaError(ResearchError):
    code = "quota_exceeded"
    retryable = True
    user_message = "Research temporarily unavailable: Gemini quota or rate limit exceeded. Quantitative system continues normally."


class ResearchBudgetError(ResearchError):
    code = "budget_exceeded"
    user_message = "Monthly research budget reached (local estimate). Raise research.monthly_budget_usd to continue; cached results still work."


class ResearchUnavailableError(ResearchError):
    code = "unavailable"
    retryable = True
    user_message = "Research temporarily unavailable. Quantitative system continues normally."


class LLMUnavailableError(ResearchUnavailableError):
    show_detail = True
    code = "llm_unavailable"
    fail_fast = True
    user_message = (
        "The research language model is not reachable. If you use Ollama: run `ollama serve` and "
        "`ollama pull <model>`. Quantitative system continues normally."
    )


class SearchUnavailableError(ResearchUnavailableError):
    show_detail = True
    code = "search_unavailable"
    fail_fast = True
    user_message = (
        "The web search service is not reachable. If you use SearXNG: run "
        "`docker compose -f ops/docker-compose.searxng.yml up -d`. Quantitative system continues normally."
    )


class ResearchTimeoutError(ResearchError):
    code = "timeout"
    retryable = True
    user_message = "Research request timed out. Quantitative system continues normally."


class SchemaValidationError(ResearchError):
    code = "schema_validation_failed"
    retryable = True
    user_message = "Gemini returned malformed structured output after retries; no report was produced."


class ToolError(ResearchError):
    code = "tool_error"
    user_message = "An internal research tool failed."


class QuantContextUnavailableError(ToolError):
    code = "quant_context_unavailable"
    user_message = "The HMM snapshot for this symbol could not be loaded."


class ToolRejectedError(ToolError):
    code = "tool_rejected"
    user_message = "Requested tool is not on the research allowlist."


class ToolBudgetExceededError(ToolError):
    code = "tool_budget_exceeded"
    user_message = "Internal tool-call budget exhausted."
