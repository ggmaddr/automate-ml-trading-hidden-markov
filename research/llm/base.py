"""Provider-neutral LLM plumbing: the `LLMBackend` contract plus the shared
retry / deadline / validate-or-repair logic every adapter reuses.

Adding a provider = one adapter module (see ollama.py) + one line in
`research/llm/__init__.py::PROVIDERS`. Nothing else in `research/` knows which
provider is active.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional, Protocol, Type, TypeVar, runtime_checkable

import httpx
from pydantic import BaseModel, ValidationError

from research.config import ResearchConfig
from research.exceptions import (
    LLMUnavailableError,
    ResearchAuthError,
    ResearchError,
    ResearchModelError,
    ResearchQuotaError,
    ResearchTimeoutError,
    ResearchUnavailableError,
    SchemaValidationError,
)
from research.logging_util import log_event, redact

T = TypeVar("T", bound=BaseModel)


class Deadline:
    """Wall-clock budget shared by every model/search call in one research run."""

    def __init__(self, seconds: float, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._end = clock() + seconds

    def remaining(self) -> float:
        return self._end - self._clock()


@dataclass
class UsageCounters:
    """Cumulative consumption for one backend instance."""

    input_tokens: int = 0
    output_tokens: int = 0
    grounded_searches: int = 0
    finance_searches: int = 0      # SerpApi Google Finance searches

    def snapshot(self) -> "UsageCounters":
        return UsageCounters(self.input_tokens, self.output_tokens, self.grounded_searches, self.finance_searches)

    def since(self, earlier: "UsageCounters") -> "UsageCounters":
        return UsageCounters(
            self.input_tokens - earlier.input_tokens,
            self.output_tokens - earlier.output_tokens,
            self.grounded_searches - earlier.grounded_searches,
            self.finance_searches - earlier.finance_searches,
        )


@runtime_checkable
class LLMBackend(Protocol):
    """Anything that can turn a prompt into a Pydantic-validated object."""

    provider: str
    model: str
    usage: UsageCounters
    call_count: int

    def generate_structured(
        self,
        prompt_for: Callable[[Optional[str]], str],
        schema: Type[T],
        *,
        system: str,
        deadline: Deadline,
        stage: str,
        request_id: str = "",
    ) -> T: ...

    def list_models(self) -> list[str]: ...


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def strip_json_fences(text: str) -> str:
    text = text.strip()
    match = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, flags=re.DOTALL)
    return match.group(1) if match else text


def strip_think_blocks(text: str) -> str:
    """Reasoning models may prepend <think>...</think>; it is not part of the answer."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def classify_http_status(status: int, message: str) -> ResearchError:
    """Provider-neutral HTTP status -> typed research error (message redacted)."""
    detail = redact(f"HTTP {status}: {message}")[:300]
    lowered = message.lower()
    if status in (401, 403) or "api key" in lowered or "api_key" in lowered:
        return ResearchAuthError(detail)
    if status == 429:
        return ResearchQuotaError(detail)
    if status in (408, 504):
        return ResearchTimeoutError(detail)
    if status >= 500:
        return ResearchUnavailableError(detail)
    if status in (400, 404):
        return ResearchModelError(detail)
    return ResearchUnavailableError(detail)


def classify_transport_error(exc: Exception, where: str = "") -> ResearchError:
    detail = redact(f"{type(exc).__name__}: {exc}")[:300]
    if isinstance(exc, httpx.ConnectError):
        return LLMUnavailableError(f"cannot connect {where}".strip() + f" ({detail})")
    if isinstance(exc, (httpx.TimeoutException, TimeoutError)):
        return ResearchTimeoutError(detail)
    return ResearchUnavailableError(detail)


def call_with_retries(
    fn: Callable[[float], Any],
    *,
    classify: Callable[[Exception], ResearchError],
    max_retries: int,
    request_timeout: float,
    deadline: Deadline,
    sleep: Callable[[float], None],
    stage: str,
    request_id: str,
    on_attempt: Callable[[], None] = lambda: None,
) -> Any:
    """Run `fn(timeout_seconds)` with typed errors, exponential backoff for
    retryable failures, and a hard overall deadline."""
    last: Optional[ResearchError] = None
    for attempt in range(max_retries + 1):
        remaining = deadline.remaining()
        if remaining < 1:
            raise last or ResearchTimeoutError("overall research deadline exceeded")
        on_attempt()
        try:
            return fn(min(request_timeout, remaining))
        except ResearchError as exc:
            last = exc
        except Exception as exc:  # noqa: BLE001 - classified into a typed error
            last = classify(exc)
        if not last.retryable or last.fail_fast or attempt == max_retries:
            raise last
        delay = min(2.0 ** attempt, 8.0, max(deadline.remaining() - 1, 0))
        log_event("research.call.retry", level=30, request_id=request_id, stage=stage,
                  attempt=attempt + 1, error_type=last.code)
        sleep(delay)
    raise last or ResearchUnavailableError()  # pragma: no cover - loop always returns/raises


def structured_loop(
    get_text: Callable[[Optional[str]], Optional[str]],
    schema: Type[T],
    *,
    max_retries: int,
    stage: str,
    request_id: str,
) -> T:
    """Call the model, validate against `schema`, and on malformed/invalid
    output retry with a repair note up to `max_retries` times. Invalid data is
    never coerced; exhaustion raises SchemaValidationError."""
    repair: Optional[str] = None
    for attempt in range(max_retries + 1):
        raw = get_text(repair)
        try:
            if not raw:
                raise ValueError("empty response")
            return schema.model_validate_json(strip_json_fences(strip_think_blocks(raw)))
        except (ValidationError, ValueError) as exc:
            repair = redact(str(exc))[:400]
            log_event(
                "research.schema.validation_failed", level=30, request_id=request_id, stage=stage,
                attempt=attempt + 1, error_type=type(exc).__name__,
            )
    raise SchemaValidationError(f"{stage}: output failed validation after {max_retries + 1} attempt(s)")


class HttpLLMBackend:
    """Base for JSON-over-HTTP providers (Ollama, OpenAI-compatible, Anthropic).

    Subclasses implement `_build_request` and `_parse_response`; this class
    supplies retries, error mapping, usage counting and structured output.
    """

    provider = "http"

    def __init__(
        self,
        config: ResearchConfig,
        *,
        client: Optional[httpx.Client] = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.config = config
        self.model = config.model
        self._client = client or httpx.Client()
        self._sleep = sleep
        self.usage = UsageCounters()
        self.call_count = 0

    # -- subclass hooks -----------------------------------------------------

    def _build_request(self, system: str, prompt: str, schema: Type[BaseModel]) -> tuple[str, dict, dict]:
        """Return (url, headers, json_body)."""
        raise NotImplementedError

    def _parse_response(self, data: dict) -> tuple[str, int, int]:
        """Return (text, input_tokens, output_tokens)."""
        raise NotImplementedError

    def list_models(self) -> list[str]:
        raise NotImplementedError

    # -- shared -------------------------------------------------------------

    def _classify(self, exc: Exception) -> ResearchError:
        return classify_transport_error(exc, where=f"to {self.provider}")

    def _post(self, url: str, headers: dict, body: dict, timeout: float) -> dict:
        response = self._client.post(url, headers=headers, json=body, timeout=timeout)
        if response.status_code >= 400:
            try:
                message = str(response.json())
            except ValueError:
                message = response.text
            raise self._status_error(response.status_code, message)
        return response.json()

    def _status_error(self, status: int, message: str) -> ResearchError:
        return classify_http_status(status, message)

    def generate_structured(
        self,
        prompt_for: Callable[[Optional[str]], str],
        schema: Type[T],
        *,
        system: str,
        deadline: Deadline,
        stage: str,
        request_id: str = "",
    ) -> T:
        def get_text(repair: Optional[str]) -> Optional[str]:
            url, headers, body = self._build_request(system, prompt_for(repair), schema)

            def once(timeout: float) -> dict:
                return self._post(url, headers, body, timeout)

            data = call_with_retries(
                once, classify=self._classify, max_retries=self.config.max_retries,
                request_timeout=float(self.config.request_timeout_seconds), deadline=deadline,
                sleep=self._sleep, stage=stage, request_id=request_id,
                on_attempt=lambda: setattr(self, "call_count", self.call_count + 1),
            )
            text, tokens_in, tokens_out = self._parse_response(data)
            self.usage.input_tokens += tokens_in
            self.usage.output_tokens += tokens_out
            return text

        return structured_loop(
            get_text, schema, max_retries=self.config.max_retries, stage=stage, request_id=request_id
        )
