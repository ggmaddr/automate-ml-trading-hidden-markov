"""Gemini backend: `LLMBackend` (structured output) + native Google Search grounding.

All Gemini traffic goes through `GeminiAgent`. It exposes no function-calling
and no tool other than Google Search (+ URL context), so nothing the model
emits can trigger application behaviour. Shared retry / deadline / repair
logic lives in research/llm/base.py and is reused by every provider.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Optional, Type

from research.config import ResearchConfig, get_api_key
from research.exceptions import (
    ResearchAuthError,
    ResearchError,
    ResearchModelError,
    ResearchNotConfiguredError,
    ResearchQuotaError,
    ResearchTimeoutError,
    ResearchUnavailableError,
)
from research.grounding import GroundingData, extract_grounding
from research.llm.base import (  # noqa: F401 - Deadline/UsageCounters re-exported for existing importers
    Deadline,
    T,
    UsageCounters,
    call_with_retries,
    structured_loop,
)
from research.logging_util import redact


@dataclass
class GroundedResponse:
    text: str
    grounding: GroundingData


class GeminiAgent:
    provider = "gemini"

    def __init__(
        self,
        config: ResearchConfig,
        api_key: Optional[str] = None,
        client: Any = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.config = config
        self.model = config.model
        self._sleep = sleep
        self.call_count = 0
        self.usage = UsageCounters()
        if client is None:
            key = api_key or get_api_key()
            if not key:
                raise ResearchNotConfiguredError("set GEMINI_API_KEY in .env (see .env.example)")
            from google import genai

            client = genai.Client(api_key=key)
        self._client = client

    # ------------------------------------------------------------------
    # Error handling
    # ------------------------------------------------------------------

    @staticmethod
    def _classify(exc: Exception) -> ResearchError:
        """Map SDK/network exceptions to typed research errors (messages redacted)."""
        detail = redact(f"{type(exc).__name__}: {exc}")[:300]
        code = getattr(exc, "code", None)
        if isinstance(code, int):
            lowered = str(exc).lower()
            if code in (401, 403) or "api key" in lowered or "api_key" in lowered:
                return ResearchAuthError(detail)
            if code == 429:
                return ResearchQuotaError(detail)
            if code == 504:
                return ResearchTimeoutError(detail)
            if code >= 500:
                return ResearchUnavailableError(detail)
            if code in (400, 404):
                return ResearchModelError(detail)
            return ResearchUnavailableError(detail)
        name = type(exc).__name__
        if isinstance(exc, TimeoutError) or "Timeout" in name:
            return ResearchTimeoutError(detail)
        return ResearchUnavailableError(detail)

    def _track_tokens(self, response: Any) -> None:
        meta = getattr(response, "usage_metadata", None)
        if meta is None:
            return
        self.usage.input_tokens += getattr(meta, "prompt_token_count", None) or 0
        # thinking tokens are billed as output
        self.usage.output_tokens += (getattr(meta, "candidates_token_count", None) or 0) + (
            getattr(meta, "thoughts_token_count", None) or 0
        )

    def _generate(self, *, contents: str, config_kwargs: dict, deadline: Deadline, stage: str, request_id: str) -> Any:
        from google.genai import types

        def once(timeout: float) -> Any:
            response = self._client.models.generate_content(
                model=self.config.model,
                contents=contents,
                config=types.GenerateContentConfig(
                    http_options=types.HttpOptions(timeout=int(timeout * 1000)), **config_kwargs
                ),
            )
            self._track_tokens(response)
            return response

        return call_with_retries(
            once, classify=self._classify, max_retries=self.config.max_retries,
            request_timeout=float(self.config.request_timeout_seconds), deadline=deadline,
            sleep=self._sleep, stage=stage, request_id=request_id,
            on_attempt=lambda: setattr(self, "call_count", self.call_count + 1),
        )

    # ------------------------------------------------------------------
    # Native grounded search
    # ------------------------------------------------------------------

    def grounded_search(self, prompt: str, *, system: str, deadline: Deadline, request_id: str = "") -> GroundedResponse:
        """One Google-Search-grounded generation. Returns the prose plus the
        REAL grounding metadata; callers must not trust the prose beyond it."""
        from google.genai import types

        tools = [types.Tool(google_search=types.GoogleSearch())]
        if self.config.enable_url_context:
            tools.append(types.Tool(url_context=types.UrlContext()))
        response = self._generate(
            contents=prompt, config_kwargs={"system_instruction": system, "tools": tools},
            deadline=deadline, stage="grounded_search", request_id=request_id,
        )
        grounding = extract_grounding(response)
        self.usage.grounded_searches += len(grounding.queries)
        return GroundedResponse(text=response.text or "", grounding=grounding)

    # ------------------------------------------------------------------
    # Structured output
    # ------------------------------------------------------------------

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
        """JSON-mode generation validated by Pydantic; see `structured_loop`."""

        def get_text(repair: Optional[str]) -> Optional[str]:
            return self._generate(
                contents=prompt_for(repair),
                config_kwargs={
                    "system_instruction": system,
                    "response_mime_type": "application/json",
                    "response_schema": schema,
                },
                deadline=deadline, stage=stage, request_id=request_id,
            ).text

        return structured_loop(
            get_text, schema, max_retries=self.config.max_retries, stage=stage, request_id=request_id
        )

    # ------------------------------------------------------------------

    def list_models(self) -> list[str]:
        """Model ids the current key can use for generateContent (for `--list-models`)."""
        try:
            return sorted(
                m.name.removeprefix("models/")
                for m in self._client.models.list()
                if "generateContent" in (getattr(m, "supported_actions", None) or [])
            )
        except Exception as exc:  # noqa: BLE001
            raise self._classify(exc) from exc
