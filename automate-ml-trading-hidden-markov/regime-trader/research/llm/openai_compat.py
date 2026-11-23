"""OpenAI and OpenAI-compatible backend (/v1/chat/completions).

Works with OpenAI itself and with any server speaking the same protocol
(LM Studio, vLLM, OpenRouter, ...) by setting `research.base_url`. Uses JSON
mode plus the schema in the prompt; the Pydantic validate-or-repair loop is
what guarantees the shape, so no provider-specific strict-schema features
are required.
"""

from __future__ import annotations

import json
from typing import Type

from pydantic import BaseModel

from research.config import ResearchConfig, get_api_key
from research.exceptions import LLMUnavailableError
from research.llm.base import HttpLLMBackend

DEFAULT_BASE_URL = "https://api.openai.com"


class OpenAICompatBackend(HttpLLMBackend):
    provider = "openai"

    def __init__(self, config: ResearchConfig, **kwargs) -> None:
        super().__init__(config, **kwargs)
        self.base_url = (config.base_url or DEFAULT_BASE_URL).rstrip("/")
        self._api_key = get_api_key(config=config)

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}

    def _build_request(self, system: str, prompt: str, schema: Type[BaseModel]) -> tuple[str, dict, dict]:
        instructions = (
            f"{prompt}\n\nReturn ONLY a JSON object that validates against this JSON Schema:\n"
            f"{json.dumps(schema.model_json_schema())}"
        )
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": instructions}],
            "response_format": {"type": "json_object"},
        }
        return f"{self.base_url}/v1/chat/completions", self._headers(), body

    def _parse_response(self, data: dict) -> tuple[str, int, int]:
        choices = data.get("choices") or []
        text = ((choices[0].get("message") or {}).get("content") or "") if choices else ""
        usage = data.get("usage") or {}
        return text, int(usage.get("prompt_tokens") or 0), int(usage.get("completion_tokens") or 0)

    def list_models(self) -> list[str]:
        import httpx

        try:
            response = self._client.get(f"{self.base_url}/v1/models", headers=self._headers(), timeout=15)
            response.raise_for_status()
        except httpx.ConnectError as exc:
            raise LLMUnavailableError(f"cannot connect to {self.base_url}") from exc
        except httpx.HTTPError as exc:
            raise self._classify(exc) from exc
        return sorted(m["id"] for m in response.json().get("data", []))
