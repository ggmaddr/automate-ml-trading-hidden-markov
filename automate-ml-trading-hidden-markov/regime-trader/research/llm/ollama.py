"""Ollama backend (local, free). Uses the native /api/chat so we can set
`num_ctx` (the default 4k context silently truncates long prompts) and pass a
JSON schema in `format` for schema-constrained decoding."""

from __future__ import annotations

from typing import Optional, Type

import httpx
from pydantic import BaseModel

from research.config import ResearchConfig
from research.exceptions import LLMUnavailableError, ResearchError
from research.llm.base import HttpLLMBackend, classify_http_status

DEFAULT_BASE_URL = "http://localhost:11434"


class OllamaBackend(HttpLLMBackend):
    provider = "ollama"

    def __init__(self, config: ResearchConfig, **kwargs) -> None:
        super().__init__(config, **kwargs)
        self.base_url = (config.base_url or DEFAULT_BASE_URL).rstrip("/")

    def _build_request(self, system: str, prompt: str, schema: Type[BaseModel]) -> tuple[str, dict, dict]:
        body: dict = {
            "model": self.model,
            "stream": False,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            "format": schema.model_json_schema(),
            "options": {"num_ctx": self.config.num_ctx, "temperature": 0.2},
        }
        if self.config.think is not None:
            body["think"] = self.config.think
        return f"{self.base_url}/api/chat", {}, body

    def _parse_response(self, data: dict) -> tuple[str, int, int]:
        return (
            (data.get("message") or {}).get("content", ""),
            int(data.get("prompt_eval_count") or 0),
            int(data.get("eval_count") or 0),
        )

    def _status_error(self, status: int, message: str) -> ResearchError:
        if status == 404 and "not found" in message.lower():
            return LLMUnavailableError(f"model '{self.model}' is not pulled: run `ollama pull {self.model}`")
        return classify_http_status(status, message)

    def list_models(self) -> list[str]:
        try:
            response = self._client.get(f"{self.base_url}/api/tags", timeout=10)
            response.raise_for_status()
        except httpx.ConnectError as exc:
            raise LLMUnavailableError(f"cannot connect to Ollama at {self.base_url}") from exc
        except httpx.HTTPError as exc:
            raise self._classify(exc) from exc
        return sorted(m["name"] for m in response.json().get("models", []))
