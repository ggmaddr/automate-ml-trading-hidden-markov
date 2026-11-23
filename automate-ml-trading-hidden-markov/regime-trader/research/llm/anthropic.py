"""Anthropic backend (/v1/messages). Structured output via a single forced
tool whose input schema is the Pydantic schema; the tool is never executed —
its arguments are just the JSON we validate."""

from __future__ import annotations

import json
from typing import Type

import httpx
from pydantic import BaseModel

from research.config import ResearchConfig, get_api_key
from research.exceptions import LLMUnavailableError
from research.llm.base import HttpLLMBackend

DEFAULT_BASE_URL = "https://api.anthropic.com"
API_VERSION = "2023-06-01"
MAX_OUTPUT_TOKENS = 4096
TOOL_NAME = "submit_result"


class AnthropicBackend(HttpLLMBackend):
    provider = "anthropic"

    def __init__(self, config: ResearchConfig, **kwargs) -> None:
        super().__init__(config, **kwargs)
        self.base_url = (config.base_url or DEFAULT_BASE_URL).rstrip("/")
        self._api_key = get_api_key(config=config) or ""

    def _headers(self) -> dict:
        return {"x-api-key": self._api_key, "anthropic-version": API_VERSION}

    def _build_request(self, system: str, prompt: str, schema: Type[BaseModel]) -> tuple[str, dict, dict]:
        body = {
            "model": self.model,
            "max_tokens": MAX_OUTPUT_TOKENS,
            "system": system,
            "messages": [{"role": "user", "content": prompt}],
            "tools": [{
                "name": TOOL_NAME,
                "description": "Return the final structured result.",
                "input_schema": schema.model_json_schema(),
            }],
            "tool_choice": {"type": "tool", "name": TOOL_NAME},
        }
        return f"{self.base_url}/v1/messages", self._headers(), body

    def _parse_response(self, data: dict) -> tuple[str, int, int]:
        text = ""
        for block in data.get("content") or []:
            if block.get("type") == "tool_use" and block.get("name") == TOOL_NAME:
                text = json.dumps(block.get("input") or {})
                break
            if block.get("type") == "text" and not text:
                text = block.get("text", "")
        usage = data.get("usage") or {}
        return text, int(usage.get("input_tokens") or 0), int(usage.get("output_tokens") or 0)

    def list_models(self) -> list[str]:
        try:
            response = self._client.get(f"{self.base_url}/v1/models", headers=self._headers(), timeout=15)
            response.raise_for_status()
        except httpx.ConnectError as exc:
            raise LLMUnavailableError(f"cannot connect to {self.base_url}") from exc
        except httpx.HTTPError as exc:
            raise self._classify(exc) from exc
        return sorted(m["id"] for m in response.json().get("data", []))
