"""LLM provider registry.

Adding a provider = write one adapter module exposing a class that satisfies
`LLMBackend` (see base.py), then add one line to PROVIDERS below. Nothing
else in `research/` knows which provider is active.
"""

from __future__ import annotations

import importlib
from typing import Callable

from research.config import ResearchConfig, get_api_key
from research.exceptions import ResearchNotConfiguredError
from research.llm.base import Deadline, LLMBackend, UsageCounters  # noqa: F401

# provider -> (module, class). Imported lazily so unused providers cost nothing.
PROVIDERS: dict[str, tuple[str, str]] = {
    "gemini": ("research.agent", "GeminiAgent"),
    "ollama": ("research.llm.ollama", "OllamaBackend"),
    "openai": ("research.llm.openai_compat", "OpenAICompatBackend"),
    "anthropic": ("research.llm.anthropic", "AnthropicBackend"),
}


def build_llm(config: ResearchConfig) -> LLMBackend:
    """Construct the configured backend, or raise a typed, actionable error."""
    if config.provider != "ollama" and not config.model:
        raise ResearchNotConfiguredError(
            f"no model configured for provider '{config.provider}': set research.models.{config.provider} "
            "in config/settings.yaml or RESEARCH_MODEL"
        )
    # Hosted APIs need a key; an OpenAI-compatible server on a custom base_url (LM Studio, vLLM, ...) may not.
    needs_key = config.provider not in ("ollama",) and not (config.provider == "openai" and config.base_url)
    if needs_key and not get_api_key(config=config):
        raise ResearchNotConfiguredError(f"set {config.resolved_api_key_env} in .env (see .env.example)")
    module_name, class_name = PROVIDERS[config.provider]
    cls: Callable[..., LLMBackend] = getattr(importlib.import_module(module_name), class_name)
    return cls(config)
