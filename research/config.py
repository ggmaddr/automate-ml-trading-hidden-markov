"""Research configuration, loaded from the existing `config/settings.yaml`.

Model names live in ONE place: `research.models` (one entry per provider) in
config/settings.yaml. Switch provider with `research.provider` or the
RESEARCH_PROVIDER env var; override the model with RESEARCH_MODEL (or
GEMINI_MODEL when the provider is gemini). Nothing else in the codebase
names a model.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal, Mapping, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SETTINGS_PATH = PROJECT_ROOT / "config" / "settings.yaml"

# Fallback only when settings.yaml has no `research:` section at all.
_FALLBACK_MODEL = "gemini-3.8-flash"


class ResearchConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    provider: Literal["ollama", "openai", "anthropic", "gemini"] = "gemini"
    model: str = _FALLBACK_MODEL                       # resolved from `models[provider]` by load_research_config
    models: dict[str, str] = Field(default_factory=dict)   # provider -> model id
    base_url: Optional[str] = None                     # provider endpoint override (Ollama / OpenAI-compatible servers)
    num_ctx: int = Field(16384, ge=2048)               # Ollama context window (its 4k default silently truncates prompts)
    think: Optional[bool] = None                       # Ollama: enable/disable model "thinking" (None = model default)
    api_key_env: Optional[str] = None                  # env var holding the key; default per provider
    search_provider: Literal["auto", "searxng", "gemini_grounding"] = "auto"
    search_base_url: str = "http://localhost:8888"
    search_categories: list[str] = Field(default_factory=lambda: ["news", "general"])   # SearXNG categories, in priority order
    # Optional extra evidence source: Google Finance's news list for the ticker, via SerpApi (needs SERPAPI_API_KEY).
    finance_news_provider: Literal["none", "serpapi"] = "none"
    finance_exchanges: list[str] = Field(default_factory=lambda: ["NASDAQ", "NYSE"])   # tried in order for TICKER:EXCHANGE
    serpapi_free_searches_per_month: int = Field(250, ge=0)
    search_max_results: int = Field(8, ge=1, le=30)
    fetch_top_n: int = Field(4, ge=0, le=10)           # pages fetched per search round (0 = snippets only)
    # Optional per-provider token prices for the usage estimate: {provider: {input_per_mtok, output_per_mtok}}
    pricing: dict[str, dict[str, float]] = Field(default_factory=dict)
    max_internal_tool_calls: int = Field(5, ge=1, le=20)
    max_search_rounds: int = Field(2, ge=1, le=5)
    max_sources: int = Field(12, ge=1, le=50)
    min_sources: int = Field(3, ge=1, le=50)
    cache_ttl_minutes: int = Field(360, ge=0)
    max_retries: int = Field(2, ge=0, le=5)
    request_timeout_seconds: int = Field(90, ge=5)
    overall_timeout_seconds: int = Field(240, ge=10)   # hard cap across every call in one research run
    default_lookback_days: int = Field(14, ge=1)
    max_lookback_days: int = Field(90, ge=1)
    enable_url_context: bool = True
    enable_google_search: bool = True
    resolve_source_redirects: bool = True              # turn Gemini's grounding redirect links into publisher URLs
    # Usage / cost guard. Defaults mirror Google's published Gemini 3.x pricing as of Oct 2026
    # (https://ai.google.dev/gemini-api/docs/pricing) — they change, so they live in config.
    free_grounded_searches_per_month: int = Field(5000, ge=0)   # shared across Gemini 3+ models; needs a billing-enabled key
    grounded_search_price_per_1000_usd: float = Field(14.0, ge=0)
    input_price_per_million_tokens_usd: float = Field(0.75, ge=0)
    output_price_per_million_tokens_usd: float = Field(3.75, ge=0)
    monthly_budget_usd: float = Field(5.0, ge=0)                # local soft cap on ESTIMATED spend; 0 disables
    cache_path: str = "research_cache.db"              # relative paths resolve against the project root

    # Hard safety rails: Literal[False] makes a `true` here a config error,
    # not a switch. Phase 10 has no code path that could honour them.
    allow_broker_tools: Literal[False] = False
    allow_strategy_mutation: Literal[False] = False
    allow_risk_mutation: Literal[False] = False

    @property
    def effective_search_provider(self) -> str:
        if self.search_provider != "auto":
            return self.search_provider
        return "gemini_grounding" if self.provider == "gemini" else "searxng"

    @property
    def resolved_api_key_env(self) -> str:
        return self.api_key_env or {
            "gemini": "GEMINI_API_KEY", "openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY", "ollama": "",
        }[self.provider]

    @property
    def resolved_cache_path(self) -> Path:
        path = Path(self.cache_path)
        return path if path.is_absolute() else PROJECT_ROOT / path


def load_settings(path: Optional[Path] = None) -> dict:
    with open(path or SETTINGS_PATH) as f:
        return yaml.safe_load(f) or {}


def load_research_config(
    settings: Optional[dict] = None, env: Optional[Mapping[str, str]] = None
) -> ResearchConfig:
    """Build ResearchConfig from the `research:` section of settings (loaded
    from config/settings.yaml when not given), applying GEMINI_MODEL."""
    env = os.environ if env is None else env
    if settings is None:
        settings = load_settings()
    section = dict(settings.get("research") or {})
    if env.get("RESEARCH_PROVIDER"):
        section["provider"] = env["RESEARCH_PROVIDER"].strip().lower()
    if env.get("RESEARCH_SEARCH"):
        section["search_provider"] = env["RESEARCH_SEARCH"].strip().lower()
    provider = section.get("provider", "gemini")
    # model precedence: env override > models[provider] > legacy `model`
    models = section.get("models") or {}
    if env.get("RESEARCH_MODEL"):
        section["model"] = env["RESEARCH_MODEL"].strip()
    elif provider == "gemini" and env.get("GEMINI_MODEL"):
        section["model"] = env["GEMINI_MODEL"].strip()
    elif models.get(provider):
        section["model"] = models[provider]
    elif provider != "gemini" and "model" not in section:
        section["model"] = ""   # no model configured for this provider -> caught by backend checks
    return ResearchConfig(**section)


def get_api_key(env: Optional[Mapping[str, str]] = None, config: Optional[ResearchConfig] = None) -> Optional[str]:
    """API key for the configured provider (GEMINI_API_KEY when no config is given)."""
    env = os.environ if env is None else env
    var = config.resolved_api_key_env if config is not None else "GEMINI_API_KEY"
    key = (env.get(var) or "").strip() if var else ""
    return key or None
