"""Research health for the monitoring UI: healthy / unavailable / disabled."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Mapping, Optional

from research.cache import ResearchCache
from research.config import ResearchConfig, get_api_key


@dataclass
class ResearchHealth:
    status: str                      # "healthy" | "unavailable" | "disabled"
    detail: str
    last_success_at: Optional[datetime] = None
    cache_hit_rate: Optional[float] = None


def get_research_health(
    config: ResearchConfig, cache: Optional[ResearchCache] = None, env: Optional[Mapping[str, str]] = None
) -> ResearchHealth:
    if not config.enabled:
        return ResearchHealth("disabled", "research.enabled is false")
    cache = cache or ResearchCache(config.resolved_cache_path, config.cache_ttl_minutes)
    try:
        stats = cache.stats()
    except Exception as exc:  # noqa: BLE001 - health display must never raise
        return ResearchHealth("unavailable", f"metrics unreadable: {type(exc).__name__}")
    needs_key = config.provider != "ollama" and not (config.provider == "openai" and config.base_url)
    if needs_key and not get_api_key(env, config):
        return ResearchHealth("unavailable", f"no {config.resolved_api_key_env} configured", stats.last_success_at, stats.hit_rate)
    if not config.model:
        return ResearchHealth("unavailable", f"no model set for provider {config.provider}", stats.last_success_at, stats.hit_rate)
    if stats.last_outcome == "error":
        return ResearchHealth("unavailable", f"last request failed ({stats.last_error_code})", stats.last_success_at, stats.hit_rate)
    return ResearchHealth("healthy", "ok" if stats.requests else "ok (no requests yet)", stats.last_success_at, stats.hit_rate)
