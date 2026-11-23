"""Web-search providers for models without built-in search.

Adding a provider: implement `SearchProvider` (below) in its own module and
add it to `SEARCH_PROVIDERS`.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Optional, Protocol

from research.config import ResearchConfig
from research.llm.base import Deadline


@dataclass
class SearchHit:
    title: str
    url: str
    snippet: str = ""
    published: Optional[str] = None


class SearchProvider(Protocol):
    name: str

    def search(self, query: str, *, max_results: int, lookback_days: int, deadline: Deadline) -> list[SearchHit]: ...

    def check(self) -> str:
        """Probe the service; return a short status string or raise a typed error."""
        ...


SEARCH_PROVIDERS: dict[str, tuple[str, str]] = {
    "searxng": ("research.search.searxng", "SearxngSearch"),
}


def build_search_provider(config: ResearchConfig) -> SearchProvider:
    module_name, class_name = SEARCH_PROVIDERS[config.effective_search_provider]
    return getattr(importlib.import_module(module_name), class_name)(config)
