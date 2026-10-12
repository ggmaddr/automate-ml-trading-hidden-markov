"""SearXNG search provider (self-hosted; see ops/docker-compose.searxng.yml).

SearXNG must have the `json` format enabled in its settings.yml, which the
bundled ops config does.
"""

from __future__ import annotations

from typing import Optional

import httpx

from research.config import ResearchConfig
from research.exceptions import ResearchTimeoutError, ResearchUnavailableError, SearchUnavailableError
from research.llm.base import Deadline
from research.logging_util import redact
from research.search import SearchHit


def _time_range(lookback_days: int) -> str:
    if lookback_days <= 1:
        return "day"
    if lookback_days <= 7:
        return "week"
    if lookback_days <= 31:
        return "month"
    return "year"


class SearxngSearch:
    name = "searxng"

    def __init__(self, config: ResearchConfig, client: Optional[httpx.Client] = None) -> None:
        self.base_url = config.search_base_url.rstrip("/")
        self.timeout = min(float(config.request_timeout_seconds), 30.0)
        self.categories = list(config.search_categories) or ["general"]
        self._client = client or httpx.Client()

    def _get(self, params: dict, timeout: float) -> dict:
        try:
            response = self._client.get(f"{self.base_url}/search", params=params, timeout=timeout)
        except httpx.ConnectError as exc:
            raise SearchUnavailableError(f"cannot connect to SearXNG at {self.base_url}") from exc
        except httpx.TimeoutException as exc:
            raise ResearchTimeoutError(redact(f"SearXNG timed out: {exc}")) from exc
        except httpx.HTTPError as exc:
            raise ResearchUnavailableError(redact(f"SearXNG error: {type(exc).__name__}")) from exc
        if response.status_code == 403:
            raise SearchUnavailableError("SearXNG refused the request (is `json` enabled under search.formats?)")
        if response.status_code >= 400:
            raise ResearchUnavailableError(f"SearXNG HTTP {response.status_code}")
        try:
            return response.json()
        except ValueError as exc:
            raise SearchUnavailableError("SearXNG did not return JSON (enable the json format)") from exc

    def search(self, query: str, *, max_results: int, lookback_days: int, deadline: Deadline) -> list[SearchHit]:
        """Results from each configured category in priority order (news first: it returns
        articles rather than quote pages), de-duplicated by URL, capped at `max_results`."""
        hits: list[SearchHit] = []
        seen: set[str] = set()
        for category in self.categories:
            timeout = max(1.0, min(self.timeout, deadline.remaining()))
            data = self._get(
                {"q": query, "format": "json", "language": "en", "categories": category,
                 "time_range": _time_range(lookback_days)}, timeout,
            )
            for item in data.get("results") or []:
                url = item.get("url") or ""
                if not url.startswith(("http://", "https://")) or url in seen:
                    continue
                seen.add(url)
                hits.append(SearchHit(
                    title=(item.get("title") or "").strip() or url,
                    url=url,
                    snippet=(item.get("content") or "").strip(),
                    published=item.get("publishedDate") or None,
                ))
                if len(hits) >= max_results:
                    return hits
        return hits

    def check(self) -> str:
        data = self._get({"q": "test", "format": "json"}, 10.0)
        return f"reachable, {len(data.get('results') or [])} results for a test query"
