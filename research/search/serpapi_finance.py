"""Google Finance news via SerpApi (https://serpapi.com/google-finance-api).

SerpApi is a paid third-party service (free plan: 250 searches/month) that
returns Google Finance's own page data as JSON — quote, key stats, financials
and the news list for a ticker. This project uses ONLY its `news_results` as an
extra, real-link evidence source: the quote/stat fields are market data, not
research evidence. It is NOT Google Finance "Deep Search", which has no API.

One search per research run (cached SerpApi results are free on their side).
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import Optional

import httpx

from research.config import ResearchConfig, get_api_key
from research.exceptions import (
    ResearchAuthError,
    ResearchNotConfiguredError,
    ResearchQuotaError,
    ResearchTimeoutError,
    ResearchUnavailableError,
)
from research.llm.base import Deadline
from research.logging_util import redact
from research.search import SearchHit

ENDPOINT = "https://serpapi.com/search.json"
_RELATIVE = re.compile(r"(\d+)\s+(minute|hour|day|week|month)s?\s+ago", re.IGNORECASE)
_UNIT_DAYS = {"minute": 0, "hour": 0, "day": 1, "week": 7, "month": 30}


def parse_news_date(text: Optional[str], now: Optional[datetime] = None) -> Optional[str]:
    """Best-effort ISO date (YYYY-MM-DD) from SerpApi's date strings ("3 days ago",
    "Oct 5, 2026"). Returns None when it can't be read — never a guessed date."""
    if not text:
        return None
    now = now or datetime.now(timezone.utc)
    match = _RELATIVE.search(text)
    if match:
        count, unit = int(match.group(1)), match.group(2).lower()
        delta = {"minute": timedelta(minutes=count), "hour": timedelta(hours=count)}.get(unit, timedelta(days=_UNIT_DAYS[unit] * count))
        return (now - delta).date().isoformat()
    for fmt in ("%b %d, %Y", "%B %d, %Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(text.strip(), fmt).date().isoformat()
        except ValueError:
            continue
    return None


class SerpApiFinanceNews:
    name = "serpapi_finance"

    def __init__(self, config: ResearchConfig, client: Optional[httpx.Client] = None, api_key: Optional[str] = None) -> None:
        self.exchanges = list(config.finance_exchanges) or ["NASDAQ"]
        self.timeout = min(float(config.request_timeout_seconds), 30.0)
        self._client = client or httpx.Client()
        self._api_key = api_key or get_api_key(config=_serp_key_config(config))
        if not self._api_key:
            raise ResearchNotConfiguredError("set SERPAPI_API_KEY in .env (see .env.example)")
        self.searches = 0     # successful (billable) searches made by this instance

    def _get(self, symbol: str, exchange: str, timeout: float) -> dict:
        try:
            response = self._client.get(
                ENDPOINT, timeout=timeout,
                params={"engine": "google_finance", "q": f"{symbol}:{exchange}", "hl": "en", "window": "5D",
                        "api_key": self._api_key},
            )
        except httpx.TimeoutException as exc:
            raise ResearchTimeoutError(redact(f"SerpApi timed out: {exc}")) from exc
        except httpx.HTTPError as exc:
            raise ResearchUnavailableError(redact(f"SerpApi unreachable: {type(exc).__name__}")) from exc
        try:
            data = response.json()
        except ValueError:
            data = {}
        error = str(data.get("error", "")) if isinstance(data, dict) else ""
        if response.status_code in (401, 403) or "invalid api key" in error.lower():
            raise ResearchAuthError(redact("SerpApi rejected the API key"))
        if response.status_code == 429 or "run out of searches" in error.lower():
            raise ResearchQuotaError(redact("SerpApi quota or rate limit reached"))
        if response.status_code >= 500:
            raise ResearchUnavailableError(f"SerpApi HTTP {response.status_code}")
        if not error:
            self.searches += 1       # errored searches are not billed by SerpApi
        return data if isinstance(data, dict) else {}

    @staticmethod
    def _flatten(news_results: list) -> list[dict]:
        out: list[dict] = []
        for item in news_results or []:
            if isinstance(item, dict) and isinstance(item.get("items"), list):   # grouped format
                out.extend(i for i in item["items"] if isinstance(i, dict))
            elif isinstance(item, dict):
                out.append(item)
        return out

    def news(self, symbol: str, *, max_results: int, deadline: Deadline, now: Optional[datetime] = None) -> list[SearchHit]:
        """News links Google Finance lists for `symbol`, trying each configured exchange until one answers."""
        for exchange in self.exchanges:
            data = self._get(symbol, exchange, max(1.0, min(self.timeout, deadline.remaining())))
            items = self._flatten(data.get("news_results"))
            hits: list[SearchHit] = []
            for item in items:
                url = item.get("link") or ""
                if not url.startswith(("http://", "https://")):
                    continue
                snippet = (item.get("snippet") or "").strip()
                source = (item.get("source") or "").strip()
                hits.append(SearchHit(
                    title=(item.get("title") or "").strip() or (f"{source}: " if source else "") + snippet[:90] or url,
                    url=url, snippet=snippet, published=parse_news_date(item.get("date"), now),
                ))
                if len(hits) >= max_results:
                    break
            if hits:
                return hits
        return []

    def check(self) -> str:
        data = self._get("GOOGL", self.exchanges[0], 15.0)
        return f"reachable, {len(self._flatten(data.get('news_results')))} news items for a test ticker"


def _serp_key_config(config: ResearchConfig) -> ResearchConfig:
    """get_api_key reads config.resolved_api_key_env; point it at SERPAPI_API_KEY."""
    return config.model_copy(update={"api_key_env": "SERPAPI_API_KEY"})
