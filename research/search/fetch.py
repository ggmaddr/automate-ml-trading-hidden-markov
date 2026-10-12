"""Safe page fetcher for the search researcher.

Page text is UNTRUSTED input. This module's job is limiting what a hostile or
careless URL can do to the machine: http(s) only, public IPs only (checked on
every redirect hop), size/time/content-type caps. Fetched text is only ever
shown to the model as quoted data; the model has no tools to act on it.

Known residual risk: the address is validated, then httpx resolves the name
again to connect, so a DNS-rebinding attacker could race the check. Tool
results never leave the process and there is no function calling, which bounds
the impact; do not point this at an environment with sensitive internal HTTP
services reachable by hostname without adding connect-time IP pinning.
"""

from __future__ import annotations

import ipaddress
import socket
from html.parser import HTMLParser
from typing import Callable, Optional
from urllib.parse import urljoin, urlparse

import httpx

MAX_BYTES = 1_500_000
MAX_REDIRECTS = 3
MAX_TEXT_CHARS = 6000
ALLOWED_CONTENT_TYPES = ("text/html", "application/xhtml+xml", "text/plain")
USER_AGENT = "regime-trader-research/1.0 (+read-only research fetcher)"
_SKIP_TAGS = {"script", "style", "noscript", "nav", "header", "footer", "aside", "form", "svg", "iframe"}


def _default_resolver(host: str) -> list[str]:
    return [info[4][0] for info in socket.getaddrinfo(host, None)]


def is_public_url(url: str, resolver: Callable[[str], list[str]] = _default_resolver) -> bool:
    """True only for http(s) URLs whose host resolves exclusively to globally routable addresses."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        return False
    try:
        addresses = resolver(parsed.hostname)
    except (OSError, UnicodeError):
        return False
    if not addresses:
        return False
    for raw in addresses:
        try:
            if not ipaddress.ip_address(raw.split("%")[0]).is_global:
                return False
        except ValueError:
            return False
    return True


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in _SKIP_TAGS:
            self._skip_depth += 1

    def handle_endtag(self, tag):
        if tag in _SKIP_TAGS and self._skip_depth:
            self._skip_depth -= 1

    def handle_data(self, data):
        if not self._skip_depth and data.strip():
            self.parts.append(data.strip())


def extract_text(html: str, limit: int = MAX_TEXT_CHARS) -> str:
    parser = _TextExtractor()
    try:
        parser.feed(html)
    except Exception:  # noqa: BLE001 - malformed HTML: use what was parsed
        pass
    text = " ".join(parser.parts)
    return " ".join(text.split())[:limit]


class PageFetcher:
    def __init__(
        self,
        client: Optional[httpx.Client] = None,
        resolver: Callable[[str], list[str]] = _default_resolver,
        timeout: float = 8.0,
    ) -> None:
        self._client = client or httpx.Client(headers={"User-Agent": USER_AGENT})
        self._resolver = resolver
        self._timeout = timeout

    def fetch(self, url: str) -> Optional[str]:
        """Readable text of the page, or None if blocked/unfetchable. Never raises."""
        try:
            return self._fetch(url)
        except Exception:  # noqa: BLE001 - a bad page must never fail the research run
            return None

    def _fetch(self, url: str) -> Optional[str]:
        current = url
        for _ in range(MAX_REDIRECTS + 1):
            if not is_public_url(current, self._resolver):
                return None
            with self._client.stream(
                "GET", current, follow_redirects=False, timeout=self._timeout, headers={"User-Agent": USER_AGENT}
            ) as response:
                if response.status_code in (301, 302, 303, 307, 308):
                    location = response.headers.get("location")
                    if not location:
                        return None
                    current = urljoin(current, location)
                    continue
                if response.status_code != 200:
                    return None
                ctype = response.headers.get("content-type", "").split(";")[0].strip().lower()
                if ctype not in ALLOWED_CONTENT_TYPES:
                    return None
                body = bytearray()
                for chunk in response.iter_bytes():
                    body.extend(chunk)
                    if len(body) > MAX_BYTES:
                        break
                html = bytes(body[:MAX_BYTES]).decode(response.encoding or "utf-8", errors="replace")
                return extract_text(html) if ctype != "text/plain" else " ".join(html.split())[:MAX_TEXT_CHARS]
        return None
