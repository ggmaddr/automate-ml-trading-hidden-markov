"""Google Search grounding: extract real source provenance from Gemini responses.

Everything here reads the structured `grounding_metadata` the API returns
(search queries, grounding chunks, grounding supports). It never parses
citation-looking text out of the model's prose.
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Optional
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

_REDIRECT_HOSTS = {"vertexaisearch.cloud.google.com"}
_DOMAIN_LIKE = re.compile(r"^(?:[a-z0-9-]+\.)+[a-z]{2,}$", re.IGNORECASE)
_TRACKING_PARAMS = {"gclid", "fbclid", "ref", "ref_src", "mc_cid", "mc_eid"}


@dataclass
class GroundedSource:
    key: str                  # normalized URL, the dedupe identity
    url: str
    title: str
    domain: Optional[str]
    published: Optional[str] = None   # publication date from search metadata (never model-guessed)
    source_id: str = ""       # S1..Sn, assigned deterministically by GroundingData.finalize()


@dataclass
class GroundedPassage:
    """A span of generated text the search tool itself attributed to sources."""
    text: str
    source_keys: list[str]


@dataclass
class GroundingData:
    queries: list[str] = field(default_factory=list)
    sources: dict[str, GroundedSource] = field(default_factory=dict)   # insertion-ordered
    passages: list[GroundedPassage] = field(default_factory=list)

    @property
    def has_grounding(self) -> bool:
        return bool(self.sources)

    def merge(self, other: "GroundingData") -> None:
        for q in other.queries:
            if q not in self.queries:
                self.queries.append(q)
        for key, src in other.sources.items():
            self.sources.setdefault(key, src)
        self.passages.extend(other.passages)

    def finalize(self, max_sources: int) -> "GroundingData":
        """Cap to `max_sources`, assign S-ids in first-seen order, and drop
        passage references to dropped sources. Deterministic."""
        kept = dict(list(self.sources.items())[:max_sources])
        for i, src in enumerate(kept.values(), start=1):
            src.source_id = f"S{i}"
        passages = []
        for p in self.passages:
            keys = [k for k in p.source_keys if k in kept]
            if keys:
                passages.append(GroundedPassage(p.text, keys))
        return GroundingData(queries=list(self.queries), sources=kept, passages=passages)

    def source_by_id(self) -> dict[str, GroundedSource]:
        return {s.source_id: s for s in self.sources.values() if s.source_id}


# ---------------------------------------------------------------------------
# URL helpers
# ---------------------------------------------------------------------------

def host_of(url: str) -> Optional[str]:
    host = urlparse(url).hostname
    if not host:
        return None
    return host[4:] if host.startswith("www.") else host


def normalize_url(url: str) -> str:
    """Canonical form for dedupe: lowercase scheme/host, no fragment, no
    tracking params, no trailing slash."""
    parsed = urlparse(url.strip())
    host = (parsed.hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    query = urlencode(
        [(k, v) for k, v in parse_qsl(parsed.query, keep_blank_values=True)
         if not k.lower().startswith("utm_") and k.lower() not in _TRACKING_PARAMS]
    )
    path = parsed.path.rstrip("/")
    port = f":{parsed.port}" if parsed.port else ""
    return urlunparse(("https" if parsed.scheme in ("http", "https") else parsed.scheme, host + port, path, "", query, ""))


def is_redirect_url(url: str) -> bool:
    return (urlparse(url).hostname or "") in _REDIRECT_HOSTS


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def extract_grounding(response: Any) -> GroundingData:
    """Pull queries, deduplicated sources, and attributed passages from one
    `generate_content` response. Tolerates any missing piece of metadata."""
    data = GroundingData()
    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        return data
    meta = getattr(candidates[0], "grounding_metadata", None)
    if meta is None:
        return data

    for q in getattr(meta, "web_search_queries", None) or []:
        if q and q not in data.queries:
            data.queries.append(q)

    chunk_keys: list[Optional[str]] = []
    for chunk in getattr(meta, "grounding_chunks", None) or []:
        web = getattr(chunk, "web", None)
        uri = getattr(web, "uri", None) if web else None
        if not uri:
            chunk_keys.append(None)
            continue
        key = normalize_url(uri)
        title = (getattr(web, "title", None) or "").strip()
        domain = getattr(web, "domain", None) or (title.lower() if _DOMAIN_LIKE.match(title) else None)
        if key not in data.sources:
            data.sources[key] = GroundedSource(
                key=key, url=uri, title=title or domain or (host_of(uri) or uri), domain=domain
            )
        chunk_keys.append(key)

    for support in getattr(meta, "grounding_supports", None) or []:
        segment = getattr(support, "segment", None)
        text = (getattr(segment, "text", None) or "").strip() if segment else ""
        indices = getattr(support, "grounding_chunk_indices", None) or []
        keys: list[str] = []
        for i in indices:
            if 0 <= i < len(chunk_keys) and chunk_keys[i] and chunk_keys[i] not in keys:
                keys.append(chunk_keys[i])
        if text and keys:
            data.passages.append(GroundedPassage(text=text, source_keys=keys))
    return data


# ---------------------------------------------------------------------------
# Redirect resolution (best effort)
# ---------------------------------------------------------------------------

def default_redirect_resolver(url: str, timeout: float = 5.0) -> Optional[str]:
    """Follow ONE hop of Google's grounding redirect and return its Location
    header — the publisher URL — without fetching the publisher's page."""
    import httpx

    try:
        resp = httpx.get(url, follow_redirects=False, timeout=timeout)
    except Exception:  # noqa: BLE001 - best effort; caller keeps the redirect URL
        return None
    location = resp.headers.get("location")
    if resp.status_code in (301, 302, 303, 307, 308) and location and location.startswith(("http://", "https://")):
        return location
    return None


def resolve_redirects(
    data: GroundingData, resolver: Callable[[str], Optional[str]] = default_redirect_resolver, max_workers: int = 6
) -> GroundingData:
    """Replace Google redirect links with publisher URLs where resolvable,
    re-deduplicating (two redirects can point at the same article)."""
    targets = [s for s in data.sources.values() if is_redirect_url(s.url)]
    if not targets:
        return data
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        resolved = list(pool.map(lambda s: resolver(s.url), targets))
    resolved_by_key = {s.key: r for s, r in zip(targets, resolved)}

    new_sources: dict[str, GroundedSource] = {}
    remap: dict[str, str] = {}
    for key, src in data.sources.items():
        final = resolved_by_key.get(key)
        if final:
            new_key = normalize_url(final)
            host = host_of(final)
            src = GroundedSource(
                key=new_key, url=final,
                title=src.title if src.title and src.title != src.domain else (host or src.title),
                domain=host,
            )
            key_out = new_key
        else:
            key_out = key
        remap[key] = key_out
        new_sources.setdefault(key_out, src)

    passages = []
    for p in data.passages:
        keys: list[str] = []
        for k in p.source_keys:
            nk = remap.get(k, k)
            if nk not in keys:
                keys.append(nk)
        passages.append(GroundedPassage(p.text, keys))
    return GroundingData(queries=list(data.queries), sources=new_sources, passages=passages)
