"""SQLite result cache + lightweight usage metrics for the research layer.

Only validated, successful `ResearchResult`s are ever stored. Usage rows
(request count, hit/miss, latency, search queries, sources, model) feed the
health indicator in the dashboard.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import time
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Optional

from pydantic import ValidationError

from research.config import ResearchConfig
from research.prompts import RESEARCH_PROMPT_VERSION
from research.schemas import QuantContext, ResearchRequest, ResearchResult

_SCHEMA = """
CREATE TABLE IF NOT EXISTS results (
    cache_key TEXT PRIMARY KEY,
    symbol TEXT NOT NULL,
    model TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    payload TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS usage (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    request_id TEXT,
    symbol TEXT,
    outcome TEXT NOT NULL,          -- 'success' | 'error'
    error_code TEXT,
    cache_hit INTEGER NOT NULL,
    latency_ms INTEGER,
    search_queries INTEGER,
    source_count INTEGER,
    model TEXT,
    input_tokens INTEGER DEFAULT 0,
    output_tokens INTEGER DEFAULT 0,
    grounded_searches INTEGER DEFAULT 0,
    finance_searches INTEGER DEFAULT 0
);
"""


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def quant_fingerprint(quant: Optional[QuantContext]) -> str:
    """Identity of the quant state a research result was built against: the
    as-of bar date plus regime/stability. A new bar or a regime change makes
    old research stale; intraday re-runs on the same bar do not."""
    if quant is None or quant.regime_id is None:
        return "no-quant"
    return f"{quant.timestamp:%Y-%m-%d}|{quant.regime_id}|{int(quant.is_confirmed)}"


def make_cache_key(request: ResearchRequest, config: ResearchConfig, fingerprint: str) -> str:
    question = re.sub(r"\s+", " ", (request.question or "").strip().lower())
    parts = {
        "symbol": request.symbol,
        "question": question,
        "lookback_days": request.lookback_days,
        "prompt_version": RESEARCH_PROMPT_VERSION,
        "model": config.model,
        "provider": config.provider,
        "search": config.effective_search_provider,
        "quant": fingerprint,
    }
    return hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()


@dataclass
class CacheStats:
    requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    successes: int = 0
    failures: int = 0
    hit_rate: Optional[float] = None
    last_success_at: Optional[datetime] = None
    last_outcome: Optional[str] = None          # outcome of the most recent request
    last_error_code: Optional[str] = None


@dataclass
class MonthUsage:
    requests: int = 0
    gemini_runs: int = 0            # requests that actually searched (not cache hits)
    input_tokens: int = 0
    output_tokens: int = 0
    grounded_searches: int = 0
    finance_searches: int = 0


class ResearchCache:
    def __init__(
        self, path: Path | str, ttl_minutes: int, clock: Callable[[], datetime] = _utcnow
    ) -> None:
        self.path = Path(path)
        self.ttl = timedelta(minutes=ttl_minutes)
        self._clock = clock
        self._init_done = False

    def _connect(self) -> sqlite3.Connection:
        if not self._init_done:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.path, timeout=10)
        if not self._init_done:
            conn.executescript(_SCHEMA)
            existing = {row[1] for row in conn.execute("PRAGMA table_info(usage)")}
            for column in ("input_tokens", "output_tokens", "grounded_searches", "finance_searches"):
                if column not in existing:  # DBs created before usage tracking existed
                    conn.execute(f"ALTER TABLE usage ADD COLUMN {column} INTEGER DEFAULT 0")
            conn.commit()
            self._init_done = True
        return conn

    # -- results ----------------------------------------------------------

    def get(self, key: str) -> Optional[ResearchResult]:
        """A fresh, valid cached result flagged `cache_hit=True`, else None
        (expired and corrupt entries are deleted)."""
        started = time.monotonic()
        with closing(self._connect()) as conn:
            row = conn.execute("SELECT payload, expires_at FROM results WHERE cache_key = ?", (key,)).fetchone()
            if row is None:
                return None
            payload, expires_at = row
            if datetime.fromisoformat(expires_at) <= self._clock():
                conn.execute("DELETE FROM results WHERE cache_key = ?", (key,))
                conn.commit()
                return None
            try:
                result = ResearchResult.model_validate_json(payload)
            except ValidationError:
                conn.execute("DELETE FROM results WHERE cache_key = ?", (key,))
                conn.commit()
                return None
        return result.model_copy(update={"cache_hit": True, "latency_ms": int((time.monotonic() - started) * 1000)})

    def put(self, key: str, result: ResearchResult) -> None:
        now = self._clock()
        stored = result.model_copy(update={"cache_hit": False})
        with closing(self._connect()) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO results (cache_key, symbol, model, prompt_version, created_at, expires_at, payload)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    key, result.request.symbol, result.diagnostics.model, result.diagnostics.prompt_version,
                    now.isoformat(), (now + self.ttl).isoformat(), stored.model_dump_json(),
                ),
            )
            conn.commit()

    # -- metrics ----------------------------------------------------------

    def record_usage(
        self, *, request_id: str, symbol: str, ok: bool, cache_hit: bool, latency_ms: int,
        search_queries: int = 0, source_count: int = 0, model: str = "", error_code: Optional[str] = None,
        input_tokens: int = 0, output_tokens: int = 0, grounded_searches: int = 0, finance_searches: int = 0,
    ) -> None:
        with closing(self._connect()) as conn:
            conn.execute(
                "INSERT INTO usage (ts, request_id, symbol, outcome, error_code, cache_hit, latency_ms,"
                " search_queries, source_count, model, input_tokens, output_tokens, grounded_searches, finance_searches)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (self._clock().isoformat(), request_id, symbol, "success" if ok else "error", error_code,
                 int(cache_hit), latency_ms, search_queries, source_count, model,
                 input_tokens, output_tokens, grounded_searches, finance_searches),
            )
            conn.commit()

    def month_usage(self, now: Optional[datetime] = None) -> "MonthUsage":
        """Gemini consumption since the start of the current UTC calendar month
        (cache hits and failed runs that still consumed quota are included)."""
        now = now or self._clock()
        month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0).isoformat()
        with closing(self._connect()) as conn:
            runs, fresh, inp, out, searches, finance = conn.execute(
                "SELECT COUNT(*), COALESCE(SUM(cache_hit = 0 AND grounded_searches > 0), 0),"
                " COALESCE(SUM(input_tokens), 0), COALESCE(SUM(output_tokens), 0), COALESCE(SUM(grounded_searches), 0),"
                " COALESCE(SUM(finance_searches), 0) FROM usage WHERE ts >= ?", (month_start,),
            ).fetchone()
        return MonthUsage(requests=runs, gemini_runs=int(fresh), input_tokens=int(inp), output_tokens=int(out),
                          grounded_searches=int(searches), finance_searches=int(finance))

    def stats(self) -> CacheStats:
        with closing(self._connect()) as conn:
            total, hits, ok = conn.execute(
                "SELECT COUNT(*), COALESCE(SUM(cache_hit), 0), COALESCE(SUM(outcome = 'success'), 0) FROM usage"
            ).fetchone()
            last = conn.execute("SELECT outcome, error_code FROM usage ORDER BY id DESC LIMIT 1").fetchone()
            last_ok = conn.execute("SELECT ts FROM usage WHERE outcome = 'success' ORDER BY id DESC LIMIT 1").fetchone()
        successes = int(ok)
        return CacheStats(
            requests=total,
            cache_hits=int(hits),
            cache_misses=total - int(hits),
            successes=successes,
            failures=total - successes,
            hit_rate=(hits / total) if total else None,
            last_success_at=datetime.fromisoformat(last_ok[0]) if last_ok else None,
            last_outcome=last[0] if last else None,
            last_error_code=last[1] if last else None,
        )
