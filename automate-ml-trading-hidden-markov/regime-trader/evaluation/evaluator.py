"""Small evaluation harness for the research agent (Phase 10, section 16).

    python -m evaluation.evaluator run   [--provider ollama] [--model M] [--limit N] [--symbols GOOGL NVDA] [--force-refresh]
    python -m evaluation.evaluator score [--provider ollama] [--reference-dir evaluation/google_finance_reference]

`run` sends each benchmark question through the real ResearchOrchestrator
(spends Gemini quota) and saves every ResearchResult as JSON under
evaluation/results/<provider>/. `score` computes engineering/research-quality metrics
from those files — it never scores investment recommendations.

Automatically measured: source coverage, domain diversity, source freshness
(only where a publication date was supplied), citation completeness, schema
completeness, contradiction/unknown capture, latency, and — when a manually
saved Google Finance report `<SYMBOL>_<n>.txt` (n = question number) exists —
URL/domain overlap with that reference. Material-event recall and
unsupported-claim rate need a human reading both reports; the README
describes that rubric.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

from research.schemas import ResearchResult

EVAL_DIR = Path(__file__).resolve().parent
QUESTIONS_PATH = EVAL_DIR / "research_questions.json"
RESULTS_DIR = EVAL_DIR / "results"
REFERENCE_DIR = EVAL_DIR / "google_finance_reference"
_URL_RE = re.compile(r"https?://[^\s)\]>\"']+")


def load_questions(path: Path = QUESTIONS_PATH) -> list[dict]:
    return json.loads(path.read_text())


def _domain(url: str) -> str:
    host = urlparse(url).hostname or ""
    return host[4:] if host.startswith("www.") else host


def score_result(result: ResearchResult, reference_text: Optional[str] = None, now: Optional[datetime] = None) -> dict:
    now = now or datetime.now(timezone.utc)
    brief = result.brief
    evidence = brief.catalysts + brief.risks + brief.recent_events
    domains = {e.source_domain or _domain(e.source_url) for e in evidence}
    dated = [e for e in evidence if e.published_at]
    ages = []
    for e in dated:
        published = e.published_at if e.published_at.tzinfo else e.published_at.replace(tzinfo=timezone.utc)
        ages.append((now - published).days)

    checks = [
        bool(brief.summary), bool(evidence), bool(brief.search_queries), brief.source_count > 0,
        brief.unknowns is not None, brief.evidence_stance != "insufficient_evidence",
    ]
    metrics = {
        "symbol": brief.symbol,
        "question": brief.question,
        "source_count": brief.source_count,
        "evidence_items": len(evidence),
        "distinct_evidence_domains": len(domains),
        "median_evidence_age_days": sorted(ages)[len(ages) // 2] if ages else None,
        "dated_evidence_fraction": (len(dated) / len(evidence)) if evidence else None,
        "citation_completeness": (sum(1 for e in evidence if e.source_url and e.source_title) / len(evidence)) if evidence else None,
        "conflicts_reported": len(brief.conflicting_evidence),
        "unknowns_reported": len(brief.unknowns),
        "schema_completeness": sum(checks) / len(checks),
        "dropped_unsupported_claims": result.diagnostics.dropped_unsupported_claims,
        "latency_ms": result.latency_ms,
        "llm_calls": result.diagnostics.llm_calls,
    }
    if reference_text is not None:
        ref_urls = {u.rstrip(".,") for u in _URL_RE.findall(reference_text)}
        ref_domains = {_domain(u) for u in ref_urls}
        ours = {e.source_domain or _domain(e.source_url) for e in evidence}
        metrics["reference_domain_overlap"] = len(ours & ref_domains) / len(ref_domains) if ref_domains else None
        metrics["reference_domains"] = len(ref_domains)
    return metrics


def cmd_run(args: argparse.Namespace) -> int:
    from dotenv import load_dotenv

    import os

    from research.config import load_research_config
    from research.orchestrator import ResearchOrchestrator
    from research.schemas import ResearchRequest

    load_dotenv()
    env = dict(os.environ)
    if args.provider:
        env["RESEARCH_PROVIDER"] = args.provider
    if args.model:
        env["RESEARCH_MODEL"] = args.model
    config = load_research_config(env=env)
    out_dir = RESULTS_DIR / config.provider
    questions = load_questions()
    if args.symbols:
        wanted = {s.upper() for s in args.symbols}
        questions = [q for q in questions if q["symbol"] in wanted]
    if args.limit:
        questions = questions[: args.limit]

    out_dir.mkdir(parents=True, exist_ok=True)
    orchestrator = ResearchOrchestrator(config)
    failures = 0
    for number, q in enumerate(load_questions(), start=1):
        if q not in questions:
            continue
        outcome = orchestrator.research(ResearchRequest(symbol=q["symbol"], question=q["question"], force_refresh=args.force_refresh))
        out_path = out_dir / f"{q['symbol']}_{number}.json"
        if outcome.ok:
            out_path.write_text(outcome.result.model_dump_json(indent=2))
            print(f"[ok]   {out_path.name}  sources={outcome.result.brief.source_count} latency={outcome.result.latency_ms}ms")
        else:
            failures += 1
            print(f"[fail] {q['symbol']} #{number}: {outcome.error.code} — {outcome.error.message}")
    return 1 if failures else 0


def cmd_score(args: argparse.Namespace) -> int:
    reference_dir = Path(args.reference_dir)
    rows = []
    pattern = f"{args.provider}/*.json" if args.provider else "*/*.json"
    for path in sorted(RESULTS_DIR.glob(pattern)):
        result = ResearchResult.model_validate_json(path.read_text())
        ref_path = reference_dir / f"{path.stem}.txt"
        reference = ref_path.read_text() if ref_path.exists() else None
        row = score_result(result, reference)
        row["file"] = path.name
        row["provider"] = path.parent.name
        row["has_reference"] = reference is not None
        rows.append(row)
    if not rows:
        print(f"No results in {RESULTS_DIR}. Run `python -m evaluation.evaluator run` first.")
        return 1
    print(json.dumps(rows, indent=2, default=str))
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="evaluation.evaluator")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="Run benchmark questions through the research agent")
    run.add_argument("--provider", choices=["ollama", "openai", "anthropic", "gemini"])
    run.add_argument("--model")
    run.add_argument("--limit", type=int)
    run.add_argument("--symbols", nargs="+")
    run.add_argument("--force-refresh", action="store_true")
    run.set_defaults(func=cmd_run)
    score = sub.add_parser("score", help="Score saved results (optionally against Google Finance references)")
    score.add_argument("--provider", choices=["ollama", "openai", "anthropic", "gemini"])
    score.add_argument("--reference-dir", default=str(REFERENCE_DIR))
    score.set_defaults(func=cmd_score)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
