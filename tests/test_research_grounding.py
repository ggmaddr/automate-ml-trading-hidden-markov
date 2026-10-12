"""Grounding extraction from recorded Gemini response fixtures."""

from research.grounding import GroundingData, extract_grounding, normalize_url, resolve_redirects
from research_fakes import fake_resolver, grounded_response, ungrounded_response


def test_sources_extracted_and_deduplicated_from_grounding_metadata():
    data = extract_grounding(grounded_response())
    urls = [s.url for s in data.sources.values()]
    # 4 chunks, but chunks 0 and 2 are the same article (utm param / www / scheme differences)
    assert len(urls) == 3
    assert any("sec.gov" in u for u in urls)
    assert data.sources[normalize_url("https://reuters.com/technology/alphabet-results-2026-10-01")].domain == "reuters.com"


def test_queries_retained_and_passages_linked_to_sources():
    data = extract_grounding(grounded_response())
    assert data.queries == ["GOOGL latest earnings results", "Alphabet regulatory probe EU ad tech"]
    assert len(data.passages) == 3
    assert len(data.passages[0].source_keys) == 2


def test_source_counts_and_ids_are_deterministic():
    a = extract_grounding(grounded_response()).finalize(12)
    b = extract_grounding(grounded_response()).finalize(12)
    assert [(s.source_id, s.url) for s in a.sources.values()] == [(s.source_id, s.url) for s in b.sources.values()]
    assert list(a.source_by_id()) == ["S1", "S2", "S3"]


def test_finalize_caps_sources_and_prunes_passages():
    data = extract_grounding(grounded_response()).finalize(1)
    assert len(data.sources) == 1
    assert all(len(p.source_keys) == 1 for p in data.passages)


def test_response_without_metadata_has_no_grounding():
    data = extract_grounding(ungrounded_response("model memory"))
    assert not data.has_grounding and data.sources == {} and data.queries == []


def test_merge_dedupes_across_rounds():
    a, b = extract_grounding(grounded_response()), extract_grounding(grounded_response())
    a.merge(b)
    assert len(a.sources) == 3 and len(a.queries) == 2


def test_redirects_resolved_and_rededuplicated():
    data = extract_grounding(grounded_response())
    resolved = resolve_redirects(data, resolver=fake_resolver)
    urls = [s.url for s in resolved.sources.values()]
    assert "https://www.cnbc.com/2026/10/02/alphabet-probe.html" in urls
    assert not any("vertexaisearch" in u for u in urls)
    # a resolver that fails keeps the original (still clickable) link
    kept = resolve_redirects(extract_grounding(grounded_response()), resolver=lambda u: None)
    assert any("vertexaisearch" in s.url for s in kept.sources.values())
    # two redirects resolving to the same article collapse to one source
    same = resolve_redirects(extract_grounding(grounded_response()), resolver=lambda u: "https://sec.gov/Archives/edgar/data/1652044/10q.htm")
    assert len(same.sources) == 2
