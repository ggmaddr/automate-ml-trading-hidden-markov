"""No-API manual workflow helpers."""

from datetime import datetime, timezone

from research.manual import build_deep_search_prompt, extract_domains, save_manual_report
from research_fakes import make_quant


def test_prompt_has_question_and_no_advice_request():
    p = build_deep_search_prompt("GOOGL", None, 14)
    assert "last 14 days" in p and "GOOGL" in p and "Do not give buy/sell/hold advice" in p
    assert build_deep_search_prompt("GOOGL", "What changed?", 14).startswith("What changed?")


def test_extract_domains_dedupes():
    text = "See https://www.reuters.com/a, https://reuters.com/b and (https://sec.gov/x)."
    assert extract_domains(text) == ["reuters.com", "sec.gov"]


def test_save_manual_report(tmp_path):
    path = save_manual_report("googl", "q?", "Report body", make_quant(), tmp_path,
                              now=datetime(2026, 10, 10, 12, tzinfo=timezone.utc))
    assert path.name == "GOOGL_20261010_120000.md"
    text = path.read_text()
    assert "Report body" in text and "NEUTRAL" in text and "q?" in text
