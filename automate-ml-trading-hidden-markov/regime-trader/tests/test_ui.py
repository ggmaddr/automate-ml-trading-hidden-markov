"""UI layer: design tokens, escaping, gauges, strands, and page smoke tests."""

import math

import numpy as np
import pandas as pd
import pytest

from ui import html as h
from ui import svg, tokens
from ui.charts import correlation_grid
from ui.widgets.portfolio_field import build_strands, strand_table


@pytest.mark.parametrize("name,fg,bg", tokens.TEXT_PAIRS)
def test_text_colors_meet_wcag_aa(name, fg, bg):
    assert tokens.contrast_ratio(fg, bg) >= 4.5, name


def test_contrast_helper_known_values():
    assert tokens.contrast_ratio("#000000", "#FFFFFF") == pytest.approx(21.0)
    assert tokens.contrast_ratio("#777777", "#777777") == pytest.approx(1.0)


EVIL = '<script>alert(1)</script><img src=x onerror=alert(1)>"\''


def test_builders_escape_untrusted_text():
    out = "".join([
        h.eyebrow(EVIL), h.hero(EVIL), h.kpi(EVIL, EVIL, EVIL), h.pill(EVIL), h.notice(EVIL, title=EVIL),
        h.bullets([EVIL]), h.tile("", EVIL, EVIL, EVIL),
        h.evidence_item(EVIL, EVIL, EVIL, 0.5, EVIL, "https://a.com/x", EVIL),
    ])
    assert "<script" not in out and "<img" not in out and "onerror=alert" not in out.replace("&quot;", "").replace("&#x27;", "") or "&lt;" in out
    assert "&lt;script&gt;" in out


@pytest.mark.parametrize("url", ["javascript:alert(1)", "data:text/html,x", "ftp://x.com", "//evil.com", "https://a.com/x y", 'https://a.com/"onmouseover=', ""])
def test_unsafe_urls_become_inert(url):
    assert h.safe_url(url) == "#"
    assert 'href="#"' in h.source_chip("t", url)


def test_safe_url_and_link_hardening():
    chip = h.source_chip("Reuters", "https://www.reuters.com/a?x=1&y=2")
    assert 'href="https://www.reuters.com/a?x=1&amp;y=2"' in chip and 'rel="noopener noreferrer nofollow"' in chip


def test_gauge_fraction_clamps_and_tones():
    assert svg.gauge_fraction(0.015, 0.03) == pytest.approx(0.5)
    assert svg.gauge_fraction(5, 1) == 1.0 and svg.gauge_fraction(-1, 1) == 0.0 and svg.gauge_fraction(1, 0) == 0.0
    assert [svg.gauge_tone(f) for f in (0.0, 0.49, 0.5, 0.89, 0.9, 1.0)] == ["ok", "ok", "warn", "warn", "halt", "halt"]


def test_gauge_markup_is_safe_and_accessible():
    g = svg.gauge(0.5, EVIL, EVIL, EVIL, "shield")
    assert "<script" not in g and 'role="img"' in g and "aria-label=" in g and "data:image/svg+xml;base64," in g
    assert "<svg" not in g          # the SVG is embedded as an image, never inline


def test_gauge_needle_angle():
    import base64, re
    for fraction, expected_dx_sign in ((0.0, -1), (0.5, 0), (1.0, 1)):
        uri = re.search(r"base64,([A-Za-z0-9+/=]+)", svg.gauge(fraction, "t", "v", "c")).group(1)
        markup = base64.b64decode(uri).decode()
        x2 = float(re.search(r'<line x1="130.0" y1="124.0" x2="([\d.]+)"', markup).group(1))
        assert (x2 > 130.5) - (x2 < 129.5) == expected_dx_sign


def _equity(seed, n=300, drift=0.0004):
    rng = np.random.default_rng(seed)
    return pd.Series(100 * np.exp(np.cumsum(rng.normal(drift, 0.01, n))), index=pd.bdate_range("2021-01-01", periods=n))


def test_strands_are_deterministic_bounded_and_real_data_driven():
    series = {"PORTFOLIO": _equity(1), "SPY": _equity(2)}
    a, b = build_strands(series), build_strands(series)
    assert a == b and len(a) == 96 and all(len(s) == 48 for s in a)
    assert max(abs(v) for s in a for v in s) == pytest.approx(1.0) and all(math.isfinite(v) for s in a for v in s)
    assert build_strands({"PORTFOLIO": _equity(9)}) != a                       # different data -> different geometry
    assert build_strands({}) == [] and build_strands({"X": _equity(1, n=5)}) == []


def test_strand_table_fallback():
    table = strand_table({"PORTFOLIO": _equity(1), "SPY": _equity(2)})
    assert list(table["series"]) == ["PORTFOLIO", "SPY"] and "total return" in table.columns


def test_correlation_grid_escapes_and_shades():
    corr = pd.DataFrame([[1.0, 0.3], [0.3, 1.0]], index=["A<b>", "B"], columns=["A<b>", "B"])
    html = correlation_grid(corr)
    assert "&lt;b&gt;" in html and "+1.00" in html and "+0.30" in html and "<b>" not in html


# ---- pages ------------------------------------------------------------------------

@pytest.fixture
def app():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file("dashboard_app.py", default_timeout=90)
    at.session_state["auto_refresh"] = False     # a running fragment loop would never finish under AppTest
    return at


@pytest.mark.parametrize("page", ["views/live.py", "views/backtest.py", "views/research.py", "views/methodology.py"])
def test_every_page_renders_without_exceptions(app, page):
    app.run()
    if page != "views/live.py":
        app.switch_page(page).run()
    assert not app.exception, [e.value for e in app.exception]


def test_methodology_numbers_come_from_settings(app):
    import yaml

    settings = yaml.safe_load(open("config/settings.yaml"))
    app.run()
    app.switch_page("views/methodology.py").run()
    page_html = " ".join(el.proto.body for el in app.get("html"))
    assert f"{settings['risk']['max_leverage']:.2f}x" in page_html
    assert f"{settings['hmm']['stability_bars']} bars to confirm" in page_html
