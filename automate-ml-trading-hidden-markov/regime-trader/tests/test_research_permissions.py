"""Safety boundaries: the research layer cannot trade or change strategy/risk/HMM settings."""

import ast
import hashlib
from pathlib import Path

import pytest

import research
from research.tools import ALLOWED_RESEARCH_TOOLS, ResearchToolRegistry
from research_fakes import StubBacktestReader, StubHMMReader

RESEARCH_DIR = Path(research.__file__).parent
SETTINGS = Path(__file__).resolve().parents[1] / "config" / "settings.yaml"

# Modules the research package must never import: order submission, broker clients,
# live trading orchestration, and anything that can mutate strategy/risk state.
FORBIDDEN_PREFIXES = (
    "broker", "alpaca", "alpaca_trade_api", "core.trading_bot", "core.risk_manager",
    "core.regime_strategies", "core.signal_generator", "monitoring.alerts",
)
MUTATION_WORDS = ("submit", "order", "cancel", "place", "liquidate", "set_", "update_", "modify", "write", "execute", "trade")


def _imports(path: Path) -> set[str]:
    names = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            names |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
            names |= {f"{node.module}.{a.name}" for a in node.names}
    return names


def test_research_agent_has_no_order_tools():
    assert ALLOWED_RESEARCH_TOOLS == {"get_hmm_snapshot", "get_regime_history", "get_backtest_summary"}
    reg = ResearchToolRegistry(StubHMMReader(), StubBacktestReader())
    assert reg.tool_names <= ALLOWED_RESEARCH_TOOLS
    for name in reg.tool_names:
        assert name.startswith("get_") and not any(w in name for w in MUTATION_WORDS)
    for forbidden in ("submit_order", "cancel_order", "place_order", "close_position", "liquidate_all"):
        assert forbidden not in reg.tool_names


def test_research_package_never_imports_broker_or_trading_modules():
    offenders = {}
    for path in RESEARCH_DIR.rglob("*.py"):
        bad = {m for m in _imports(path) if m.split(".")[0] in {p.split(".")[0] for p in FORBIDDEN_PREFIXES if "." not in p}
               or any(m == p or m.startswith(p + ".") for p in FORBIDDEN_PREFIXES)}
        if bad:
            offenders[str(path.relative_to(RESEARCH_DIR))] = bad
    assert offenders == {}


def test_research_agent_cannot_modify_strategy(tmp_path):
    reg = ResearchToolRegistry(StubHMMReader(), StubBacktestReader())
    before = hashlib.sha256(SETTINGS.read_bytes()).hexdigest()
    for tool in ("set_strategy_param", "update_allocation", "promote_strategy_version"):
        with pytest.raises(Exception):
            reg.call(tool)
    reg.call("get_hmm_snapshot", symbol="SPY")
    assert hashlib.sha256(SETTINGS.read_bytes()).hexdigest() == before
    assert not any("regime_strategies" in m for p in RESEARCH_DIR.rglob("*.py") for m in _imports(p))


def test_research_agent_cannot_modify_risk_limits():
    reg = ResearchToolRegistry(StubHMMReader(), StubBacktestReader())
    for tool in ("set_risk_limit", "disable_circuit_breaker", "update_risk_config"):
        with pytest.raises(Exception):
            reg.call(tool)
    assert not any("risk_manager" in m for p in RESEARCH_DIR.rglob("*.py") for m in _imports(p))


def test_research_does_not_modify_hmm_parameters(tmp_path, trained_hmm, synthetic_ohlcv):
    from research.tools.hmm_tools import HMMRuntimeReader

    path = tmp_path / "m.pkl"
    trained_hmm.save(str(path))
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    reader = HMMRuntimeReader(path, bars_provider=lambda s: synthetic_ohlcv)
    reader.get_hmm_snapshot("SPY")
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
    # the reader loads its OWN engine copy: the shared engine's stability state is untouched
    trained_hmm._reset_stability_state()
    before_history = list(trained_hmm._raw_state_history)
    reader.get_regime_history("SPY")
    assert trained_hmm._raw_state_history == before_history


def test_gemini_wrapper_exposes_no_function_calling():
    """Only Google Search / URL context are ever passed as tools — never function declarations."""
    from research_fakes import FakeGeminiClient, make_config
    from research.agent import Deadline, GeminiAgent

    client = FakeGeminiClient()
    GeminiAgent(make_config(), client=client).grounded_search("p", system="s", deadline=Deadline(60))
    for tool in client.models.calls[0]["config"].tools:
        assert tool.function_declarations is None and (tool.google_search or tool.url_context)


def test_secrets_not_sent_to_gemini(monkeypatch):
    from research.prompts import build_grounded_search_prompt, summarize_quant_line
    from research_fakes import make_quant
    from datetime import date

    monkeypatch.setenv("ALPACA_SECRET_KEY", "super-secret-alpaca")
    prompt = build_grounded_search_prompt("GOOGL", "q", 14, summarize_quant_line(make_quant()), date(2026, 10, 10))
    assert "super-secret-alpaca" not in prompt
