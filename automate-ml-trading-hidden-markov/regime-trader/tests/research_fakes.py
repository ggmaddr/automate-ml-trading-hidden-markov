"""Test doubles for the research layer: a scripted fake Gemini client built
on recorded response fixtures, plus stub tool readers. No network, no quota."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from google.genai import errors, types

from research.config import ResearchConfig
from research.schemas import BacktestContext, QuantContext, RegimeHistory, RegimeHistoryPoint

FIXTURES = Path(__file__).parent / "fixtures"


def load_json(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text())


def grounded_response(name: str = "grounded_round1.json") -> types.GenerateContentResponse:
    return types.GenerateContentResponse.model_validate(load_json(name))


def ungrounded_response(text: str = "Some answer from model memory.") -> types.GenerateContentResponse:
    return types.GenerateContentResponse.model_validate(
        {"candidates": [{"content": {"role": "model", "parts": [{"text": text}]}}]}
    )


def text_response(text: str) -> types.GenerateContentResponse:
    return ungrounded_response(text)


def api_error(code: int, message: str = "boom") -> errors.APIError:
    cls = errors.ClientError if code < 500 else errors.ServerError
    return cls(code, {"error": {"code": code, "message": message, "status": "X"}})


class FakeModels:
    """Scripted `client.models`. Queues per call kind: 'search', 'brief',
    'comparison'. An item is a response, JSON text, a dict (-> JSON), or an
    Exception (raised). The last item repeats once a queue runs dry."""

    def __init__(self, script: dict[str, list[Any]]) -> None:
        self.script = {k: list(v) for k, v in script.items()}
        self.calls: list[dict] = []

    @staticmethod
    def _kind(config: types.GenerateContentConfig) -> str:
        if config.tools:
            return "search"
        schema = getattr(config.response_schema, "__name__", "")
        return {"BriefDraft": "brief", "ComparisonDraft": "comparison"}[schema]

    def generate_content(self, *, model: str, contents: str, config: types.GenerateContentConfig):
        kind = self._kind(config)
        self.calls.append({"kind": kind, "model": model, "contents": contents, "config": config})
        queue = self.script[kind]
        item = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(item, Exception):
            raise item
        if isinstance(item, dict):
            item = json.dumps(item)
        if isinstance(item, str):
            return text_response(item)
        return item

    def kinds(self) -> list[str]:
        return [c["kind"] for c in self.calls]


class FakeGeminiClient:
    def __init__(self, script: Optional[dict[str, list[Any]]] = None) -> None:
        self.models = FakeModels(script or happy_script())


def happy_script() -> dict[str, list[Any]]:
    return {
        "search": [grounded_response()],
        "brief": [load_json("brief_draft.json")],
        "comparison": [load_json("comparison_draft.json")],
    }


class StubHMMReader:
    """Stands in for HMMRuntimeReader; records calls, never touches a model."""

    def __init__(self, quant: Optional[QuantContext] = None, fail: bool = False) -> None:
        self.quant = quant or make_quant()
        self.fail = fail
        self.calls: list[str] = []

    def get_hmm_snapshot(self, symbol: str) -> QuantContext:
        from research.exceptions import QuantContextUnavailableError

        self.calls.append("get_hmm_snapshot")
        if self.fail:
            raise QuantContextUnavailableError("no model")
        return self.quant.model_copy(update={"symbol": symbol})

    def get_regime_history(self, symbol: str, bars: int = 60) -> RegimeHistory:
        self.calls.append("get_regime_history")
        point = RegimeHistoryPoint(
            timestamp=datetime(2026, 10, 9), regime_id=1, label="NEUTRAL", probability=0.8,
            expected_volatility=0.01, is_confirmed=True, consecutive_bars=5, flicker_rate=0.0, is_flickering=False,
        )
        return RegimeHistory(symbol=symbol, points=[point])


class StubBacktestReader:
    def __init__(self, fail: bool = False, available: bool = True) -> None:
        self.fail = fail
        self.available = available
        self.calls: list[str] = []

    def get_backtest_summary(self, symbol: str) -> BacktestContext:
        self.calls.append("get_backtest_summary")
        if self.fail:
            raise RuntimeError("csv exploded")
        return BacktestContext(symbol=symbol, available=self.available, total_return=0.2 if self.available else None)


def make_quant(**overrides) -> QuantContext:
    base = dict(
        symbol="GOOGL", timestamp=datetime(2026, 10, 9), regime_id=1, regime_label="NEUTRAL",
        regime_probability=0.82, state_probabilities={"BEAR": 0.1, "NEUTRAL": 0.82, "BULL": 0.08},
        expected_volatility=0.011, expected_return=0.0004, volatility_rank=0.5, is_confirmed=True,
        consecutive_bars=12, flicker_rate=1.0, is_flickering=False, model_version="hmm-k3-trained-2026-10-04",
        feature_version="14f-abcd1234", recent_return=0.021, recent_volatility=0.012,
    )
    base.update(overrides)
    return QuantContext(**base)


def make_config(**overrides) -> ResearchConfig:
    base = dict(model="test-model", max_retries=2, request_timeout_seconds=30, overall_timeout_seconds=120,
                resolve_source_redirects=True)
    base.update(overrides)
    return ResearchConfig(**base)


def fake_resolver(url: str) -> Optional[str]:
    return {"https://vertexaisearch.cloud.google.com/grounding-api-redirect/ABC123": "https://www.cnbc.com/2026/10/02/alphabet-probe.html"}.get(url)
