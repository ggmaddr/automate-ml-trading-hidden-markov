"""LLM backend contract: provider-neutral behaviour of the HTTP adapters (no network)."""

import json

import httpx
import pytest

from research.config import load_research_config
from research.exceptions import (
    LLMUnavailableError, ResearchNotConfiguredError, ResearchQuotaError, SchemaValidationError,
)
from research.llm import build_llm
from research.llm.base import Deadline, LLMBackend
from research.llm.ollama import OllamaBackend
from research.schemas import QueryPlanDraft
from research_fakes import make_config

GOOD = {"queries": ["nvda news", "nvda earnings"]}


def ollama(handler, **cfg):
    config = make_config(provider="ollama", model="qwen3:8b", think=False, max_retries=2, **cfg)
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return OllamaBackend(config, client=client, sleep=lambda s: None)


def reply(content, prompt_tokens=100, out_tokens=20):
    return httpx.Response(200, json={"message": {"content": content}, "prompt_eval_count": prompt_tokens, "eval_count": out_tokens})


def call(backend):
    return backend.generate_structured(lambda r: f"plan queries. rejected: {r}" if r else "plan queries", QueryPlanDraft, system="sys", deadline=Deadline(60), stage="t")


def test_ollama_request_is_schema_constrained_with_context_window():
    seen = {}

    def handler(request):
        seen.update(json.loads(request.content), url=str(request.url))
        return reply(json.dumps(GOOD))

    backend = ollama(handler)
    assert call(backend).queries == GOOD["queries"]
    assert seen["url"] == "http://localhost:11434/api/chat"
    assert seen["format"]["properties"]["queries"] and seen["stream"] is False
    assert seen["options"]["num_ctx"] == 16384 and seen["think"] is False
    assert [m["role"] for m in seen["messages"]] == ["system", "user"]
    assert isinstance(backend, LLMBackend)


def test_usage_and_call_counting():
    backend = ollama(lambda r: reply(json.dumps(GOOD), 1000, 50))
    call(backend)
    assert (backend.usage.input_tokens, backend.usage.output_tokens, backend.call_count) == (1000, 50, 1)


def test_fenced_json_and_think_blocks_are_accepted():
    text = "<think>let me plan</think>\n```json\n" + json.dumps(GOOD) + "\n```"
    assert call(ollama(lambda r: reply(text))).queries == GOOD["queries"]


def test_malformed_output_is_repaired_then_succeeds():
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        return reply("not json" if len(bodies) == 1 else json.dumps(GOOD))

    call(ollama(handler))
    assert len(bodies) == 2 and "rejected" in bodies[1]["messages"][1]["content"]


def test_validation_failure_exhausts_retries_without_coercion():
    bad = {"queries": []}   # violates min_length
    backend = ollama(lambda r: reply(json.dumps(bad)))
    with pytest.raises(SchemaValidationError):
        call(backend)
    assert backend.call_count == 3


def test_model_not_pulled_is_actionable_and_not_retried():
    backend = ollama(lambda r: httpx.Response(404, json={"error": "model 'qwen3:8b' not found"}))
    with pytest.raises(LLMUnavailableError) as exc:
        call(backend)
    assert "ollama pull qwen3:8b" in exc.value.message and backend.call_count == 1


def test_server_down_fails_fast_with_start_hint():
    def handler(request):
        raise httpx.ConnectError("refused")

    backend = ollama(handler)
    with pytest.raises(LLMUnavailableError) as exc:
        call(backend)
    assert backend.call_count == 1 and "ollama" in exc.value.message.lower()


def test_rate_limit_is_retried_then_typed():
    backend = ollama(lambda r: httpx.Response(429, json={"error": "slow down"}))
    with pytest.raises(ResearchQuotaError):
        call(backend)
    assert backend.call_count == 3


def test_list_models():
    backend = ollama(lambda r: httpx.Response(200, json={"models": [{"name": "b:1"}, {"name": "a:2"}]}))
    assert backend.list_models() == ["a:2", "b:1"]


def test_build_llm_selects_provider_from_config_only(monkeypatch):
    cfg = load_research_config({"research": {"provider": "ollama", "models": {"ollama": "qwen3:8b", "gemini": "g-model"}}}, env={})
    assert isinstance(build_llm(cfg), OllamaBackend)
    # one env var flips the provider; the model follows it
    flipped = load_research_config({"research": {"provider": "ollama", "models": {"ollama": "qwen3:8b", "gemini": "g-model"}}},
                                   env={"RESEARCH_PROVIDER": "gemini"})
    assert (flipped.provider, flipped.model) == ("gemini", "g-model")
    with pytest.raises(ResearchNotConfiguredError) as exc:
        build_llm(flipped)            # no GEMINI_API_KEY in the test env
    assert "GEMINI_API_KEY" in exc.value.message


def test_provider_without_model_is_a_clear_config_error():
    cfg = load_research_config({"research": {"provider": "openai", "models": {"ollama": "x"}}}, env={})
    with pytest.raises(ResearchNotConfiguredError) as exc:
        build_llm(cfg)
    assert "research.models.openai" in exc.value.message


# ---- hosted providers (mocked HTTP) --------------------------------------------

def hosted(cls, provider, handler, **cfg):
    config = make_config(provider=provider, model="some-model", max_retries=1, **cfg)
    return cls(config, client=httpx.Client(transport=httpx.MockTransport(handler)), sleep=lambda s: None)


def test_openai_compat_request_shape_auth_and_usage(monkeypatch):
    from research.llm.openai_compat import OpenAICompatBackend

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key-123456789012345678")
    seen = {}

    def handler(request):
        seen.update(json.loads(request.content), auth=request.headers["authorization"], url=str(request.url))
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(GOOD)}}],
                                         "usage": {"prompt_tokens": 30, "completion_tokens": 7}})

    backend = hosted(OpenAICompatBackend, "openai", handler)
    assert call(backend).queries == GOOD["queries"]
    assert seen["url"] == "https://api.openai.com/v1/chat/completions"
    assert seen["auth"] == "Bearer sk-test-key-123456789012345678"
    assert seen["response_format"] == {"type": "json_object"} and "JSON Schema" in seen["messages"][1]["content"]
    assert (backend.usage.input_tokens, backend.usage.output_tokens) == (30, 7)


def test_openai_compatible_custom_server_needs_no_key(monkeypatch):
    from research.llm.openai_compat import OpenAICompatBackend

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    cfg = load_research_config({"research": {"provider": "openai", "models": {"openai": "local-model"},
                                             "base_url": "http://localhost:1234"}}, env={})
    backend = build_llm(cfg)
    assert isinstance(backend, OpenAICompatBackend) and backend.base_url == "http://localhost:1234"
    with pytest.raises(ResearchNotConfiguredError):          # without a custom server a key is required
        build_llm(load_research_config({"research": {"provider": "openai", "models": {"openai": "m"}}}, env={}))


def test_anthropic_forced_tool_roundtrip_and_headers(monkeypatch):
    from research.llm.anthropic import AnthropicBackend

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-1234567890123456789")
    seen = {}

    def handler(request):
        seen.update(json.loads(request.content), key=request.headers["x-api-key"], ver=request.headers["anthropic-version"])
        return httpx.Response(200, json={"content": [{"type": "tool_use", "name": "submit_result", "input": GOOD}],
                                         "usage": {"input_tokens": 50, "output_tokens": 9}})

    backend = hosted(AnthropicBackend, "anthropic", handler)
    assert call(backend).queries == GOOD["queries"]
    assert seen["tool_choice"] == {"type": "tool", "name": "submit_result"} and seen["max_tokens"] > 0
    assert seen["tools"][0]["input_schema"]["properties"]["queries"] and seen["key"].startswith("sk-ant-")
    assert seen["system"] == "sys" and seen["ver"]
    assert (backend.usage.input_tokens, backend.usage.output_tokens) == (50, 9)


@pytest.mark.parametrize("status,code", [(401, "auth_failed"), (404, "bad_request"), (500, "unavailable")])
def test_hosted_http_errors_map_to_typed_errors(monkeypatch, status, code):
    from research.exceptions import ResearchError
    from research.llm.anthropic import AnthropicBackend

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-1234567890123456789")
    backend = hosted(AnthropicBackend, "anthropic", lambda r: httpx.Response(status, json={"error": {"message": "x"}}))
    with pytest.raises(ResearchError) as exc:
        call(backend)
    assert exc.value.code == code


def test_keys_are_redacted_everywhere(monkeypatch):
    from research.logging_util import redact

    monkeypatch.setenv("OPENAI_API_KEY", "sk-proj-abcdefghijklmnopqrstuvwxyz0123")
    assert "abcdefghij" not in redact("failed with key sk-proj-abcdefghijklmnopqrstuvwxyz0123")
    assert "abcdefghij" not in redact("header x-api-key: sk-ant-abcdefghijklmnopqrstuvwxyz0123")
