"""Tests for the experimental model adapters (sglang, litellm).

The third-party client packages are stubbed out so these tests exercise the
adapters' request construction without any network access.
"""

import sys
import types

import pytest

from guidance import assistant, gen
from guidance.models.experimental import LiteLLM, SglangModel


class RecordingClient:
    """Stand-in for the client wrapper that records the request kwargs."""

    def __init__(self):
        self.calls: list[dict] = []

    def streaming_chat_completions(self, model, messages, logprobs, **kwargs):
        self.calls.append(kwargs)

        class _EmptyStream:
            def __enter__(self):
                return iter([])

            def __exit__(self, *args):
                return False

        return _EmptyStream()


def _stub_module(monkeypatch: pytest.MonkeyPatch, name: str, **attrs) -> None:
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, name, module)


def _sent_stop(model, **gen_kwargs):
    model._interpreter.client = RecordingClient()
    with assistant():
        model += gen(name="x", **gen_kwargs)
    return model._interpreter.client.calls[-1]["stop"]


def _sglang_model(monkeypatch) -> SglangModel:
    _stub_module(monkeypatch, "openai", OpenAI=lambda **kwargs: types.SimpleNamespace(**kwargs))
    return SglangModel(model="sglang-model", base_url="http://localhost:30000/v1", api_key="none")


def _litellm_model(monkeypatch, endpoint_model: str) -> LiteLLM:
    _stub_module(
        monkeypatch,
        "litellm",
        Router=lambda model_list: types.SimpleNamespace(completion=lambda **kwargs: None),
    )
    return LiteLLM(
        model_description={"model_name": "my-model", "litellm_params": {"model": endpoint_model}},
    )


def test_sglang_gen_stop(monkeypatch):
    model = _sglang_model(monkeypatch)
    assert _sent_stop(model, stop="Word") == "Word"


def test_sglang_gen_stop_regex(monkeypatch):
    model = _sglang_model(monkeypatch)
    assert _sent_stop(model, stop_regex=r"\d+") == r"\d+"


@pytest.mark.parametrize(
    ("endpoint_model", "expected_stop"),
    [
        ("openai/gpt-4o-mini", "Word"),
        ("hosted_vllm/foo", "Word"),
        ("xai/grok-2", ["Word"]),
    ],
)
def test_litellm_gen_stop(monkeypatch, endpoint_model, expected_stop):
    model = _litellm_model(monkeypatch, endpoint_model)
    assert _sent_stop(model, stop="Word") == expected_stop


@pytest.mark.parametrize(
    ("endpoint_model", "expected_stop"),
    [
        ("hosted_vllm/foo", r"\d+"),
        ("xai/grok-2", [r"\d+"]),
    ],
)
def test_litellm_gen_stop_regex(monkeypatch, endpoint_model, expected_stop):
    model = _litellm_model(monkeypatch, endpoint_model)
    assert _sent_stop(model, stop_regex=r"\d+") == expected_stop
