import pytest

from guidance.models.experimental import SglangModel, VLLMModel

openai = pytest.importorskip("openai")


@pytest.mark.parametrize("model_cls", [VLLMModel, SglangModel])
def test_client_kwargs_reach_openai_client(model_cls):
    model = model_cls(
        model="m",
        base_url="http://localhost:8000/v1",
        api_key="x",
        timeout=60,
        max_retries=2,
    )
    client = model._interpreter.client.client
    assert isinstance(client, openai.OpenAI)
    assert client.timeout == 60
    assert client.max_retries == 2


@pytest.mark.parametrize("model_cls", [VLLMModel, SglangModel])
def test_reasoning_effort_reaches_interpreter(model_cls):
    model = model_cls(
        model="m",
        base_url="http://localhost:8000/v1",
        api_key="x",
        reasoning_effort="low",
    )
    assert model._interpreter.reasoning_effort == "low"
