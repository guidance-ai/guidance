import pytest

import pytest

from guidance import models, gen


from . import common_chat_testing
from ..utils import env_or_fail

# Everything in here needs credentials to work
# Mark is configured in pyproject.toml
pytestmark = pytest.mark.needs_credentials

# How to fill out the environment variables to
# set up the models
_chat_models = {"phi3": "PHI3", "mistral": "MISTRAL_CHAT", "llama3": "LLAMA3_CHAT"}

_completion_models = {"mistral": "MISTRAL_COMPLETION"}


def _get_azure_ai_model(env_string: str, endpoint_type: str):

    azureai_studio_endpoint = env_or_fail(f"AZURE_AI_STUDIO_{env_string}_ENDPOINT")
    azureai_studio_deployment = env_or_fail(f"AZURE_AI_STUDIO_{env_string}_DEPLOYMENT")
    azureai_studio_key = env_or_fail(f"AZURE_AI_STUDIO_{env_string}_KEY")

    lm = models.AzureAIStudio(
        azureai_studio_endpoint=azureai_studio_endpoint,
        azureai_studio_deployment=azureai_studio_deployment,
        azureai_studio_key=azureai_studio_key,
        endpoint_type=endpoint_type,
    )
    assert isinstance(lm, models.AzureAIStudio)
    return lm


def _get_chat_model(model_name: str):
    env_string = _chat_models[model_name]

    return _get_azure_ai_model(env_string, "chat")


def _get_completion_model(model_name: str):
    env_string = _completion_models[model_name]

    return _get_azure_ai_model(env_string, "completion")


@pytest.mark.parametrize("chat_model_name", _chat_models.keys())
def test_azureai_chat_smoke(rate_limiter, chat_model_name: str):
    lm = _get_chat_model(chat_model_name)

    common_chat_testing.smoke_chat(lm, chat_model_name != "mistral")


@pytest.mark.parametrize("chat_model_name", _chat_models.keys())
def test_azureai_chat_longer_1(rate_limiter, chat_model_name: str):
    lm = _get_chat_model(chat_model_name)
    common_chat_testing.longer_chat_1(lm, chat_model_name != "mistral")


@pytest.mark.parametrize("chat_model_name", _chat_models.keys())
def test_azureai_chat_longer_2(rate_limiter, chat_model_name: str):
    lm = _get_chat_model(chat_model_name)
    common_chat_testing.longer_chat_2(lm, chat_model_name != "mistral")


@pytest.mark.parametrize("completion_model_name", _completion_models.keys())
def test_azureai_completion_cmoke(rate_limiter, completion_model_name: str):
    lm = _get_completion_model(completion_model_name)

    lm += "What is 2+2?"
    lm += gen(max_tokens=4, name="text", temperature=0.5)

    print(str(lm))
    print(f"{lm.engine.metrics=}")
    assert len(lm["text"]) > 0
    assert lm.engine.metrics.engine_input_tokens > 2, "Expect some input tokens"
    assert lm.engine.metrics.engine_output_tokens > 0, "Expect some output tokens"
