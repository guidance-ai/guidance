import os
import pathlib

from urllib.parse import parse_qs, urlparse

import pytest

from guidance import assistant, gen, models, system, user

from . import common_chat_testing
from ..utils import env_or_fail

# Everything in here needs credentials to work
# Mark is configured in pyproject.toml
pytestmark = pytest.mark.needs_credentials


def test_azureai_openai_chat_smoke(rate_limiter):
    azureai_endpoint = env_or_fail("AZUREAI_CHAT_ENDPOINT")
    azureai_key = env_or_fail("AZUREAI_CHAT_KEY")
    model = env_or_fail("AZUREAI_CHAT_MODEL")

    lm = models.AzureOpenAI(
        model=model, azure_endpoint=azureai_endpoint, api_key=azureai_key
    )
    assert isinstance(lm, models.AzureOpenAI)

    common_chat_testing.smoke_chat(lm)


def test_azureai_openai_chat_longer_1(rate_limiter):
    azureai_endpoint = env_or_fail("AZUREAI_CHAT_ENDPOINT")
    azureai_key = env_or_fail("AZUREAI_CHAT_KEY")
    model = env_or_fail("AZUREAI_CHAT_MODEL")

    lm = models.AzureOpenAI(
        model=model, azure_endpoint=azureai_endpoint, api_key=azureai_key
    )
    assert isinstance(lm, models.AzureOpenAI)

    common_chat_testing.longer_chat_1(lm)


def test_azureai_openai_chat_longer_2(rate_limiter):
    azureai_endpoint = env_or_fail("AZUREAI_CHAT_ENDPOINT")
    azureai_key = env_or_fail("AZUREAI_CHAT_KEY")
    model = env_or_fail("AZUREAI_CHAT_MODEL")

    lm = models.AzureOpenAI(
        model=model, azure_endpoint=azureai_endpoint, api_key=azureai_key
    )
    assert isinstance(lm, models.AzureOpenAI)

    common_chat_testing.longer_chat_2(lm)


def test_azureai_openai_chat_alt_args(rate_limiter):
    azureai_endpoint = env_or_fail("AZUREAI_CHAT_ENDPOINT")
    azureai_key = env_or_fail("AZUREAI_CHAT_KEY")
    model = env_or_fail("AZUREAI_CHAT_MODEL")

    parsed_url = urlparse(azureai_endpoint)
    parsed_query = parse_qs(parsed_url.query)
    azureai_deployment = pathlib.Path(parsed_url.path).parts[3]
    version = parsed_query["api-version"]
    min_azureai_endpoint = f"{parsed_url.scheme}://{parsed_url.netloc}"

    lm = models.AzureOpenAI(
        model=model,
        azure_endpoint=min_azureai_endpoint,
        version=version,
        api_key=azureai_key,
        azure_deployment=azureai_deployment,
    )

    common_chat_testing.smoke_chat(lm)


def test_azureai_openai_completion_smoke(rate_limiter):
    azureai_endpoint = env_or_fail("AZUREAI_COMPLETION_ENDPOINT")
    azureai_key = env_or_fail("AZUREAI_COMPLETION_KEY")
    model = env_or_fail("AZUREAI_COMPLETION_MODEL")

    print(f"endpoint: {' '.join(azureai_endpoint)}")
    print(f"model: {' '.join(model)}")

    lm = models.AzureOpenAI(
        model=model, azure_endpoint=azureai_endpoint, api_key=azureai_key
    )
    assert isinstance(lm, models.AzureOpenAI)
    assert isinstance(lm.engine, models._openai.OpenAIEngine)

    result = lm + "What is 2+2?" + gen(max_tokens=10, name="text")
    print(f"result: {result['text']}")
    assert len(result["text"]) > 0
    assert lm.engine.metrics.engine_input_tokens > 0
    assert lm.engine.metrics.engine_output_tokens > 0


def test_azureai_openai_completion_alt_args(rate_limiter):
    azureai_endpoint = env_or_fail("AZUREAI_COMPLETION_ENDPOINT")
    azureai_key = env_or_fail("AZUREAI_COMPLETION_KEY")
    model = env_or_fail("AZUREAI_COMPLETION_MODEL")

    parsed_url = urlparse(azureai_endpoint)
    parsed_query = parse_qs(parsed_url.query)
    azureai_deployment = pathlib.Path(parsed_url.path).parts[3]
    version = parsed_query["api-version"]
    min_azureai_endpoint = f"{parsed_url.scheme}://{parsed_url.netloc}"

    lm = models.AzureOpenAI(
        model=model,
        azure_endpoint=min_azureai_endpoint,
        version=version,
        api_key=azureai_key,
        azure_deployment=azureai_deployment,
    )
    assert isinstance(lm, models.AzureOpenAI)
    assert isinstance(lm.engine, models._openai.OpenAIEngine)

    result = lm + "What is 2+2?" + gen(max_tokens=10, name="text")
    print(f"result: {result['text']}")
    assert len(result["text"]) > 0
    assert lm.engine.metrics.engine_input_tokens > 0
    assert lm.engine.metrics.engine_output_tokens > 0


def test_azureai_openai_chat_loop(rate_limiter):
    azureai_endpoint = env_or_fail("AZUREAI_CHAT_ENDPOINT")
    azureai_key = env_or_fail("AZUREAI_CHAT_KEY")
    model = env_or_fail("AZUREAI_CHAT_MODEL")

    lm = models.AzureOpenAI(
        model=model, azure_endpoint=azureai_endpoint, api_key=azureai_key
    )
    assert isinstance(lm, models.AzureOpenAI)

    for i in range(2):
        print(f"Iteration: {i}")
        with system():
            generation = lm + "You will just return whatever number I give you"

        with user():
            generation += f"The number is: {i}"

        with assistant():
            generation += gen(name="answer", max_tokens=2)

        print(str(lm))
        print("\n\n")
