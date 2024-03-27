import os

import pytest

from guidance import models, gen, system, user, assistant


# Everything in here needs credentials to work
# Mark is configured in pyproject.toml
pytestmark = pytest.mark.needs_credentials


def _env_or_fail(var_name: str) -> str:
    env_value = os.getenv(var_name, None)

    assert env_value is not None, f"Env '{var_name}' not found"

    return env_value


def test_azureai_openai_chat_smoke(rate_limiter):
    azureai_endpoint = _env_or_fail("AZUREAI_CHAT_ENDPOINT")
    azureai_key = _env_or_fail("AZUREAI_CHAT_KEY")
    model = _env_or_fail("AZUREAI_CHAT_MODEL")

    lm = models.AzureOpenAI(
        model=model, azure_endpoint=azureai_endpoint, api_key=azureai_key
    )
    assert isinstance(lm, models.AzureOpenAIChat)

    with system():
        lm += "You are a math wiz."

    with user():
        lm += "What is 1 + 1?"

    with assistant():
        lm += gen(max_tokens=10, name="text")
        lm += "Pick a number: "

    print(str(lm))
    assert len(lm["text"]) > 0


def test_azureai_openai_completion_smoke(rate_limiter):
    azureai_endpoint = _env_or_fail("AZUREAI_COMPLETION_ENDPOINT")
    azureai_key = _env_or_fail("AZUREAI_COMPLETION_KEY")
    model = _env_or_fail("AZUREAI_COMPLETION_MODEL")

    lm = models.AzureOpenAI(
        model=model, azure_endpoint=azureai_endpoint, api_key=azureai_key
    )
    assert isinstance(lm, models.AzureOpenAICompletion)

    result = lm + "What is 2+2?" + gen(max_tokens=10, name="text")
    print(f"result: {result['text']}")
    assert len(result["text"]) > 0


def test_azureai_openai_chat_loop(rate_limiter):
    azureai_endpoint = _env_or_fail("AZUREAI_CHAT_ENDPOINT")
    azureai_key = _env_or_fail("AZUREAI_CHAT_KEY")
    model = _env_or_fail("AZUREAI_CHAT_MODEL")

    lm = models.AzureOpenAI(
        model=model, azure_endpoint=azureai_endpoint, api_key=azureai_key
    )
    assert isinstance(lm, models.AzureOpenAIChat)

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
