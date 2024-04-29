import pytest

import pytest

from guidance import assistant, gen, models, system, user

from ..utils import env_or_fail

# Everything in here needs credentials to work
# Mark is configured in pyproject.toml
pytestmark = pytest.mark.needs_credentials


def test_azureai_phi3_chat_smoke(rate_limiter):
    azureai_studio_endpoint = env_or_fail("AZURE_AI_STUDIO_PHI3_ENDPOINT")
    azureai_studio_deployment = env_or_fail("AZURE_AI_STUDIO_PHI3_DEPLOYMENT")
    azureai_studio_key = env_or_fail("AZURE_AI_STUDIO_PHI3_KEY")

    lm = models.AzureAIStudioChat(
        azureai_studio_endpoint=azureai_studio_endpoint,
        azureai_studio_deployment=azureai_studio_deployment,
        azureai_studio_key=azureai_studio_key,
    )
    assert isinstance(lm, models.AzureAIStudioChat)

    with system():
        lm += "You are a math wiz."

    with user():
        lm += "What is 1 + 1?"

    with assistant():
        lm += gen(max_tokens=10, name="text", temperature=0.5)
        lm += "Pick a number: "

    print(str(lm))
    assert len(lm["text"]) > 0


def test_azureai_mistral_chat_smoke(rate_limiter):
    azureai_studio_endpoint = env_or_fail("AZURE_AI_STUDIO_MISTRAL_CHAT_ENDPOINT")
    azureai_studio_deployment = env_or_fail("AZURE_AI_STUDIO_MISTRAL_CHAT_DEPLOYMENT")
    azureai_studio_key = env_or_fail("AZURE_AI_STUDIO_MISTRAL_CHAT_KEY")

    lm = models.AzureAIStudioChat(
        azureai_studio_endpoint=azureai_studio_endpoint,
        azureai_studio_deployment=azureai_studio_deployment,
        azureai_studio_key=azureai_studio_key,
    )
    assert isinstance(lm, models.AzureAIStudioChat)
    lm.engine.cache.clear()

    # No "system" role for Mistral?
    # with system():
    #    lm += "You are a math wiz."

    with user():
        lm += "What is 1 + 1?"

    with assistant():
        lm += gen(max_tokens=15, name="text", temperature=0.5)
        lm += "\nPick a number: "

    print(str(lm))
    assert len(lm["text"]) > 0


def test_azureai_llama3_chat_smoke(rate_limiter):
    azureai_studio_endpoint = env_or_fail("AZURE_AI_STUDIO_LLAMA3_CHAT_ENDPOINT")
    azureai_studio_deployment = env_or_fail("AZURE_AI_STUDIO_LLAMA3_CHAT_DEPLOYMENT")
    azureai_studio_key = env_or_fail("AZURE_AI_STUDIO_LLAMA3_CHAT_KEY")

    lm = models.AzureAIStudioChat(
        azureai_studio_endpoint=azureai_studio_endpoint,
        azureai_studio_deployment=azureai_studio_deployment,
        azureai_studio_key=azureai_studio_key,
    )
    assert isinstance(lm, models.AzureAIStudioChat)

    with system():
        lm += "You are a math wiz."

    with user():
        lm += "What is 1 + 1?"

    with assistant():
        lm += gen(max_tokens=10, name="text", temperature=0.5)
        lm += "Pick a number: "

    print(str(lm))
    assert len(lm["text"]) > 0
