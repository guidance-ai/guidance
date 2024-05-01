import pytest

import pytest

from guidance import assistant, gen, models, system, user

from ..utils import env_or_fail

# Everything in here needs credentials to work
# Mark is configured in pyproject.toml
pytestmark = pytest.mark.needs_credentials

# How to fill out the environment variables to
# set up the models
_chat_models = {"phi3": "PHI3", "mistral": "MISTRAL_CHAT", "llama3": "LLAMA3_CHAT"}


def _get_chat_model(model_name: str):
    env_string = _chat_models[model_name]

    azureai_studio_endpoint = env_or_fail(f"AZURE_AI_STUDIO_{env_string}_ENDPOINT")
    azureai_studio_deployment = env_or_fail(f"AZURE_AI_STUDIO_{env_string}_DEPLOYMENT")
    azureai_studio_key = env_or_fail(f"AZURE_AI_STUDIO_{env_string}_KEY")

    lm = models.AzureAIStudioChat(
        azureai_studio_endpoint=azureai_studio_endpoint,
        azureai_studio_deployment=azureai_studio_deployment,
        azureai_studio_key=azureai_studio_key,
        clear_cache=True,
    )
    assert isinstance(lm, models.AzureAIStudioChat)
    return lm


@pytest.mark.parametrize("chat_model_name", _chat_models.keys())
def test_azureai_chat_smoke(rate_limiter, chat_model_name: str):
    lm = _get_chat_model(chat_model_name)

    # This makes me unhappy
    if chat_model_name != "mistral":
        with system():
            lm += "You are a math wiz."

    with user():
        lm += "What is 1 + 1?"

    with assistant():
        lm += gen(max_tokens=10, name="text", temperature=0.5)
        lm += "Pick a number: "

    print(str(lm))
    assert len(lm["text"]) > 0
    assert str(lm).endswith("Pick a number: <|im_end|>")


@pytest.mark.parametrize("chat_model_name", _chat_models.keys())
def test_azureai_chat_longer_1(rate_limiter, chat_model_name: str):
    lm = _get_chat_model(chat_model_name)

    # This makes me unhappy
    if chat_model_name != "mistral":
        with system():
            lm += "You are a math wiz."

    with user():
        lm += "What is 1 + 1?"

    with assistant():
        lm += gen(max_tokens=10, name="text")
        lm += "Pick a number: "

    print(str(lm))
    assert len(lm["text"]) > 0
    assert str(lm).endswith("Pick a number: <|im_end|>")

    with user():
        lm += "10. Now you pick a number between 0 and 20"

    with assistant():
        lm += gen(max_tokens=2, name="number")

    print(str(lm))
    assert len(lm["number"]) > 0


@pytest.mark.parametrize("chat_model_name", _chat_models.keys())
def test_azureai_chat_longer_2(rate_limiter, chat_model_name: str):
    lm = _get_chat_model(chat_model_name)

    # This makes me unhappy
    if chat_model_name != "mistral":
        with system():
            lm += "You are a math wiz."

    with user():
        lm += "What is 1 + 1?"

    with assistant():
        lm += "2"

    with user():
        lm += "What is 2 + 3?"

    with assistant():
        lm += gen(max_tokens=10, name="text")
        lm += "Pick a number: "

    print(str(lm))
    assert len(lm["text"]) > 0
    assert str(lm).endswith("Pick a number: <|im_end|>")

    with user():
        lm += "10. Now you pick a number between 0 and 20"

    with assistant():
        lm += gen(max_tokens=2, name="number")

    print(str(lm))
    assert len(lm["number"]) > 0
