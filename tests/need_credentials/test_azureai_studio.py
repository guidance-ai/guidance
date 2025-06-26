import pytest

from guidance import models
from guidance.models._azureai import create_azure_aifoundry_model

from ..model_specific import common_chat_testing
from ..utils import env_or_fail, slowdown

# pytest.skip("Deployments temporarily deleted", allow_module_level=True)

# How to fill out the environment variables to
# set up the models
# Temporarily remove mistral pending endpoint investigation
# _chat_models = {"phi3": "PHI3", "llama3": "LLAMA3_CHAT"}
_chat_models = {"phi4": "PHI4"}


def _get_chat_model(model_name: str):
    env_string = _chat_models[model_name]

    azureai_studio_endpoint = env_or_fail(f"AZUREAI_STUDIO_{env_string}_ENDPOINT")
    azureai_studio_model_name = env_or_fail(f"AZUREAI_STUDIO_{env_string}_MODEL_NAME")
    azureai_studio_key = env_or_fail(f"AZUREAI_STUDIO_{env_string}_KEY")

    lm = create_azure_aifoundry_model(
        azure_endpoint=azureai_studio_endpoint,
        api_key=azureai_studio_key,
        # token_credential=DefaultAzureCredential(),
        model_name=azureai_studio_model_name,
    )
    assert isinstance(lm, models.Model)
    return lm


@pytest.mark.parametrize("chat_model_name", _chat_models.keys())
def test_azureai_chat_smoke(chat_model_name: str):
    slowdown()

    lm = _get_chat_model(chat_model_name)

    common_chat_testing.smoke_chat(lm, chat_model_name != "mistral")


@pytest.mark.parametrize("chat_model_name", _chat_models.keys())
def test_azureai_chat_longer_1(chat_model_name: str):
    slowdown()

    lm = _get_chat_model(chat_model_name)
    common_chat_testing.longer_chat_1(lm, chat_model_name != "mistral")


@pytest.mark.parametrize("chat_model_name", _chat_models.keys())
def test_azureai_chat_longer_2(chat_model_name: str):
    slowdown()

    lm = _get_chat_model(chat_model_name)
    common_chat_testing.longer_chat_2(lm, chat_model_name != "mistral")
