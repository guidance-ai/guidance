import pytest
import requests
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

from guidance import assistant, gen, gen_audio, image, models, system, user
from guidance.models import create_azure_openai_model

from ..model_specific import common_chat_testing
from ..utils import env_or_skip, slowdown


@pytest.fixture(scope="function")
def azureai_chat_model():
    slowdown()

    azureai_endpoint = env_or_skip("AZUREAI_OPENAI_CHAT_ENDPOINT")
    azureai_deployment = env_or_skip("AZUREAI_OPENAI_CHAT_DEPLOYMENT_NAME")
    azureai_api_version = env_or_skip("AZUREAI_OPENAI_CHAT_API_VERSION")
    model_name = None  # env_or_skip("AZUREAI_OPENAI_CHAT_MODEL")

    print(f"{azureai_endpoint=}")
    print(f"{azureai_deployment=}")
    print(f"{model_name=}")

    token_provider = get_bearer_token_provider(DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default")

    lm = create_azure_openai_model(
        model_name=model_name,
        azure_deployment=azureai_deployment,
        azure_endpoint=azureai_endpoint,
        azure_ad_token_provider=token_provider,
        api_version=azureai_api_version,
    )
    assert isinstance(lm, models.Model)
    print(f"{type(lm._interpreter)=}")

    return lm


@pytest.fixture(scope="function")
def azureai_audio_model():
    slowdown()

    azureai_endpoint = env_or_skip("AZUREAI_OPENAI_AUDIO_ENDPOINT")
    azureai_deployment = env_or_skip("AZUREAI_OPENAI_AUDIO_DEPLOYMENT_NAME")
    azureai_api_version = env_or_skip("AZUREAI_OPENAI_AUDIO_API_VERSION")
    model_name = env_or_skip("AZUREAI_OPENAI_AUDIO_MODEL")

    print(f"{azureai_endpoint=}")
    print(f"{azureai_deployment=}")
    print(f"{model_name=}")

    token_provider = get_bearer_token_provider(DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default")

    lm = create_azure_openai_model(
        model_name=model_name,
        azure_deployment=azureai_deployment,
        azure_endpoint=azureai_endpoint,
        azure_ad_token_provider=token_provider,
        api_version=azureai_api_version,
        has_audio_support=True,
    )
    assert isinstance(lm, models.Model)
    print(f"{type(lm._interpreter)=}")

    return lm


@pytest.fixture(scope="function")
def azureai_image_model():
    slowdown()

    azureai_endpoint = env_or_skip("AZUREAI_OPENAI_IMAGE_ENDPOINT")
    azureai_deployment = env_or_skip("AZUREAI_OPENAI_IMAGE_DEPLOYMENT_NAME")
    azureai_api_version = env_or_skip("AZUREAI_OPENAI_IMAGE_API_VERSION")
    model_name = env_or_skip("AZUREAI_OPENAI_IMAGE_MODEL")

    print(f"{azureai_endpoint=}")
    print(f"{azureai_deployment=}")
    print(f"{model_name=}")

    token_provider = get_bearer_token_provider(DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default")

    lm = create_azure_openai_model(
        model_name=model_name,
        azure_deployment=azureai_deployment,
        azure_endpoint=azureai_endpoint,
        azure_ad_token_provider=token_provider,
        api_version=azureai_api_version,
        has_image_support=True,
    )
    assert isinstance(lm, models.Model)
    print(f"{type(lm._interpreter)=}")

    return lm


def test_azureai_openai_chat_smoke(azureai_chat_model):
    common_chat_testing.smoke_chat(azureai_chat_model)


def test_azureai_openai_chat_json(azureai_chat_model: models.Model):
    common_chat_testing.json_output_smoke(azureai_chat_model)


def test_azureai_openai_audio_smoke(azureai_audio_model: models.Model):
    lm = azureai_audio_model
    with system():
        lm += "Talk like a pirate."
    with user():
        lm += "What is the capital of France?"
    with assistant():
        lm += gen_audio()


def test_azureai_openai_image_smoke(azureai_image_model: models.Model):
    lm = azureai_image_model

    image_bytes = requests.get("https://picsum.photos/300/200").content

    with system():
        lm += "You are a helpful AI assistant."
    with user():
        lm += "Can you describe this image for me?"
        lm += image(image_bytes)
    with assistant():
        lm += gen(name="img_describe")
    assert "img_describe" in lm
    assert len(lm["img_describe"]) > 0


def test_azureai_openai_chat_longer_1(azureai_chat_model):
    common_chat_testing.longer_chat_1(azureai_chat_model)


def test_azureai_openai_chat_longer_2(azureai_chat_model):
    common_chat_testing.longer_chat_2(azureai_chat_model)


def test_azureai_openai_chat_loop(azureai_chat_model):
    for i in range(2):
        print(f"Iteration: {i}")
        with system():
            generation = azureai_chat_model + "You will just return whatever number I give you"

        with user():
            generation += f"The number is: {i}"

        with assistant():
            generation += gen(name="answer", max_tokens=2)

        print(str(azureai_chat_model))
        print("\n\n")
