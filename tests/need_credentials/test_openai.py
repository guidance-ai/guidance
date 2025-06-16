import pytest
import requests

from guidance import assistant, gen, gen_audio, image, system, user
from guidance.models import Model, OpenAI

from ..model_specific import common_chat_testing
from ..utils import env_or_skip, slowdown


@pytest.fixture(scope="function")
def openai_model() -> Model:
    slowdown()

    openai_api_key = env_or_skip("OPENAI_API_KEY")
    model_name = "gpt-4o-mini"

    lm = OpenAI(model_name, api_key=openai_api_key)
    assert isinstance(lm, Model)
    print(f"{type(lm._interpreter)=}")

    return lm


@pytest.fixture(scope="function")
def openai_audio_model() -> Model:
    slowdown()

    openai_api_key = env_or_skip("OPENAI_API_KEY")
    model_name = "gpt-4o-mini-audio-preview"

    lm = OpenAI(model_name, api_key=openai_api_key)
    assert isinstance(lm, Model)
    print(f"{type(lm._interpreter)=}")

    return lm


@pytest.fixture(scope="function")
def openai_image_model() -> Model:
    slowdown()

    openai_api_key = env_or_skip("OPENAI_API_KEY")
    model_name = "gpt-4o-mini"

    lm = OpenAI(model_name, api_key=openai_api_key)
    assert isinstance(lm, Model)
    print(f"{type(lm._interpreter)=}")

    return lm


def test_openai_chat_smoke(openai_model: Model):
    common_chat_testing.smoke_chat(openai_model)


def test_openai_audio_smoke(openai_audio_model: Model):
    lm = openai_audio_model
    with system():
        lm += "Talk like a pirate."
    with user():
        lm += "What is the capital of France?"
    with assistant():
        lm += gen_audio()


def test_openai_image_smoke(openai_image_model: Model):
    lm = openai_image_model

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


def test_openai_chat_json(openai_model: Model):
    common_chat_testing.json_output_smoke(openai_model)
