import pytest

from guidance.models import Model, OpenAI

from ..model_specific import common_chat_testing
from ..utils import env_or_skip, slowdown


@pytest.fixture(scope="function")
def openai_chat_model():
    slowdown()

    openai_api_key = env_or_skip("OPENAI_API_KEY")
    model_name = None  # env_or_skip("AZUREAI_OPENAI_CHAT_MODEL")

    print(f"{model_name=}")

    lm = OpenAI("gpt-4o-mini", api_key=openai_api_key)
    assert isinstance(lm, Model)
    print(f"{type(lm._interpreter)=}")

    return lm


def test_openai_chat_smoke(openai_chat_model: Model):
    common_chat_testing.smoke_chat(openai_chat_model)
