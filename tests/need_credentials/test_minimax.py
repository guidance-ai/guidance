import pytest

from guidance import assistant, gen, system, user
from guidance.models import MiniMax, Model

from ..model_specific import common_chat_testing
from ..utils import env_or_skip, slowdown


@pytest.fixture(scope="function")
def minimax_model() -> Model:
    slowdown()

    api_key = env_or_skip("MINIMAX_API_KEY")
    model_name = "MiniMax-M2.7"

    lm = MiniMax(model_name, api_key=api_key)
    assert isinstance(lm, Model)
    print(f"{type(lm._interpreter)=}")

    return lm


def test_minimax_chat_smoke(minimax_model: Model):
    common_chat_testing.smoke_chat(minimax_model)


def test_minimax_chat_json(minimax_model: Model):
    common_chat_testing.json_output_smoke(minimax_model)


def test_minimax_chat_multi_turn(minimax_model: Model):
    lm = minimax_model
    with system():
        lm += "You are a helpful assistant."

    with user():
        lm += "What is 1 + 1?"

    with assistant():
        lm += gen(max_tokens=10, name="answer1")

    assert len(lm["answer1"]) > 0

    with user():
        lm += "Now add 3 to that result."

    with assistant():
        lm += gen(max_tokens=10, name="answer2")

    assert len(lm["answer2"]) > 0
