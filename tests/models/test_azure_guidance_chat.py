import numpy as np
import pytest

import guidance
from guidance import assistant, gen, select, user

from ..utils import get_model


@pytest.fixture(scope="module")
def azure_guidance_chat(selected_model, selected_model_name):
    if selected_model_name in ["azure_guidance_chat"]:
        return selected_model
    else:
        pytest.skip("Requires Azure Guidance Chat model")


def test_azure_guidance_chat(azure_guidance_chat: guidance.models.Model):
    lm = azure_guidance_chat
    with user():
        lm += "What is 1 + 1?"
    with assistant():
        lm += gen(max_tokens=10, name="text")
        lm += "Pick a number: "
    assert len(lm["text"]) > 0


def test_azure_guidance_chat_select(azure_guidance_chat: guidance.models.Model):
    lm = azure_guidance_chat
    with user():
        lm += "Pick a number: "
    with assistant():
        lm += select(
            ["1", "11", "111", "1111", "11111", "111111", "1111111"], name="the number"
        )
    print(repr( str(lm) ))
    assert lm["the number"][-1] == "1"


def test_azure_guidance_chat_loop(azure_guidance_chat: guidance.models.Model):
    # tests issue #509
    model = azure_guidance_chat

    for i in range(2):
        with user():
            lm = model + f"The number is: {i}"
        with assistant():
            lm += gen(name="answer", max_tokens=2)
