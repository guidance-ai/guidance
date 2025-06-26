import pytest

import guidance
from guidance import assistant, gen, select, system, user


def test_togetherai_basic():
    try:
        lm = guidance.models.TogetherAI("mistralai/Mistral-7B-v0.1")
    except:
        pytest.skip("Skipping TogetherAI test because we can't load the model!")
    lm += "Count to 20: 1,2,3,4,"
    stop = "\n"
    lm += f"""{gen(max_tokens=1, stop=stop, name="text")}"""
    assert str(lm)[-1] == "5"


def test_togetherai_select():
    try:
        lm = guidance.models.TogetherAI("mistralai/Mistral-7B-v0.1")
    except:
        pytest.skip("Skipping TogetherAI test because we can't load the model!")
    nums = ["1", "11", "111", "1111", "11111", "111111", "1111111"]
    lm += "Pick a number: "
    lm += select(nums, name="number")
    assert str(lm["number"]) in nums


def test_togetherai_chat():
    try:
        lm = guidance.models.TogetherAIChat("teknium/OpenHermes-2-Mistral-7B")
    except:
        pytest.skip("Skipping TogetherAI test because we can't load the model!")
    with system():
        lm += "You are a math wiz."

    with user():
        lm += "What is 1 + 1?"

    with assistant():
        lm += gen(max_tokens=10, name="text")
        lm += "Pick a number: "

    assert len(lm["text"]) > 0


def test_togetherai_chat_without_roles():
    try:
        lm = guidance.models.TogetherAIChat("teknium/OpenHermes-2-Mistral-7B")
    except:
        pytest.skip("Skipping TogetherAI test because we can't load the model!")
    with pytest.raises(ValueError) as error_info:
        lm += "You are a math wiz. What is 1+1?" + gen(max_tokens=10, name="text")


def test_togetherai_chat_loop():
    try:
        model = guidance.models.TogetherAIChat("teknium/OpenHermes-2-Mistral-7B", echo=False)
    except:
        pytest.skip("Skipping TogetherAI test because we can't load the model!")

    with system():
        lm = model + "You will just return whatever number I give you"

    for i in range(2):
        with user():
            lm += f"The number is: {i}"

        with assistant():
            lm += gen(name="answer", max_tokens=10)
    assert len(lm["answer"]) > 0
