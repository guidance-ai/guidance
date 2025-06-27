import pytest

import guidance
from guidance import assistant, gen, select, system, user



def test_anthropic_chat():
    try:
        lm = guidance.models.Anthropic(model="claude-3-haiku-20240307")
    except:
        pytest.skip("Skipping Anthropic test because we can't load the model!")
    with system():
        lm += "You are a math wiz."

    with user():
        lm += "What is 1 + 1?"

    with assistant():
        lm += gen(max_tokens=10, name="text")
        lm += "Pick a number: "

    assert len(lm["text"]) > 0


def test_anthropic_select():
    try:
        lm = guidance.models.Anthropic(model="claude-instant-1.2")
    except:
        pytest.skip("Skipping Anthropic test because we can't load the model!")

    # We can't meaningfully test or enforce select on this model
    with pytest.raises(guidance.models._model.ConstraintException):
        with user():
            lm += "Write the next number in the list: 1,2,3,4,5,6,"
        with assistant():
            lm += select(["harsha", "scott", "marco"], name="the number")


def test_anthropic_chat_loop():
    # tests issue #509
    try:
        model = guidance.models.Anthropic(model="claude-3-haiku-20240307")
    except:
        pytest.skip("Skipping Anthropic test because we can't load the model!")

    for i in range(2):
        with system():
            lm = model + "You will just return whatever number I give you"

        with user():
            lm += f"The number is: {i}"

        with assistant():
            lm += gen(name="answer", max_tokens=2)


# def test_direct_anthropic_api():
#     import anthropic

#     client = anthropic.Anthropic()

#     with client.messages.stream(
#         max_tokens=10,
#         system="You are a counting robot. Do nothing but continue counting numbers in the same format the user presented.",
#         messages=[{"role": "user", "content": "1,2,3,4,5,"}],
#         model="claude-3-haiku-20240307",
#     ) as stream:
#         text_list = []
#         for text in stream.text_stream:
#             print(text, end="", flush=True)
#             text_list.append(text)

#     assert len(text_list) > 0
