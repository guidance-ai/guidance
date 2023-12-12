import pytest
import guidance
from guidance import gen, capture, select, user, system, assistant
from ..utils import get_model
import tiktoken

def test_anthropic_chat():
    try:
        lm = guidance.models.AnthropicChat(model="claude-instant-1.2")
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
        lm = guidance.models.AnthropicChat(model="claude-instant-1.2")
    except:
        pytest.skip("Skipping Anthropic test because we can't load the model!")
    with user():
        lm += "Pick a number: "
    with assistant():
        lm += select(["1", "11", "111", "1111", "11111", "111111", "1111111"], name='the number')
    
    assert str(lm)[-1] in "123"


def test_anthropic_chat_loop():
    # tests issue #509
    try:
        model = guidance.models.AnthropicChat(model="claude-instant-1.2")
    except:
        pytest.skip("Skipping Anthropic test because we can't load the model!")

    for i in range(2):

        with system():
            lm = model + "You will just return whatever number I give you"
        
        with user():
            lm += f'The number is: {i}'
        
        with assistant():
            lm += gen(name='answer', max_tokens=2)