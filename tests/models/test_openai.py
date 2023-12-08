import pytest
import guidance
from guidance import gen, capture, select, user, system, assistant
from ..utils import get_model
import tiktoken

def test_openai_class_detection():
    # TODO: expand this with other variants of openAI models
    test_models = {
        "gpt-3.5-turbo": guidance.models.OpenAIChat,
        "gpt-4": guidance.models.OpenAIChat,
        "gpt-4-vision-preview": guidance.models.OpenAIChat,
        "ft:gpt-3.5-turbo": guidance.models.OpenAIChat,
        "ft:gpt-4": guidance.models.OpenAIChat,
        "ft:gpt-4-vision-preview": guidance.models.OpenAIChat,
        "ft:gpt-3.5-turbo:my-org:custom_suffix:id": guidance.models.OpenAIChat,
        "gpt-3.5-turbo-instruct": guidance.models.OpenAIInstruct,
        "ft:gpt-3.5-turbo-instruct": guidance.models.OpenAIInstruct,
        "text-curie-001": guidance.models.OpenAICompletion,
        "ft:text-curie-001": guidance.models.OpenAICompletion,
        "text-davinci-003": guidance.models.OpenAICompletion,
    }

    for model_name, model_class in test_models.items():
        # setting random tokenizer and fake API key to allow this test to run without tiktoken detection errors
        initialized_model = guidance.models.OpenAI(model_name, tokenizer=tiktoken.encoding_for_model("gpt-3.5-turbo"), api_key="blah")
        assert isinstance(initialized_model, model_class)
    

def test_openai_basic():
    try:
        lm = guidance.models.OpenAI("text-curie-001")
    except:
        pytest.skip("Skipping OpenAI test because we can't load the model!")
    # lm = guidance.models.Transformers("gpt2")
    lm += "Count to 20: 1,2,3,4,"
    nl = "\n"
    lm += f"""\
5,6,7"""
    lm += f"""{gen(max_tokens=1, suffix=nl)}aaaaaa"""
    assert str(lm)[-5:] == "aaaaa"

def test_openai_select():
    try:
        lm = guidance.models.OpenAI("text-curie-001")
    except:
        pytest.skip("Skipping OpenAI test because we can't load the model!")
    lm += "Pick a number: "
    lm += select(["1", "11", "111", "1111", "11111", "111111", "1111111"], name='the number')
    assert str(lm)[-1] in "123"

def test_openai_chat():
    try:
        lm = guidance.models.OpenAI("gpt-3.5-turbo")
    except:
        pytest.skip("Skipping OpenAI test because we can't load the model!")
    with system():
        lm += "You are a math wiz."

    with user():
        lm += "What is 1 + 1?"

    with assistant():
        lm += gen(max_tokens=10, name="text")
        lm += "Pick a number: "

    assert len(lm["text"]) > 0

def test_openai_chat_without_roles():
    # fake model tokenizer and API key to allow this test to run without tiktoken detection errors
    lm = guidance.models.OpenAI("gpt-3.5-turbo", tokenizer=tiktoken.encoding_for_model("gpt-3.5-turbo"), api_key="blah")
    with pytest.raises(ValueError) as error_info:
        lm += "You are a math wiz. What is 1+1?" + gen(max_tokens=10, name="text")

def test_openai_chat_loop():
    # tests issue #509
    try:
        model = guidance.models.OpenAI("gpt-3.5-turbo", echo=False)
    except:
        pytest.skip("Skipping OpenAI test because we can't load the model!")

    for i in range(2):

        with system():
            lm = model + "You will just return whatever number I give you"
        
        with user():
            lm += f'The number is: {i}'
        
        with assistant():
            lm += gen(name='answer', max_tokens=2)