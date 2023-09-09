import guidance
import pytest
from .utils_pythonic import get_llm

# --------------------
# basic functionality
# --------------------

@pytest.mark.parametrize("model_name", ["openai:text-davinci-003", "transformers:gpt2"])
def test_string_append(model_name):
    lm = get_llm(model_name)
    lm += "Initial Prompt."
    lm(" Secondary Prompt.")
    
    assert str(lm) == "Initial Prompt. Secondary Prompt."
    assert lm._state == "Initial Prompt. Secondary Prompt."

@pytest.mark.parametrize("model_name", ["openai:text-davinci-003", "transformers:gpt2"])
def test_append_chains(model_name):
    lm = get_llm(model_name)
    lm("Initial Prompt. ")("Secondary Prompt.")

    assert str(lm) == "Initial Prompt. Secondary Prompt."
    assert lm._state == "Initial Prompt. Secondary Prompt."

@pytest.mark.parametrize("model_name", ["openai:text-davinci-003"])
def test_basic_controlflow(model_name):
    lm = get_llm(model_name)
    
    lm("My favorite number is: ")
    if True:
        lm += "1"
    else:
        lm += "2"
        
    assert str(lm) == "My favorite number is: 1"

# --------------------
# gen function
# --------------------
@pytest.mark.parametrize("model_name", ["openai:text-davinci-003", "transformers:gpt2"])
def test_gen_basic(model_name):
    lm = get_llm(model_name)
    lm("ABCDEFGHIJKLMNOPQRSTUVWXY").gen('letter', max_tokens=1, temperature=0)
    
    assert lm['letter'] == 'Z'

@pytest.mark.parametrize("model_name", ["transformers:gpt2"])
def test_gen_pattern_int(model_name):
    import re
    lm = get_llm(model_name)
    lm("On a scale of 1-10 I would say it is ").gen('score', pattern="[0-9]+")
    assert re.match(r'[0-9]+', lm["score"])

# --------------------
# select function
# --------------------
@pytest.mark.parametrize("model_name", ["openai:text-davinci-003", "transformers:gpt2"])
def test_select_basic(model_name):
    lm = get_llm(model_name)
    lm += \
        """
        1+1=2
        2+2=4
        3+3="""
    lm.select('sum', options=['6', '8', '10'])

    assert lm['sum'] in set(['6', '8', '10'])
    assert int(lm['sum']) == 6

# Currently fails on transformers:gpt2, need to debug
@pytest.mark.parametrize("model_name", ["openai:text-davinci-003", "transformers:gpt2"])
def test_select_shared_prefix(model_name):
    lm = get_llm(model_name)
    hp_books = [
        "Harry Potter and the Philosopher's Stone",
        "Harry Potter and the Chamber of Secrets",
        "Harry Potter and the Prisoner of Azkaban",
        "Harry Potter and the Goblet of Fire",
    ]
    lm("My favorite book is ").select('book', options=hp_books)

    assert lm['book'] in set(hp_books)

@pytest.mark.parametrize("model_name", ["openai:text-davinci-003", "transformers:gpt2"])
def test_select_guidance(model_name):
    '''TODO: Implement a test where select depends on guidance programs.'''

    lm = get_llm(model_name)
    assert True


# --------------------
# chat mode
# --------------------

@pytest.mark.parametrize("model_name", ["openai:gpt-3.5-turbo"])
def test_chat_basic(model_name):
    lm = get_llm(model_name)
    lm.system("You are a helpful and terse assistent.")

    with lm.user() as lm:
        lm += \
        """ I want a response to the following question:
        Who is the best tennis player in the world?
        Name 3 world-class experts (past or present) who would be great at answering this?
        Don't answer the question yet."""

    with lm.assistant() as lm:
        lm.gen('expert_names', max_tokens=30)

    lm.user("Great, now please answer teh question as if these experts had collaborated in writing a joint anonymous answer.")

    with lm.assistant() as lm:
        lm.gen('answer', max_tokens=100)

    assert 'expert_names' in lm
    assert 'answer' in lm

# --------------------
# async
# --------------------

# --------------------
# function calling
# --------------------

# TODO: Make this work 
@pytest.mark.parametrize("model_name", ["openai:gpt-3.5-turbo"])
def test_function_calling(model_name):
    import json
    lm = get_llm(model_name)

    function_called = False
    def get_current_weather(location, unit="celsius"):
        """ Get the current weather in a given location.
        
        Parameters
        ----------
        location : string
            The city and state, e.g. San Francisco, CA
        unit : "celsius" or "fahrenheit"
        """
        weather_info = {
            "location": location,
            "temperature": "71",
            "unit": unit,
            "forecast": ["sunny", "windy"],
        }
        function_called = True
        return json.dumps(weather_info)
    
    @guidance
    def weather_helper(lm, location, functions):
        with lm.system() as lm:
            lm("You are a helpful assistant.")
            lm.tool_def(functions=functions)

        lm.user(f"Get the current weather in {location}.")

        for _ in range(10):
            with lm.assistant() as lm:
                lm = lm.gen('answer', max_tokens=50, function_call="auto")
            ans = lm["answer"]
            
            if not callable(ans):
                break
            
        lm.function(ans(), name=ans.__name__)

    lm.weather_helper("New York City", functions=[
    {
        "name": "get_current_weather",
        "callable": get_current_weather,
        "description": "Get the current weather in a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA",
                },
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["location"],
        }
    }])

    assert function_called


# --------------------
# errors
# --------------------

