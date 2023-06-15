import openai
openai.log = 'debug'
import guidance

def test_geneach_chat_gpt_with_functions():
    """ Test a geneach loop with ChatGPT.
    """

    guidance.llm = guidance.llms.OpenAI("gpt-4-0613")
    guidance.log = 'debug'

    chat_loop = guidance('''
    {{#system~}}
    You are a helpful assistant that can tell the weather.
    {{~/system}}

    {{#user~}}
    What is the weather like in Boston?
    {{~/user}}
    {{gen 'this.response' temperature=0 max_tokens=200}}''', functions=[
        {
        "name": "get_current_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
            "type": "object",
            "properties": {
            "location": {
                "type": "string",
                "description": "The city and state, e.g. San Francisco, CA"
            },
            "unit": {
                "type": "string",
                "enum": ["celsius", "fahrenheit"]
            }
            },
            "required": ["location"]
        }
        }
    ])

    out = chat_loop()
    print(out)
    assert len(out["conversation"]) == 2

# def test_geneach_chat_gpt():
#     """ Test a geneach loop with ChatGPT.
#     """

#     guidance.llm = get_llm("openai:gpt-3.5-turbo")

#     chat_loop = guidance('''
#     {{#system~}}
#     You are a helpful assistant
#     {{~/system}}

#     {{~#geneach 'conversation' stop=False}}
#     {{#user~}}
#     This is great!
#     {{~/user}}

#     {{#assistant~}}
#     {{gen 'this.response' temperature=0 max_tokens=3}}
#     {{~/assistant}}
#     {{#if (> @index 0)}}{{break}}{{/if}}
#     {{~/geneach}}''')

#     out = chat_loop()
#     assert len(out["conversation"]) == 2

# def test_rest_nostream():
#     guidance.llm = get_llm('openai:text-davinci-003', endpoint="https://api.openai.com/v1/completions", rest_call=True)
#     a = guidance('''Hello,  my name is{{gen 'name' stream=False max_tokens=5}}''', stream=False)
#     a = a()
#     assert len(a['name']) > 0

# def test_rest_stream():
#     guidance.llm = get_llm('openai:text-davinci-003', endpoint="https://api.openai.com/v1/completions", rest_call=True)
#     a = guidance('''Hello,  my name is{{gen 'name' stream=True max_tokens=5}}''', stream=False)
#     a = a()
#     assert len(a['name']) > 0

# def test_rest_chat_nostream():
#     guidance.llm =get_llm("openai:gpt-3.5-turbo", endpoint="https://api.openai.com/v1/chat/completions", rest_call=True)
#     prompt = guidance(
# '''{{#system~}}
# You are a helpful assistant.
# {{~/system}}
# {{#user~}}
# {{conversation_question}}
# {{~/user}}
# {{#assistant~}}
# {{gen "answer" max_tokens=5 stream=False}}
# {{~/assistant}}''')
#     prompt = prompt(conversation_question='Whats is the meaning of life??')
#     assert len(prompt['answer']) > 0

# def test_rest_chat_stream():
#     guidance.llm =get_llm("openai:gpt-3.5-turbo", endpoint="https://api.openai.com/v1/chat/completions", rest_call=True)
#     prompt = guidance(
# '''{{#system~}}
# You are a helpful assistant.
# {{~/system}}
# {{#user~}}
# {{conversation_question}}
# {{~/user}}
# {{#assistant~}}
# {{gen "answer" max_tokens=5 stream=True}}
# {{~/assistant}}''')
#     prompt = prompt(conversation_question='Whats is the meaning of life??')
#     assert len(prompt['answer']) > 0