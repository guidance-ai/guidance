import guidance
from ..utils import get_llm
import re


def test_chat_model_pattern():
    chat_models = [
        'gpt-4o',
        'gpt-4o-2024-05-13',
        'gpt-4-turbo',
        'gpt-4-turbo-2024-04-09',
        'gpt-4-turbo-preview',
        'gpt-4-0125-preview',
        'gpt-4-vision-preview',
        'gpt-4-1106-vision-preview',
        'gpt-4',
        'gpt-4-0613',
        'gpt-4-32k',
        'gpt-4-32k-0613',
        'gpt-3.5-turbo',
        'gpt-3.5-turbo-instruct',
        'gpt-3.5-turbo-16k-0613',
    ]

    chat_model_pattern = guidance.llms.OpenAI.chat_model_pattern
    all_good = True
    failed = []
    for model in chat_models:
        if re.match(chat_model_pattern, model):
            chat_mode = True
        else:
            chat_mode = False
            failed.append(model)

        all_good = all_good and chat_mode

    assert all_good, f"Model(s) not recognized: {failed}"


def test_geneach_chat_gpt():
    """ Test a geneach loop with ChatGPT.
    """

    guidance.llm = get_llm("openai:gpt-3.5-turbo")

    chat_loop = guidance('''
{{#system~}}
You are a helpful assistant
{{~/system}}

{{~#geneach 'conversation' stop=False}}
{{#user~}}
This is great!
{{~/user}}

{{#assistant~}}
{{gen 'this.response' temperature=0 max_tokens=3}}
{{~/assistant}}
{{#if @index > 0}}{{break}}{{/if}}
{{~/geneach}}''')

    out = chat_loop()
    assert len(out["conversation"]) == 2

def test_syntax_match():
    """ Test a geneach loop with ChatGPT.
    """

    guidance.llm = get_llm("openai:gpt-3.5-turbo")

    chat_loop = guidance('''
{{~#system~}}
You are a helpful assistant
{{~/system~}}

{{~#user~}}
This is great!
{{~/user~}}

{{~#assistant~}}
Indeed
{{~/assistant~}}''')

    out = chat_loop()
    assert str(out) == '<|im_start|>system\nYou are a helpful assistant<|im_end|><|im_start|>user\nThis is great!<|im_end|><|im_start|>assistant\nIndeed<|im_end|>'

def test_rest_nostream():
    guidance.llm = get_llm('openai:text-davinci-003', endpoint="https://api.openai.com/v1/completions", rest_call=True)
    a = guidance('''Hello,  my name is{{gen 'name' stream=False max_tokens=5}}''', stream=False)
    a = a()
    assert len(a['name']) > 0

def test_rest_stream():
    guidance.llm = get_llm('openai:text-davinci-003', endpoint="https://api.openai.com/v1/completions", rest_call=True)
    a = guidance('''Hello,  my name is{{gen 'name' stream=True max_tokens=5}}''', stream=False)
    a = a()
    assert len(a['name']) > 0

def test_rest_chat_nostream():
    guidance.llm =get_llm("openai:gpt-3.5-turbo", endpoint="https://api.openai.com/v1/chat/completions", rest_call=True)
    prompt = guidance(
'''{{#system~}}
You are a helpful assistant.
{{~/system}}
{{#user~}}
{{conversation_question}}
{{~/user}}
{{#assistant~}}
{{gen "answer" max_tokens=5 stream=False}}
{{~/assistant}}''')
    prompt = prompt(conversation_question='Whats is the meaning of life??')
    assert len(prompt['answer']) > 0

def test_rest_chat_stream():
    guidance.llm =get_llm("openai:gpt-3.5-turbo", endpoint="https://api.openai.com/v1/chat/completions", rest_call=True)
    prompt = guidance(
'''{{#system~}}
You are a helpful assistant.
{{~/system}}
{{#user~}}
{{conversation_question}}
{{~/user}}
{{#assistant~}}
{{gen "answer" max_tokens=5 stream=True}}
{{~/assistant}}''')
    prompt = prompt(conversation_question='Whats is the meaning of life??')
    assert len(prompt['answer']) > 0