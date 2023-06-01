import guidance
from ..utils import get_openai_llm

def assert_usage(model_name, prompt, expected_usage, expected_usage_usd):
    llm = guidance.llms.OpenAI(model_name, caching=True)
    llm.cache.clear()
    assert llm.get_usage_cost_usd() == 0
    guidance.Program(prompt, llm=llm)()
    assert dict(llm.usage) == expected_usage
    assert llm.get_usage_cost_usd() == expected_usage_usd
    assert dict(llm.usage_cached) == {}
    guidance.Program(prompt, llm=llm)()
    assert dict(llm.usage) == expected_usage
    assert dict(llm.usage_cached) == expected_usage
    assert llm.get_usage_cost_usd(llm.usage_cached) == llm.get_usage_cost_usd() == expected_usage_usd

def test_geneach_chat_gpt():
    """ Test a geneach loop with ChatGPT.
    """

    guidance.llm = get_openai_llm("gpt-3.5-turbo")

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
    {{#if (> @index 0)}}{{break}}{{/if}}
    {{~/geneach}}''')

    out = chat_loop()
    assert len(out["conversation"]) == 2

def test_usage():
    """ Test the usage tracking of OpenAI
    """

    assert_usage(
        "text-davinci-003",
        '''Hello, {{gen 'response' temperature=0 max_tokens=3}}''',
        {'completion_tokens': 3, 'prompt_tokens': 3, 'total_tokens': 6},
        0.00011999999999999999
    )
    assert_usage(
        "gpt-3.5-turbo",
        '''
        {{#system~}}
        You are a helpful assistant
        {{~/system}}
        {{#user~}}
        Hello,
        {{~/user}}
        {{#assistant~}}
        {{gen 'response' temperature=0 max_tokens=3}}
        {{~/assistant}}
        ''',
        {'completion_tokens': 3, 'prompt_tokens': 25, 'total_tokens': 28},
        5.6000000000000006e-05
    )
