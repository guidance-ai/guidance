import guidance
from ..utils import get_openai_llm

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