import guidance
from ..utils import get_transformers_llm

def test_basic():
    llm = get_transformers_llm('gpt2')
    with llm.session() as s:
        out = s("this is a test", max_tokens=5)
        print(out)

def test_repeat():
    llm = get_transformers_llm('gpt2')
    with llm.session() as s:
        out1 = s("this is a test", max_tokens=5)
        out2 = s("this is a test like another", max_tokens=5)
        print(out2)

def test_select():
    llm = get_transformers_llm('gpt2')
    program = guidance('''Answer "yes" or "no": "{{#select 'answer'}}yes{{or}}no{{/select}}"''', llm=llm)
    out = program()
    assert out["answer"] in ["yes", "no"]