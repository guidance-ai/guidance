import os

import guidance
from ..utils import get_llamacpp_llm


TEST_MODEL = "./gpt4-x-vicuna-13B.ggml.q5_0.bin"

def test_basic():
    llm = get_llamacpp_llm(TEST_MODEL)
    with llm.session() as s:
        out = s("this is a test", max_tokens=5)
        print(out)


def test_repeat():
    llm = get_llamacpp_llm(TEST_MODEL)
    with llm.session() as s:
        out1 = s("this is a test", max_tokens=5)
        out2 = s("this is a test like another", max_tokens=5)
        print(out2)


def test_stop():
    llm = get_llamacpp_llm(TEST_MODEL)
    program = guidance(
        """Repeat this. Repeat this. Repeat this. Repeat this. Repeat this. Repeat this.{{gen stop="this" max_tokens=10}}""",
        llm=llm)
    out = program()
    print(out)
    assert str(out) == "Repeat this. Repeat this. Repeat this. Repeat this. Repeat this. Repeat this. Repeat "


def test_pattern():
    import re
    llm = get_llamacpp_llm(TEST_MODEL)
    program = guidance("""Repeat this. Repeat this. Repeat this. Repeat this. {{gen pattern="[0-9]+" max_tokens=1}}""",
                       llm=llm)
    out = program()
    print(out)
    assert re.match("^Repeat this. Repeat this. Repeat this. Repeat this. [0-9]+$", str(out))


def test_select():
    llm = get_llamacpp_llm(TEST_MODEL)
    program = guidance('''Answer "yes" or "no": "{{#select 'answer'}}yes{{or}}no{{/select}}"''', llm=llm)
    out = program()
    print(out)
    assert out["answer"] in ["yes", "no"]
