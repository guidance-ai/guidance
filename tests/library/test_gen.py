import guidance
import pytest
from ..utils import get_llm

def test_gen():
    """Test that LM generation works."""

    llm = guidance.llms.Mock(" Sue")
    prompt = guidance("Hello my name is{{gen 'name' max_tokens=5}}", llm=llm)
    out = prompt()
    assert len(out["name"]) > 1

def test_gen_n_greater_than_one():
    llm = guidance.llms.Mock(["mock output 0", "mock output 1", "mock output 2"])
    prompt = guidance('''The best thing about the beach is{{gen 'best' n=3 temperature=0.7 max_tokens=5}}''', llm=llm)
    a = prompt()
    assert "\n".join(a["best"]) == 'mock output 0\nmock output 1\nmock output 2'

def test_gen_n_greater_than_one_hidden():
    llm = guidance.llms.Mock()

    def aggregate(best):
        return '\n'.join(['- ' + x for x in best])
    prompt = guidance('''The best thing about the beach is{{gen 'best' temperature=0.7 n=3 hidden=True}}
{{aggregate best}}''', llm=llm)
    a = prompt(aggregate=aggregate)
    assert str(a) == 'The best thing about the beach is\n- mock output 0\n- mock output 1\n- mock output 2'

def test_pattern():
    import re
    llm = get_llm("transformers:gpt2")
    out = guidance('''On a scale of 1-10 I would say it is: {{gen 'score' pattern="[0-9]+"}}''', llm=llm)()
    assert re.match(r'[0-9]+', out["score"])

def test_pattern2():
    import re

    prompt = '''
    Tweak this proverb to apply to machine learning model instructions instead.

    {{proverb}}
    - {{book}} {{chapter}}:{{verse}}

    UPDATED
    Where there is no guidance, a people falls, but in an abundance of counselors there is safety.
    - GPT {{gen 'chapter' pattern='[0-9]' max_tokens=1}}:{{gen 'verse' pattern='[0-9]+' stop='\\n'}}
    '''[1:-1]

    llm = get_llm("transformers:gpt2")
    program = guidance(prompt, llm=llm)
    executed_program = program(
        proverb="Where there is no guidance, a people falls,\nbut in an abundance of counselors there is safety.",
        book="Proverbs",
        chapter=11,
        verse=14
    )

    assert re.fullmatch(r"[0-9]", executed_program["chapter"])
    assert re.fullmatch(r"[0-9]+", executed_program["verse"])

@pytest.mark.parametrize("llm", ["transformers:gpt2", "openai:text-curie-001"])
def test_stop(llm):
    """Test that the stop argument works as expected."""
    llm = get_llm(llm)
    program = guidance("""Write "repeat this. " 10 times: repeat this. repeat this. repeat this. repeat this. repeat this. repeat this.{{gen stop="this" max_tokens=10}}""", llm=llm)
    out = program()
    assert str(out) == "Write \"repeat this. \" 10 times: repeat this. repeat this. repeat this. repeat this. repeat this. repeat this. repeat "

@pytest.mark.parametrize("llm", ["transformers:gpt2", "openai:text-curie-001"])
def test_stop_regex(llm):
    """Test that the stop_regex argument works as expected."""
    llm = get_llm(llm)
    program = guidance("""Write "repeat this. " 10 times: repeat this. repeat this. repeat this. repeat this. repeat this. repeat this.{{gen stop_regex="th.s" max_tokens=10}}""", llm=llm)
    out = program()
    assert str(out) == "Write \"repeat this. \" 10 times: repeat this. repeat this. repeat this. repeat this. repeat this. repeat this. repeat "

@pytest.mark.parametrize("llm", ["transformers:gpt2", "openai:text-curie-001"])
def test_save_stop_text(llm):
    llm = get_llm(llm)
    out = guidance("""Repeat this ten times: "s38 kdjksid sk slk", "s38 kdjksid sk slk", "s38 kdjksid sk slk", "s38 kdjksid sk slk", "{{gen 'text' stop_regex="kdj.*slk" max_tokens=10 save_stop_text=True}}""", llm=llm)()
    assert out["text_stop_text"] == "kdjksid sk slk"

@pytest.mark.parametrize("llm", ["transformers:gpt2", "openai:text-curie-001"])
def test_stop_regex_cut_short(llm):
    """Test that the stop_regex argument works as expected even when max_tokens cuts it short."""
    llm = get_llm(llm)
    out = guidance("""Repeat this ten times: "s38 kdjksid", "s38 kdjksid", "s38 kdjksid", "s38 kdjksid", "{{gen 'text' stop_regex="s38 kdjksid" max_tokens=5 save_stop_text=True}}""", llm=llm)()
    assert len(out["text"]) > 0 # make sure we got some output (it is not a stop string until it is a full match)

@pytest.mark.parametrize("llm", ["transformers:gpt2", "openai:text-curie-001"])
def test_gen_stream(llm):
    """Test that streaming the generation works."""

    llm1 = get_llm(llm, caching=False)
    prompt = guidance("Hello my name is{{gen 'name' max_tokens=10 stream=True}}", llm=llm1)
    out = prompt()
    assert len(out["name"]) > 1

    # make sure it also works with caching
    llm2 = get_llm(llm, caching=True)
    prompt = guidance("Hello my name is{{gen 'name' max_tokens=10 stream=True}}", llm=llm2)
    out = prompt()
    assert len(out["name"]) > 1