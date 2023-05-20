import guidance
from ..utils import get_transformers_llm

def test_gen():
    """ Test that LM geneation works.
    """

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
    llm = get_transformers_llm("gpt2")
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

    llm = get_transformers_llm("gpt2")
    program = guidance(prompt, llm=llm)
    executed_program = program(
        proverb="Where there is no guidance, a people falls,\nbut in an abundance of counselors there is safety.",
        book="Proverbs",
        chapter=11,
        verse=14
    )

    assert re.fullmatch(r"[0-9]", executed_program["chapter"])
    assert re.fullmatch(r"[0-9]+", executed_program["verse"])