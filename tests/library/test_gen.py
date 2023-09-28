import guidance
from guidance import gen
from ..utils import get_model

def test_basic():
    lm = get_model("transformers:gpt2")
    lm += "Write a number: " + gen('text', max_tokens=3)
    assert len(lm["text"]) > 0

def test_stop_string():
    lm = get_model("transformers:gpt2")
    lm += "Count to 10: 1, 2, 3, 4, 5, 6, 7, " + gen('text', stop=", 9")
    assert lm["text"] == "8"

def test_unicode():
    lm = get_model("transformers:gpt2")
    lm + '''Question: Josh decides to try flipping a house.  He buys a house for $80,000 and then puts in $50,000 in repairs.  This increased the value of the house by 150%.  How much profit did he make?
    Let's think step by step, and then write the answer:
    Step 1''' + gen('steps', list_append=True, stop=['\nStep', '\n\n', '\nAnswer'], temperature=0.7, max_tokens=20) + '\n'
    i = 2
    lm + f'Step {i}:' + gen('steps', list_append=True, stop=['\nStep', '\n\n', '\nAnswer'], temperature=0.7, max_tokens=20) + '\n'

def test_gsm8k():
    lm = get_model("transformers:gpt2")
    lm + '''Question: Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?
Answer: ''' + gen(max_tokens=30)

def test_pattern():
    lm = get_model("transformers:gpt2")
    lm += 'hey there my friend what is truth 23+43=' + gen(pattern=r'dog(?P<stop>.+)', max_tokens=30)
    assert str(lm) == "hey there my friend what is truth 23+43=dog"
