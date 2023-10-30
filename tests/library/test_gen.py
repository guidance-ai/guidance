import guidance
from guidance import gen, models
from ..utils import get_model

def test_basic():
    lm = models.LocalMock()
    lm += "Write a number: " + gen('text', max_tokens=3)
    assert len(lm["text"]) > 0

def test_stop_string():
    lm = models.LocalMock(b"<s>Count to 10: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10")
    lm += "Count to 10: 1, 2, 3, 4, 5, 6, 7, " + gen('text', stop=", 9")
    assert lm["text"] == "8"

def test_stop_char():
    lm = models.LocalMock(b"<s>Count to 10: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10")
    lm += "Count to 10: 1, 2, 3, 4, 5, 6, 7, " + gen('text', stop=",")
    assert lm["text"] == "8"

def test_stop_quote():
    lm = get_model("transformers:gpt2")
    lm += '''A title: "''' + gen('title', max_tokens=30, stop='"')
    assert not lm["title"].endswith('"')

def test_unicode():
    lm = get_model("transformers:gpt2")
    lm + '''Question: Josh decides to try flipping a house.  He buys a house for $80,000 and then puts in $50,000 in repairs.  This increased the value of the house by 150%.  How much profit did he make?
Let's think step by step, and then write the answer:
Step 1''' + gen('steps', list_append=True, stop=['\nStep', '\n\n', '\nAnswer'], temperature=0.7, max_tokens=20) + '\n'
    i = 2
    lm + f'Step {i}:' + gen('steps', list_append=True, stop=['\nStep', '\n\n', '\nAnswer'], temperature=0.7, max_tokens=20) + '\n'

def test_gsm8k():
    lm = models.LocalMock()
    lm + '''Question: Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?
Answer: ''' + gen(max_tokens=30)
    assert True

def test_pattern_kleene():
    lm = get_model("transformers:gpt2")
    lm += 'The Lord is my'
    x = lm + gen(name='tmp', max_tokens=10)
    y = lm + gen(name='tmp', pattern='.*', max_tokens=10)
    assert x['tmp'] == y['tmp']

def test_non_token_force():
    '''This forces some bytes that don't match a token (only longer tokens)'''
    lm = get_model("transformers:gpt2")
    lm += 'ae ae' + gen(pattern='\d')
    assert len(str(lm)) == 6
# def test_pattern():
#     lm = get_model("transformers:gpt2")
#     lm += 'hey there my friend what is truth 23+43=' + gen(pattern=r'dog(?P<stop>.+)', max_tokens=30)
#     assert str(lm) == "hey there my friend what is truth 23+43=dog"
