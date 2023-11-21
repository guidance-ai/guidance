import guidance
from guidance import gen, models, commit_point, Tool, select, capture, string
from ..utils import get_model
import re

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

def test_unicode2():
    lm = get_model("transformers:gpt2")
    prompt = 'Janet’s ducks lay 16 eggs per day'
    lm +=  prompt + gen(max_tokens=10)
    assert True

def test_gsm8k():
    lm = models.LocalMock()
    lm + '''Question: Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?
Answer: ''' + gen(max_tokens=30)
    assert True

def test_pattern_kleene():
    lm = get_model("transformers:gpt2")
    lm += 'The Lord is my'
    x = lm + gen(name='tmp', max_tokens=10)
    y = lm + gen(name='tmp', regex='.*', max_tokens=10)
    assert x['tmp'] == y['tmp']

def test_non_token_force():
    '''This forces some bytes that don't match a token (only longer tokens)'''
    lm = get_model("transformers:gpt2")
    lm += 'ae ae' + gen(regex=r'\d')
    assert len(str(lm)) == 6

def test_gen_vs_grammar():
    lm = get_model("transformers:gpt2")
    lm += 'The Lord is my'
    x = lm + gen(name='tmp', max_tokens=10)
    y = lm + gen(name='tmp',  regex='.*', max_tokens=10)
    assert x['tmp'] == y['tmp']

def test_pattern_optional():
    lm = models.LocalMock(b"<s>12342333")
    pattern = '.?233'
    lm2 = lm + '123' + gen(name='numbers', regex=pattern, max_tokens=10)
    assert lm2['numbers'] == '4233'
    lm = models.LocalMock(b"<s>1232333")
    pattern = '.?233'
    lm2 = lm + '123' + gen(name='numbers', regex=pattern, max_tokens=10)
    assert lm2['numbers'] == '233'
    pattern = r'(Scott is bad)?(\d+)?o'
    lm = models.LocalMock(b"<s>John was a little man full of things")
    lm2 = lm + 'J' + gen(name='test', regex=pattern, max_tokens=30)
    assert lm2['test'] == 'o'

def test_pattern_stops_when_fulfilled():
    lm = models.LocalMock(b"<s>123abc")
    lm += gen(regex=r'\d+', max_tokens=10, name='test')
    assert lm['test'] == '123'

def test_pattern_star():
    # lm = models.LocalMock(b"<s>1234233234<s>") # commented out because it is not a valid test
    # patterns = ['\d+233', '\d*233', '.+233', '.*233']
    # for pattern in patterns:
    #     lm2 = lm + '123' + gen(name='numbers', regex=pattern, max_tokens=10)
    #     assert lm2['numbers'] == '4233'
    lm = models.LocalMock(b"<s>123233")
    patterns = [r'\d*233','.*233']
    for pattern in patterns:
        lm2 = lm + '123' + gen(name='numbers', regex=pattern, max_tokens=10)
        assert lm2['numbers'].startswith('233')
    pattern = '.*(\n|little)'
    lm = models.LocalMock(b"<s>John was a little")
    lm2 = lm + 'J' + gen(name='test', regex=pattern, max_tokens=30)
    assert lm2['test'].startswith('ohn was a little')
    lm = models.LocalMock(b"<s>John was a litt\n")
    lm2 = lm + 'J' + gen(name='test', regex=pattern, max_tokens=30)
    assert lm2['test'].startswith('ohn was a litt\n')

def test_stop_regex():
    lm = models.LocalMock(b"<s>123a3233")
    lm2 = lm + '123' + gen(name='test', stop_regex=r'\d233', max_tokens=10)
    assert lm2['test'] == 'a'
    lm = models.LocalMock(b"<s>123aegalera3233")
    lm2 = lm + '123' + gen(name='test', stop_regex=r'\d', max_tokens=30)
    assert lm2['test'] == 'aegalera'

def test_stop_regex_star():
    lm = models.LocalMock(b"<s>123a3233")
    pattern = r'\d+233'
    lm2 = lm + '123' + gen(name='test', stop_regex=pattern, max_tokens=10)
    assert lm2['test'] == 'a'

def test_empty_pattern():
    pattern = r'(Scott is bad)?(\d+)?'
    lm = models.LocalMock(b"<s>J<s>")
    lm2 = lm + 'J' + gen(name='test', regex=pattern, max_tokens=30)
    assert lm2['test'] == ''

def test_various_regexes():
    lm = get_model("transformers:gpt2")
    prompts = ['Hi there', '2 + 2 = ', 'Scott is a', 'I have never seen a more', 'What is the', '?FD32']
    patterns = ['(Scott is a person|Scott is a persimmon)', r'Scott is a persimmon.*\.', r'\d\.*\d+']
    for prompt in prompts:
        for pattern in patterns:
            lm2 = lm + prompt + gen(name='test', regex=pattern, max_tokens=40)
            assert re.match(pattern, lm2['test']) is not None # note we can't just test any regex pattern like this, we need them to have finished in less than 40 tokens
# def test_pattern():
#     lm = get_model("transformers:gpt2")
#     lm += 'hey there my friend what is truth 23+43=' + gen(regex=r'dog(?P<stop>.+)', max_tokens=30)
#     assert str(lm) == "hey there my friend what is truth 23+43=dog"

def test_long_prompt():
    lm = get_model("transformers:gpt2")
    prompt = '''Question: Legoland has 5 kangaroos for each koala. If Legoland has 180 kangaroos, how many koalas and kangaroos are there altogether?
Let's think step by step, and then write the answer:
Step 1: For every 5 kangaroos, there is one koala, meaning for the 180 kangaroos, there are 180/5 = 36 koalas.
Step 2: Altogether, there are 36+180 = 216 koalas and kangaroos.
Answer: 216

Question: Jennifer has ten pears, 20 oranges, and twice as many apples as pears. If she gives her sister two of each fruit, how many fruits does she have left?
Let's think step by step, and then write the answer:
Step 1: Jennifer has 2*10 = 20 apples.
Step 2: She has a total of 10+20+20 = 50 fruits.
Step 3: She gives her sister 2 pears+2 oranges+2 apples = 2+2+2 = 6 fruits.
Step 4: After giving her sister 2 of each fruit, she has 50-6 = 44 fruits left.

Answer: 44

Question: A busy restaurant is counting how many customers they had during that Friday to try to predict how many they might get on Saturday. During breakfast, they had 73 customers. During lunch, they had 127 customers. During dinner, they had 87 customers. If they predict they'll get twice the amount of customers on Saturday as they had on Friday, how many customers do they predict they will get?
Let's think step by step, and then write the answer:
Step 1: On Friday the restaurant had 73 customers for breakfast + 127 customers for lunch + 87 customers for dinner = 73+127+87 = 287 customers total on Friday.
Step 2: If they predict getting 2x the amount of customers on Saturday as on Friday, they will have 287 customers x 2 = 287*2 = 574 customers on Saturday.

Answer: 574

Question: They say the first year of a dog's life equals 15 human years. The second year of a dog's life equals 9 human years and after that, every year of a dog's life equals 5 human years. According to this logic, how many human years has my 10-year-old dog lived?
Let's think step by step, and then write the answer:
Step 1: If your dog is 10 years old then in his first year of life he lived 1*15 = 15 human years
Step 2: In his second year of life, he lived 1*9 = 9 human years
Step 3: We need to calculate his remaining years or 10-2 = 8 years of dog life into human years
Step 4: If 1 year of dog life after the 2 years equates to 5 human years, then 8 years of dog life equals 8*5 = 40 human years
Step 5: In total, your dog has lived 15 + 9 + 40 = 64 human years

Answer: 64

Question: Amanda and her family are going to re-paint all the walls inside their house. Before they get started they want to divide up the work. Since all the rooms in the house have different numbers and sizes of walls in them, they figure the fairest way to divide up the work is to count all the walls in the house and assign an equal number to each person. There are 5 people in Amanda's family, including herself. There are 9 rooms in the house. 5 of the rooms have 4 walls each. The other 4 rooms each have 5 walls each. To be fair, how many walls should each person in Amanda's family paint?
Let's think step by step, and then write the answer:
Step 1: First, Amanda needs to figure out how many walls there are in the house, 5 rooms x 4 walls each = 5*4 = 20.
Step 2: The other 4 rooms have x 5 walls each = 4 * 5 = 20 walls.
Step 3: The house has 20 walls + 20 walls = 20 + 20 = 40 walls total.
Step 4: To divide the work equally between the 5 people, 40 walls / 5 people = 40/5 = 8 walls for each person.

Answer: 8

Question: Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?
Let's think step by step, and then write the answer:
Step 1'''
    lm +=  prompt + gen(max_tokens=10)
    assert True

def test_list_append():
    '''This tests is list append works across grammar appends.'''
    lm = models.LocalMock(b"<s>bababababa")
    lm += "<s>"
    for _ in range(3):
        lm += gen("my_list", list_append=True, stop="a") + "a"
    assert isinstance(lm['my_list'], list)
    assert len(lm['my_list']) == 3

def test_list_append_in_grammar():
    '''This tests is list append works within the same grammar.'''
    lm = models.LocalMock(b"<s>bababababa")
    lm += "<s>"
    lm += gen("my_list", list_append=True, stop="a") + "a" + gen("my_list", list_append=True, stop="a") + "a" + gen("my_list", list_append=True, stop="a")
    assert isinstance(lm['my_list'], list)
    assert len(lm['my_list']) == 3

def test_one_char_suffix_and_regex():
    model = models.LocalMock(b"<s>this is\na test")
    model += gen(regex=".*", suffix="\n", max_tokens=20)
    assert str(model) == "this is\n"

def test_one_char_stop_and_regex():
    model = models.LocalMock(b"<s>this is\na test")
    model += gen(regex=".*", stop="\n", max_tokens=20)
    assert str(model) == "this is"

def test_tool_call():
    import guidance
    from guidance import one_or_more, select, zero_or_more
    from guidance import capture, Tool

    @guidance(stateless=True, dedent=False) # stateless=True indicates this function does not depend on LLM generation state
    def number(lm):
        n = one_or_more(select(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']))
        # Allow for negative or positive numbers
        return lm + select(['-' + n, n])

    @guidance(stateless=True, dedent=False)
    def operator(lm):
        return lm + select(['+' , '*', '**', '/', '-'])

    @guidance(stateless=True, dedent=False)
    def expression(lm):
        # Either
        # 1. A number (terminal)
        # 2. two expressions with an operator and optional whitespace
        # 3. An expression with parentheses around it
        return lm + select([
            number(),
            expression() + zero_or_more(' ') +  operator() + zero_or_more(' ') +  expression(),
            '(' + expression() + ')'
        ])

    @guidance(stateless=True, dedent=False)
    def calculator_call(lm):
        # capture just 'names' the expression, to be saved in the LM state
        return lm + 'Calculator(' + capture(expression(), 'tool_args') + ')'

    @guidance(dedent=False)
    def calculator(lm):
        expression = lm['tool_args']
        # You typically don't want to run eval directly for save reasons
        # Here we are guaranteed to only have mathematical expressions
        lm += f' = {eval(expression)}'
        return lm
    calculator_tool = Tool(calculator_call(), calculator)
    gpt2 = get_model("transformers:gpt2")
    lm = gpt2 + 'Here are five expressions:\nCalculator(3 * 3) = 33\nCalculator(2 + 1 * 3) = 5\n'
    lm += gen(max_tokens=30, temperature=0.5, tools=[calculator_tool], stop='\n\n')