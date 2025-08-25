import re

import pytest

from guidance import gen, models, select


def test_stop_list_side_effect(selected_model: models.Model):
    """Tests a bug where a stop list has an item appended to it in place instead of being updated non-destructively. The bug only occurs whe regex is also None"""
    stop_list = ["\nStep", "\n\n", "\nAnswer"]
    stop_list_length = len(stop_list)
    lm = selected_model
    (
        lm
        + """Question: Josh decides to try flipping a house.  He buys a house for $80,000 and then puts in $50,000 in repairs.  This increased the value of the house by 150%.  How much profit did he make?
Let's think step by step, and then write the answer:
Step 1"""
        + gen(
            "steps",
            list_append=True,
            stop=["\nStep", "\n\n", "\nAnswer"],
            temperature=0.7,
            max_tokens=20,
        )
        + "\n"
    )
    assert stop_list_length == len(stop_list)
    i = 2
    (
        lm
        + f"Step {i}:"
        + gen(
            "steps",
            list_append=True,
            stop=["\nStep", "\n\n", "\nAnswer"],
            temperature=0.7,
            max_tokens=20,
        )
        + "\n"
    )
    assert stop_list_length == len(stop_list)


def test_stop_quote(selected_model):
    lm = selected_model
    lm += '''A title: "''' + gen("title", max_tokens=30, stop='"')
    assert not lm["title"].endswith('"')


def test_metrics_smoke(selected_model: models.Model):
    lm = selected_model

    lm += "Generate the next letter: a b c d "
    lm += gen("first", max_tokens=1)

    # Can't be sure of exact count due to token healing:
    # either one after the prompt, or a backtrack causing two
    usage = lm._get_usage()
    assert 1 <= usage.forward_passes <= 2

    lm += " f g"
    lm += gen("second", max_tokens=1)

    # Again, trouble with healing:
    # plus one for the new text
    # plus one or two for the healing
    new_usage = lm._get_usage()
    assert 2 <= new_usage.forward_passes <= 5

    assert new_usage.input_tokens > usage.input_tokens
    assert new_usage.forward_passes > usage.forward_passes
    assert new_usage.output_tokens > usage.output_tokens


def test_metrics_select(selected_model: models.Model):
    lm = selected_model

    lm += "I will "
    lm += select(
        [
            "ride a bicycle down the road",
            "row in a boat along the river",
            "go for a swim in the ocean",
        ]
    )

    usage = lm._get_usage()
    assert usage.input_tokens > 0
    assert usage.output_tokens > 0

    # Guidance should be able to force the generation after only a couple of tokens
    # so even though the options are long, relatively few output tokens should be
    # needed
    assert usage.ff_tokens > 0


def test_unicode(selected_model: models.Model):
    # black makes this test ugly -- easier to read with fmt: off
    # fmt: off
    lm = selected_model
    lm + '''Question: Josh decides to try flipping a house.  He buys a house for $80,000 and then puts in $50,000 in repairs.  This increased the value of the house by 150%.  How much profit did he make?
Let's think step by step, and then write the answer:
Step 1''' + gen('steps', list_append=True, stop=['\nStep', '\n\n', '\nAnswer'], temperature=0.7, max_tokens=20) + '\n'
    i = 2
    lm + f'Step {i}:' + gen('steps', list_append=True, stop=['\nStep', '\n\n', '\nAnswer'], temperature=0.7, max_tokens=20) + '\n'
    # fmt: on


def test_unicode2(selected_model: models.Model):
    lm = selected_model

    prompt = "Janet’s ducks lay 16 eggs per day"
    lm += prompt + gen(max_tokens=10)

    usage = lm._get_usage()
    assert usage.input_tokens > 1
    # Due to token healing, we can't be sure of the
    # precise output count
    assert 10 <= usage.forward_passes <= 11


def test_pattern_kleene(selected_model: models.Model):
    lm = selected_model
    lm += "The Lord is my"
    x = lm + gen(name="tmp", max_tokens=10)
    y = lm + gen(name="tmp", regex=".*", max_tokens=10)
    # Check that x and y agree up to the first newline in x
    assert y["tmp"].startswith(
        x["tmp"].split("\n")[0]
    )  # TODO: we just check startswith because exact token limits are not perfect yet...


def test_non_token_force(selected_model: models.Model):
    """This forces some bytes that don't match a token (only longer tokens)"""
    lm = selected_model
    lm += "ae ae" + gen(regex=r"\d")
    assert len(str(lm)) == 6


@pytest.mark.parametrize(
    "prompt",
    [
        "Hi there",
        "2 + 2 = ",
        "Scott is a",
        "I have never seen a more",
        "What is the",
        "?FD32",
    ],
)
@pytest.mark.parametrize(
    "pattern",
    [
        "(Scott is a person|Scott is a persimmon)",
        r"Scott is a persimmon.{0,20}\.",
        r"[0-9]\.{0,20}[0-9]+",
    ],
)
def test_various_regexes(selected_model: models.Model, prompt: str, pattern: str):
    lm = selected_model
    lm2 = lm + prompt + gen(name="test", regex=pattern, max_tokens=40)
    # note we can't just test any regex pattern like this, we need them to have finished in less than 40 tokens
    assert re.match(pattern, lm2["test"], re.DOTALL) is not None


@pytest.mark.resource_intensive
def test_long_prompt(selected_model: models.Model, selected_model_name: str):
    if selected_model_name in [
        "llamacpp_llama2_7b_cpu",
        "llamacpp_llama2_7b_gpu",
        "llamacpp_mistral_7b_cpu",
        "transformers_mistral_7b_cpu",
    ]:
        pytest.xfail("Insufficient context window in model")
    lm = selected_model
    prompt = """Question: Legoland has 5 kangaroos for each koala. If Legoland has 180 kangaroos, how many koalas and kangaroos are there altogether?
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
Step 1"""
    lm += prompt + gen(max_tokens=10)
    assert True
