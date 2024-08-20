import re
import numpy as np
import pytest
from jsonschema import validate
import json

import guidance
from guidance import gen, select, assistant, user, optional, substring, one_or_more, token_limit
from guidance.library import json as gen_json

from ..utils import get_model

@pytest.fixture(scope="module")
def azure_guidance_model(selected_model, selected_model_name):
    if selected_model_name in ["azure_guidance"]:
        return selected_model
    else:
        pytest.skip("Requires Azure Guidance model")


def test_azure_guidance_fill_in_json(azure_guidance_model: guidance.models.Model):
    lm = azure_guidance_model
    @guidance(stateless=True, dedent=False)
    def character_maker(lm, id, description, valid_weapons):
        lm += f"""\
        The following is a short character profile for an RPG game in JSON format.
        ```json
        {{
            "id": "{id}",
            "description": "{description}",
            "name": "{gen('name', stop='"')}",
            "age": {gen('age', regex='[0-9]+', stop=',')},
            "armor": "{select(options=['leather', 'chainmail', 'plate'], name='armor')}",
            "weapon": "{select(options=valid_weapons, name='weapon')}",
            "class": "{gen('class', stop='"')}",
            "mantra": "{gen('mantra', stop='"')}",
            "strength": {gen('strength', regex='[0-9]+', stop=',')},
            "items": ["{gen('item', list_append=True, stop='"')}", "{gen('item', list_append=True, stop='"')}", "{gen('item', list_append=True, stop='"')}"]
        }}```"""
        return lm
    lm += character_maker(1, 'A nimble fighter', ['axe', 'sword', 'bow'])
    result = str(lm)
    json_text = result[result.find("```json") + 8:-3]
    json.loads(json_text)  # check for valid JSON


def test_azure_guidance_basic_1(azure_guidance_model: guidance.models.Model):
    lm = azure_guidance_model
    lm += "Write a number: " + gen("text", max_tokens=3)
    assert len(lm["text"]) >= 3

def test_azure_guidance_56_eos(azure_guidance_model: guidance.models.Model):
    lm = azure_guidance_model
    # make sure we recognize EOS token correctly
    lm += "Q: 7 * 8\nA: " + gen("text", regex="[0-9]+", max_tokens=20)
    assert lm["text"] == "56"

def test_azure_guidance_56_newline(azure_guidance_model: guidance.models.Model):
    lm = azure_guidance_model
    # make sure we recognize EOS token correctly
    lm += "Q: 7 * 8\nA: " + gen("text", regex="[0-9]+", max_tokens=20) + "\n"
    assert lm["text"] == "56"

def test_azure_guidance_1003_eos(azure_guidance_model: guidance.models.Model):
    lm = azure_guidance_model
    lm += "Q: 1000 + 3\nA: " + gen("text", regex="[0-9]+", max_tokens=20)
    assert lm["text"] == "1003"

def test_azure_guidance_dolphins(azure_guidance_model: guidance.models.Model):
    lm = azure_guidance_model
    # Yes|No has an implicit forced EoS at the end, which should not be actually generated
    lm += "Q: Are dolphins fish?\nA: " + gen("dolphins", regex="Yes|No", max_tokens=10) + \
        "\nQ: Are salmons fish?\nA: " + gen("sharks", regex="Yes|No", max_tokens=10)
    assert lm["dolphins"] == "No"
    assert lm["sharks"] == "Yes"

def test_azure_guidance_1003_max_tokens(azure_guidance_model: guidance.models.Model):
    lm = azure_guidance_model
    lm += "Q: 1000 + 3\nA: " + gen("text", regex="[0-9]+", max_tokens=2)
    assert lm["text"] == "10"


def test_azure_guidance_max_tokens_1(azure_guidance_model: guidance.models.Model):
    lm = azure_guidance_model
    lm += "one, two, three, " + gen(name="a", max_tokens=1) + gen(name="b", max_tokens=1)
    assert lm["a"] == "four" and lm["b"] == ","

def test_azure_guidance_max_tokens_2(azure_guidance_model: guidance.models.Model):
    lm = azure_guidance_model
    lm += "one, two, three, " + gen(name="a", max_tokens=2) + gen(name="b", max_tokens=2)
    assert lm["a"] == "four," and lm["b"] == " five,"


def test_azure_guidance_stop_char(azure_guidance_model: guidance.models.Model):
    lm = azure_guidance_model
    lm += "Count to 10: 1, 2, 3, 4, 5, 6, 7, " + gen("text", stop=",")
    assert lm["text"] == "8"


def test_azure_guidance_stop_string(azure_guidance_model: guidance.models.Model):
    lm = azure_guidance_model
    lm += "Count to 10: 1, 2, 3, 4, 5, 6, 7, " + gen("text", stop=", 9")
    print(str(lm))
    assert lm["text"] == "8"



def test_azure_guidance_gen_base(azure_guidance_model: guidance.models.Model):
    lm = azure_guidance_model
    lm = lm + "this is a test" + gen("test", max_tokens=10)
    assert len(str(lm)) > len("this is a test")


def test_azure_guidance_gen_log_probs(azure_guidance_model: guidance.models.Model):
    lm = azure_guidance_model
    lm = lm + "this is a test" + gen("test", max_tokens=1)
    assert 1 >= np.exp(lm.log_prob("test")) >= 0


def test_azure_guidance_recursion_error(azure_guidance_model: guidance.models.Model):
    lm = azure_guidance_model

    # define a guidance program that adapts a proverb
    lm = (
        lm
        + f"""Tweak this proverb to apply to model instructions instead.
    {gen('verse', max_tokens=2)}
    """
    )
    assert len(str(lm)) > len(
        "Tweak this proverb to apply to model instructions instead.\n\n"
    )


def test_azure_guidance_select2(azure_guidance_model: guidance.models.Model):
    lm = azure_guidance_model
    lm += f'this is a test1 {select(["item1", "item2"])} and test2 {select(["item3", "item4"])}'
    assert str(lm) in [
        "this is a test1 item1 and test2 item3",
        "this is a test1 item1 and test2 item4",
        "this is a test1 item2 and test2 item3",
        "this is a test1 item2 and test2 item4",
    ]


def test_azure_guidance_repeat_calls(azure_guidance_model: guidance.models.Model):
    lm_orig = azure_guidance_model
    a = []
    lm = lm_orig + "How much is 2 + 2? " + gen(name="test", max_tokens=10)
    a.append(lm["test"])
    lm = lm_orig + "How much is 2 + 2? " + gen(name="test", max_tokens=10, regex=r"[0-9]+")
    a.append(lm["test"])
    lm = lm_orig + "How much is 2 + 2? " + gen(name="test", max_tokens=10)
    a.append(lm["test"])
    assert a[-1] == a[0]


def test_azure_guidance_suffix(azure_guidance_model: guidance.models.Model):
    lm_orig = azure_guidance_model
    lm = (
        lm_orig
        + "1. Here is a sentence "
        + gen(name="bla", list_append=True, suffix="\n")
    )
    # list_append
    assert isinstance(lm["bla"], list)
    assert len(lm["bla"]) == 1
    # the capture should not have a newline
    assert lm["bla"][0][-1] != "\n"
    # the whole lm object *should* have a newline
    assert (str(lm))[-1] == "\n"
    assert (str(lm))[-2] != "\n"


# def test_azure_guidance_subtoken_forced(azure_guidance_model: guidance.models.Model):
#     lm_orig = azure_guidance_model
#     lm = lm_orig + "How much is 2 + 2? " + gen(name="test", max_tokens=10, regex=r"\(")
#     assert str(lm) == "How much is 2 + 2? ("


def test_azure_guidance_with_temp1(azure_guidance_model: guidance.models.Model):
    lm = azure_guidance_model
    lm += "Here is a cute 5-line poem about cats and dogs:\n"
    for i in range(5):
        lm += f"LINE {i+1}: " + gen(temperature=0.8, suffix="\n")
    # we just want to make sure we don't crash the numpy sampler


def test_azure_guidance_with_temp2(azure_guidance_model: guidance.models.Model):
    lm = azure_guidance_model
    lm1 = lm + "2 + 2 =" + gen("answer", max_tokens=3)
    lm2 = lm + "2 + 2 =" + gen("answer", temperature=0.0000001, max_tokens=3)
    assert lm1["answer"] == lm2["answer"]


def test_azure_guidance_max_tokens_3(azure_guidance_model: guidance.models.Model):
    lm = azure_guidance_model
    lm += "Who won the last Kentucky derby and by how much?"
    lm += "\n\n<<The last Kentucky Derby was held"
    lm += gen(max_tokens=2)
    assert (
        str(lm)[-1] != "<"
    )  # the output should not end with "<" because that is coming from the stop sequence...


def test_azure_guidance_stop_token_0(azure_guidance_model: guidance.models.Model):
    lm = azure_guidance_model
    lm += f'<color>red</color>\n<color>{gen(stop="</color>")} and test2'
    r = str(lm)
    print(r)
    print(r[20:])
    assert "</color>" not in r[20:]
    assert " and test2" in r[20:]

def test_azure_guidance_basic_2(azure_guidance_model: guidance.models.Model):
    model = azure_guidance_model
    lm = model + "Count to 20: 1,2,3,4,"
    nl = "\n"
    lm += "5,6,7" + f"""{gen(max_tokens=1, suffix=nl)}aaaaaa"""
    assert str(lm)[-6:] == "aaaaaa"

def test_azure_guidance_fstring_simple(azure_guidance_model: guidance.models.Model):
    lm = azure_guidance_model
    lm += f'this is a test {select(["item1", "item2"])}'
    assert str(lm) in ["this is a test item1", "this is a test item2"]


def test_azure_guidance_fstring_custom_statefull(azure_guidance_model: guidance.models.Model):
    lm = azure_guidance_model

    @guidance
    def my_function(lm):
        return lm + f'another {select(["item1", "item2"])}'

    lm += f"this is a test {my_function()}"
    assert str(lm) in ["this is a test another item1", "this is a test another item2"]


def test_azure_guidance_fstring_custom_stateless(azure_guidance_model: guidance.models.Model):
    lm = azure_guidance_model

    @guidance(stateless=True)
    def my_function(lm):
        return lm + f'another {select(["item1", "item2"])}'

    lm += f"this is a test {my_function()}"
    assert str(lm) in ["this is a test another item1", "this is a test another item2"]


def test_azure_guidance_token_count(azure_guidance_model: guidance.models.Model):
    lm = azure_guidance_model
    lm2 = lm + " 1 1 1 1 1" + gen(max_tokens=9) + gen(max_tokens=9)
    assert (
        18 <= lm2.token_count <= 20
    )  # note we allow ourselves to be off by one because it is hard to know when we are continuing vs starting a new token in the parser


def test_azure_guidance_call_embeddings(azure_guidance_model: guidance.models.Model):
    model = azure_guidance_model

    @guidance(dedent=False)
    def bla(lm, bla):
        lm += bla + "ae" + gen(max_tokens=10)
        return lm

    @guidance(dedent=False)
    def ble(lm):
        lm += f"""
    ae galera! {bla('33')}
    let's do more stuff!!""" + gen(
            max_tokens=10
        )
        return lm

    assert "{{G|" not in str(model + ble())


def test_azure_guidance_stream_0(azure_guidance_model: guidance.models.Model):
    lm = azure_guidance_model
    lm = lm + select(["item1", "item2"])
    assert str(lm) in ["item1", "item2"]


def test_azure_guidance_stream_propagate_errors(azure_guidance_model: guidance.models.Model):
    lm = azure_guidance_model

    @guidance
    def my_function(lm):
        raise Exception()

    with pytest.raises(Exception):
        lm += my_function()
        list(lm)


def test_azure_guidance_stream_add_multiple(azure_guidance_model: guidance.models.Model):
    """Test to make sure multiple additions to a ModelStream are all respected"""
    lm = azure_guidance_model
    lm = lm + select(["item1", "item2"])
    lm += ""
    assert str(lm) in ["item1", "item2"]


def test_azure_guidance_1_plus_1(azure_guidance_model: guidance.models.Model):
    lm = azure_guidance_model
    with user():
        lm += "What is 1 + 1?"
    with assistant():
        lm += gen(max_tokens=10, name="text")
        lm += "Pick a number: "
    assert len(lm["text"]) > 0


def test_azure_guidance_select1(azure_guidance_model: guidance.models.Model):
    lm = azure_guidance_model
    with user():
        lm += "Pick a number: "
    with assistant():
        lm += select(
            ["1", "11", "111", "1111", "11111", "111111", "1111111"], name="the number"
        )
    print(repr( str(lm) ))
    assert lm["the number"][-1] == "1"


def test_azure_guidance_loop(azure_guidance_model: guidance.models.Model):
    # tests issue #509
    model = azure_guidance_model

    for i in range(2):
        with user():
            lm = model + f"You will just return whatever number I give you. The number is: {i}"
        with assistant():
            lm += gen(name="answer", max_tokens=2)



def test_azure_guidance_chat(azure_guidance_model: guidance.models.Model):
    lm = azure_guidance_model
    max_tokens=30

    with user():
        lm += "The economy is crashing!"

    with assistant():
        lm += gen("test1", max_tokens=max_tokens)

    with user():
        lm += "What is the best again?"

    with assistant():
        lm += gen("test2", max_tokens=max_tokens)

    assert len(lm["test1"]) > 0
    assert len(lm["test2"]) > 0

    # second time to make sure cache reuse is okay
    lm = azure_guidance_model

    with user():
        lm += "The economy is crashing!"

    with assistant():
        lm += gen("test1", max_tokens=max_tokens)

    with user():
        lm += "What is the best again?"

    with assistant():
        lm += gen("test2", max_tokens=max_tokens)

    assert len(lm["test1"]) > 0
    assert len(lm["test2"]) > 0

def test_azure_guidance_phi3_newline_chat(azure_guidance_model: guidance.models.Model):
    lm = azure_guidance_model
    lm += "You are a counting bot. Just keep counting numbers."
    with user():
        lm += "1\n2\n3\n4\n"
    with assistant():
        lm += "\n" + gen(name="five", max_tokens=1)
        lm += "\n" + gen(name="six", max_tokens=1)

def test_azure_guidance_phi3_unstable_tokenization(azure_guidance_model: guidance.models.Model):
    lm = azure_guidance_model
    lm += "You are a counting bot. Just keep counting numbers."
    with user():
        lm += "1,2,3,4,"
    with assistant():
        lm += "\n" # comment and uncomment this line to get the error
        lm += gen(name="five", max_tokens=1)
        lm += "," + gen(name="six", max_tokens=1)


def test_azure_guidance_simple_recursion(azure_guidance_model: guidance.models.Model):
    @guidance(stateless=True, dedent=False)
    def grammar(lm, depth):
        if depth != 0:
            depth -= 1
            lm += "x" + optional(grammar(depth))
        return lm
    lm = azure_guidance_model
    lm += grammar(5)


def test_azure_guidance_mutual_recursion(azure_guidance_model: guidance.models.Model):
    @guidance(stateless=True, dedent=False)
    def grammar1(lm, depth):
        if depth != 0:
            depth -= 1
            lm += "x" + grammar2(depth)
        return lm

    @guidance(stateless=True, dedent=False)
    def grammar2(lm, depth):
        if depth != 0:
            depth -= 1
            lm += "y" + optional(grammar1(depth))
        return lm

    lm = azure_guidance_model
    lm += grammar1(5)
    lm += grammar2(5)

def test_azure_guidance_multiple_mutual_recursion(azure_guidance_model: guidance.models.Model):
    @guidance(stateless=True, dedent=False)
    def grammar1(lm, depth):
        if depth != 0:
            depth -= 1
            lm += "x" + grammar2(depth)
        return lm

    @guidance(stateless=True, dedent=False)
    def grammar2(lm, depth):
        if depth != 0:
            depth -= 1
            lm += "y" + grammar3(depth)
        return lm

    @guidance(stateless=True, dedent=False)
    def grammar3(lm, depth):
        if depth != 0:
            depth -= 1
            lm += "z" + optional(grammar1(depth))
        return lm

    lm = azure_guidance_model
    lm += grammar1(5)
    lm += grammar2(5)
    lm += grammar3(5)

def test_azure_guidance_branching_mutual_recursion(azure_guidance_model: guidance.models.Model):
    @guidance(stateless=True, dedent=False)
    def grammar1(lm, depth):
        if depth != 0:
            depth -= 1
            lm += "x" + grammar2(depth)
        return lm

    @guidance(stateless=True, dedent=False)
    def grammar2(lm, depth):
        if depth != 0:
            depth -= 1
            lm += "y" + select([grammar1(depth), grammar3(depth)])
        return lm

    @guidance(stateless=True, dedent=False)
    def grammar3(lm, depth):
        if depth != 0:
            depth -= 1
            lm += "z" + optional(grammar1(depth))
        return lm

    lm = azure_guidance_model
    lm += grammar1(5)
    lm += grammar2(5)
    lm += grammar3(5)


# def test_remote_gen_json(azure_guidance_model: guidance.models.Model):
#     schema = """
# {
#     "$defs": {
#         "A": {
#             "properties": {
#                 "my_str": {
#                     "default": "me",
#                     "title": "My Str",
#                     "type": "string"
#                 },
#                 "next": {
#                     "anyOf": [
#                         {
#                             "$ref": "#/$defs/A"
#                         },
#                         {
#                             "type": "null"
#                         }
#                     ]
#                 }
#             },
#             "type": "object"
#         }
#     },
#     "type": "object",
#     "properties": {
#         "my_list": {
#             "anyOf": [
#                 {
#                     "$ref": "#/$defs/A"
#                 },
#                 {
#                     "type": "null"
#                 }
#             ]
#         }
#     }
# }
#         """
#     schema_obj = json.loads(schema)

#     m = azure_guidance_model
#     m += gen_json(schema=schema_obj, name="my_json_string")
#     print(f"Raw: {m['my_json_string']}")

#     my_obj = json.loads(m["my_json_string"])
#     print(f"Received object: {json.dumps(my_obj, indent=4)}")
#     validate(my_obj, schema_obj)


# @pytest.mark.parametrize(
#     "test_str",
#     [
#         "is this legal",
#         "I'm not sure ias;ldlkas is the best",
#         "\n\nit works\n\n",
#         "0123456789",
#     ],
# )
# def test_mocked_substring(test_str, azure_guidance_model: guidance.models.Model):
#     m = azure_guidance_model

#     lm = m + substring(test_str, name="result")
#     print(f'Substring: \'{lm["result"]}\'  ::::  \'{test_str}\'')
#     assert lm["result"] in test_str


def test_azure_guidance_stateless_inside_stateful(azure_guidance_model: guidance.models.Model):
    @guidance(stateless=False, dedent=False)
    def stateful_grammar1(lm):
        return lm + select(["+", "-"]) + stateful_grammar2()

    @guidance(stateless=False, dedent=False)
    def stateful_grammar2(lm):
        return lm + "p4" + stateless_grammar1()

    @guidance(stateless=True, dedent=False)
    def stateless_grammar1(lm):
        return lm + "3L" + stateless_grammar2()

    @guidance(stateless=True, dedent=False)
    def stateless_grammar2(lm):
        return lm + "Yy" + stateless_grammar3()

    @guidance(stateless=True, dedent=False)
    def stateless_grammar3(lm):
        return lm + select(["A", "B"])

    lm = azure_guidance_model
    lm += "begin:" + stateful_grammar1()
    result = str(lm)
    assert result == "begin:+p43LYyA" or result == "begin:-p43LYyA" or result == "begin:+p43LYyB" or result == "begin:-p43LYyB"


def test_azure_guidance_string(azure_guidance_model: guidance.models.Model):
    model = azure_guidance_model
    # limit number of tokens, otherwise test is very slow
    s = str(model + "ab" + token_limit(one_or_more("ab"), 30))
    assert len(s) >= 4
    assert bool(re.fullmatch(r'(ab)*', s)) or bool(re.fullmatch(r'(ab)*', s[:-1]))


def test_azure_guidance_stop_token_name(azure_guidance_model: guidance.models.Model):
    lm = azure_guidance_model
    lm += "Name: " + gen('name', regex="E[a-z]+", stop_regex=["[a-b]", "[x-z]"], save_stop_text="saved_name_stop")
    assert lm["saved_name_stop"] in ["a", "b", "x", "y", "z"]
    assert lm["name"].startswith("E")

def test_azure_guidance_stop_token_name2(azure_guidance_model: guidance.models.Model):
    lm = azure_guidance_model
    # repeat the token to get duplicated lexeme
    lm += "Name: " + gen('name', regex="E[a-z]+", stop_regex=["[a-b]", "[x-z]"], save_stop_text="saved_name_stop") + \
    "\nName: " + gen('name2', regex="E[a-z]+", stop_regex=["[a-b]", "[x-z]"], save_stop_text=True)
    assert lm["saved_name_stop"] in ["a", "b", "x", "y", "z"]
    assert lm["name"].startswith("E")
    assert lm["name2_stop_text"] in ["a", "b", "x", "y", "z"]
    assert lm["name2"].startswith("E")

def test_azure_guidance_max_tokens_4(azure_guidance_model: guidance.models.Model):
    lm = azure_guidance_model
    lm += "Name: " + gen('name', max_tokens=5) + " and " + gen('name2', max_tokens=5)
    assert len(lm["name"]) > 0
    assert len(lm["name2"]) > 0

def test_azure_guidance_zero_temperature(azure_guidance_model: guidance.models.Model):
    lm = azure_guidance_model
    responses = []
    for _ in range(10):
        temp = lm + "Number: " + gen("number", regex=r"\d", temperature=0.0)
        responses.append(temp["number"])
    assert len(set(responses)) == 1

def test_azure_guidance_high_temperature(azure_guidance_model: guidance.models.Model):
    lm = azure_guidance_model
    responses = []
    for _ in range(10):
        temp = lm + "Number: " + gen("number", regex=r"\d", temperature=0.9)
        responses.append(temp["number"])
    assert len(set(responses)) > 1