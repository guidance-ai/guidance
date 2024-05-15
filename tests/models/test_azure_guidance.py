import numpy as np
import pytest

import guidance
from guidance import gen, select, assistant, user

from ..utils import get_model


@pytest.fixture(scope="module")
def azure_guidance_model(selected_model, selected_model_name):
    if selected_model_name in ["azure_guidance"]:
        return selected_model
    else:
        pytest.skip("Requires Azure Guidance model")


def test_azure_guidance_gen(azure_guidance_model: guidance.models.Model):
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
    lm = lm_orig + "How much is 2 + 2? " + gen(name="test", max_tokens=10, regex=r"\d+")
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
    assert (str(lm))[-1] == "\n"
    assert (str(lm))[-2] != "\n"


def test_azure_guidance_subtoken_forced(azure_guidance_model: guidance.models.Model):
    lm_orig = azure_guidance_model
    lm = lm_orig + "How much is 2 + 2? " + gen(name="test", max_tokens=10, regex=r"\(")
    assert str(lm) == "How much is 2 + 2? ("


def test_azure_guidance_with_temp(azure_guidance_model: guidance.models.Model):
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


def test_azure_guidance_max_tokens(azure_guidance_model: guidance.models.Model):
    lm = azure_guidance_model
    lm += "Who won the last Kentucky derby and by how much?"
    lm += "\n\n<<The last Kentucky Derby was held"
    lm += gen(max_tokens=2)
    assert (
        str(lm)[-1] != "<"
    )  # the output should not end with "<" because that is coming from the stop sequence...


def test_azure_guidance_stop_token(azure_guidance_model: guidance.models.Model):
    lm = azure_guidance_model
    lm += f'<color>red</color>\n<color>{gen(stop="</color>")} and test2'
    r = str(lm)
    print(r)
    print(r[20:])
    assert "</color>" not in r[20:]
    assert " and test2" in r[20:]

def test_azure_guidance_basic(azure_guidance_model: guidance.models.Model):
    model = azure_guidance_model
    lm = model + "Count to 20: 1,2,3,4,"
    nl = "\n"
    lm += "5,6,7" + f"""{gen(max_tokens=1, suffix=nl)}aaaaaa"""
    assert str(lm)[-6:] == "aaaaaa"

def test_azure_guidance_fstring(azure_guidance_model: guidance.models.Model):
    lm = azure_guidance_model
    lm += f'this is a test {select(["item1", "item2"])}'
    assert str(lm) in ["this is a test item1", "this is a test item2"]


def test_azure_guidance_fstring_custom(azure_guidance_model: guidance.models.Model):
    lm = azure_guidance_model

    @guidance
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


def test_azure_guidance_stream(azure_guidance_model: guidance.models.Model):
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


def test_azure_guidance(azure_guidance_model: guidance.models.Model):
    lm = azure_guidance_model
    with user():
        lm += "What is 1 + 1?"
    with assistant():
        lm += gen(max_tokens=10, name="text")
        lm += "Pick a number: "
    assert len(lm["text"]) > 0


def test_azure_guidance_select(azure_guidance_model: guidance.models.Model):
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

    with user():
        lm += "The economy is crashing!"

    with assistant():
        lm += gen("test1", max_tokens=100)

    with user():
        lm += "What is the best again?"

    with assistant():
        lm += gen("test2", max_tokens=100)

    assert len(lm["test1"]) > 0
    assert len(lm["test2"]) > 0

    # second time to make sure cache reuse is okay
    lm = azure_guidance_model

    with user():
        lm += "The economy is crashing!"

    with assistant():
        lm += gen("test1", max_tokens=100)

    with user():
        lm += "What is the best again?"

    with assistant():
        lm += gen("test2", max_tokens=100)

    assert len(lm["test1"]) > 0
    assert len(lm["test2"]) > 0
