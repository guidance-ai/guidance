import numpy as np
import pytest

import guidance
from guidance import gen, select

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
