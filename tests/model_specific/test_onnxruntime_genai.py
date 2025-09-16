import platform
import sys

import numpy as np
import pytest

import guidance
from guidance import gen, select
from guidance.models._engine._interpreter import text_to_grammar
from tests.tokenizer_common import TOKENIZER_ROUND_TRIP_STRINGS


def test_onnxrt_gen(onnxrt_model: guidance.models.Model):
    lm = onnxrt_model
    lm = lm + "this is a test" + gen("test", max_tokens=10)
    assert len(str(lm)) > len("this is a test")


def test_onnxrt_gen_log_probs(onnxrt_model: guidance.models.Model):
    lm = onnxrt_model
    lm = lm + "this is a test" + gen("test", max_tokens=1)
    assert 1 >= np.exp(lm.log_prob("test")) >= 0


def test_onnxrt_recursion_error(onnxrt_model: guidance.models.Model):
    lm = onnxrt_model

    # define a guidance program that adapts a proverb
    lm = (
        lm
        + f"""Tweak this proverb to apply to model instructions instead.
    {gen("verse", max_tokens=2)}
    """
    )
    assert len(str(lm)) > len("Tweak this proverb to apply to model instructions instead.\n\n")


def test_onnxrt_select2(onnxrt_model: guidance.models.Model):
    lm = onnxrt_model
    lm += f"this is a test1 {select(['item1', 'item2'])} and test2 {select(['item3', 'item4'])}"
    assert str(lm) in [
        "this is a test1 item1 and test2 item3",
        "this is a test1 item1 and test2 item4",
        "this is a test1 item2 and test2 item3",
        "this is a test1 item2 and test2 item4",
    ]


def test_suffix(onnxrt_model: guidance.models.Model):
    llama2 = onnxrt_model
    lm = llama2 + "1. Here is a sentence " + gen(name="bla", list_append=True, suffix="\n")
    assert (str(lm))[-1] == "\n"
    assert (str(lm))[-2] != "\n"


def test_subtoken_forced(onnxrt_model: guidance.models.Model):
    llama2 = onnxrt_model
    lm = llama2 + "How much is 2 + 2? " + gen(name="test", max_tokens=10, regex=r"\(")
    assert str(lm) == "How much is 2 + 2? ("


def test_onnxrt_almost_one_batch(onnxrt_model):
    lm = onnxrt_model
    batch_size = 1
    long_str = lm.engine.tokenizer.bos_token.decode("utf-8") * (batch_size - 1)
    lm += text_to_grammar(lm.engine.tokenizer, long_str) + gen(max_tokens=10, regex=r".+")
    assert len(str(lm)) > len(long_str)


def test_onnxrt_exactly_one_batch(onnxrt_model):
    lm = onnxrt_model
    batch_size = 1
    long_str = lm.engine.tokenizer.bos_token.decode("utf-8") * batch_size
    lm += text_to_grammar(lm.engine.tokenizer, long_str) + gen(max_tokens=10, regex=r".+")
    assert len(str(lm)) > len(long_str)


def test_onnxrt_more_than_one_batch(onnxrt_model):
    lm = onnxrt_model
    batch_size = 1
    long_str = lm.engine.tokenizer.bos_token.decode("utf-8") * (batch_size + 1)
    lm += text_to_grammar(lm.engine.tokenizer, long_str) + gen(max_tokens=10, regex=r".+")
    assert len(str(lm)) > len(long_str)


def test_onnxrt_almost_two_batches(onnxrt_model):
    lm = onnxrt_model
    batch_size = 1
    long_str = lm.engine.tokenizer.bos_token.decode("utf-8") * ((2 * batch_size) - 1)
    lm += text_to_grammar(lm.engine.tokenizer, long_str) + gen(max_tokens=10, regex=r".+")
    assert len(str(lm)) > len(long_str)


def test_onnxrt_two_batches(onnxrt_model):
    lm = onnxrt_model
    batch_size = 1
    long_str = lm.engine.tokenizer.bos_token.decode("utf-8") * (2 * batch_size)
    lm += text_to_grammar(lm.engine.tokenizer, long_str) + gen(max_tokens=10, regex=r".+")
    assert len(str(lm)) > len(long_str)


def test_onnxrt_more_than_two_batches(onnxrt_model):
    lm = onnxrt_model
    batch_size = 1
    long_str = lm.engine.tokenizer.bos_token.decode("utf-8") * ((2 * batch_size) + 1)
    lm += text_to_grammar(lm.engine.tokenizer, long_str) + gen(max_tokens=10, regex=r".+")
    assert len(str(lm)) > len(long_str)


def test_llama_with_temp(onnxrt_model: guidance.models.Model):
    lm = onnxrt_model
    lm += "Here is a cute 5-line poem about cats and dogs:\n"
    for i in range(5):
        lm += f"LINE {i + 1}: " + gen(temperature=0.8, suffix="\n")
    # we just want to make sure we don't crash the numpy sampler


def test_llama_with_temp2(onnxrt_model: guidance.models.Model):
    lm = onnxrt_model
    lm1 = lm + "2 + 2 =" + gen("answer", max_tokens=3)
    lm2 = lm + "2 + 2 =" + gen("answer", temperature=0.0000001, max_tokens=3)
    assert lm1["answer"] == lm2["answer"]


def test_max_tokens(onnxrt_model: guidance.models.Model):
    lm = onnxrt_model
    lm += "Who won the last Kentucky derby and by how much?"
    lm += "\n\n<<The last Kentucky Derby was held"
    lm += gen(max_tokens=2)
    assert str(lm)[-1] != "<"  # the output should not end with "<" because that is coming from the stop sequence...
