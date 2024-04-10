import numpy as np
import guidance
from guidance import select, gen
from ..utils import get_model

import pytest


@pytest.fixture(scope="session")
def llamacpp_model(selected_model, selected_model_name):
    if selected_model_name in ["hfllama7b"]:
        return selected_model
    else:
        pytest.skip("Requires Llama-Cpp model")


def get_llama_with_batchsize(model_info, n_batch: int = 248):
    # Now load the model we actually want
    my_kwargs = model_info["kwargs"]
    my_kwargs["n_batch"] = n_batch
    lm = get_model(model_info["name"], **my_kwargs)
    return lm


def test_llama_cpp_gen(llamacpp_model: guidance.models.Model):
    lm = llamacpp_model
    lm = lm + "this is a test" + gen("test", max_tokens=10)
    assert len(str(lm)) > len("this is a test")

def test_llama_cpp_gen_log_probs(llamacpp_model: guidance.models.Model):
    lm = llamacpp_model
    lm = lm + "this is a test" + gen("test", max_tokens=1)
    assert 1 >= np.exp(lm.log_prob("test")) >= 0

def test_llama_cpp_recursion_error(llamacpp_model: guidance.models.Model):
    lm = llamacpp_model

    # define a guidance program that adapts a proverb
    lm = lm + f"""Tweak this proverb to apply to model instructions instead.
    {gen('verse', max_tokens=2)}
    """
    assert len(str(lm)) > len("Tweak this proverb to apply to model instructions instead.\n\n")

def test_llama_cpp_select2(llamacpp_model: guidance.models.Model):
    lm = llamacpp_model
    lm += f'this is a test1 {select(["item1", "item2"])} and test2 {select(["item3", "item4"])}'
    assert str(lm) in [
        "this is a test1 item1 and test2 item3", 
        "this is a test1 item1 and test2 item4",
        "this is a test1 item2 and test2 item3", 
        "this is a test1 item2 and test2 item4"]

def test_repeat_calls(llamacpp_model: guidance.models.Model):
    llama2 = llamacpp_model
    a = []
    lm = llama2 + 'How much is 2 + 2? ' + gen(name='test', max_tokens=10)
    a.append(lm['test'])
    lm = llama2 + 'How much is 2 + 2? ' + gen(name='test',max_tokens=10, regex=r'\d+')
    a.append(lm['test'])
    lm = llama2 + 'How much is 2 + 2? ' + gen(name='test', max_tokens=10)
    a.append(lm['test'])
    assert a[-1] == a[0]

def test_suffix(llamacpp_model: guidance.models.Model):
    llama2 = llamacpp_model
    lm = llama2 + '1. Here is a sentence ' + gen(name='bla', list_append=True, suffix='\n')
    assert (str(lm))[-1] == '\n'
    assert (str(lm))[-2] != '\n'

def test_subtoken_forced(llamacpp_model: guidance.models.Model):
    llama2 = llamacpp_model
    lm = llama2 + 'How much is 2 + 2? ' + gen(name='test', max_tokens=10, regex=r'\(')
    assert str(lm) == "How much is 2 + 2? ("

def test_llama_cpp_almost_one_batch(llamacpp_model, selected_model_info):
    batch_size = 248
    # We use llamacpp_model for its 'skip' functionality
    assert llamacpp_model is not None
    # Now load the model we actually want
    lm = get_llama_with_batchsize(selected_model_info, batch_size)
    long_str = lm.engine.tokenizer.bos_token.decode("utf-8") * (batch_size - 1)
    lm += long_str + gen(max_tokens=10)
    assert len(str(lm)) > len(long_str)

def test_llama_cpp_exactly_one_batch(llamacpp_model, selected_model_info):
    batch_size = 248
    # We use llamacpp_model for its 'skip' functionality
    assert llamacpp_model is not None
    # Now load the model we actually want
    lm = get_llama_with_batchsize(selected_model_info, batch_size)
    long_str = lm.engine.tokenizer.bos_token.decode("utf-8") * batch_size
    lm += long_str + gen(max_tokens=10)
    assert len(str(lm)) > len(long_str)

def test_llama_cpp_more_than_one_batch(llamacpp_model, selected_model_info):
    batch_size = 248
    # We use llamacpp_model for its 'skip' functionality
    assert llamacpp_model is not None
    # Now load the model we actually want
    lm = get_llama_with_batchsize(selected_model_info, batch_size)
    long_str = lm.engine.tokenizer.bos_token.decode("utf-8") * (batch_size + 1)
    lm += long_str + gen(max_tokens=10)
    assert len(str(lm)) > len(long_str)

def test_llama_cpp_almost_two_batches(llamacpp_model, selected_model_info):
    batch_size = 248
    # We use llamacpp_model for its 'skip' functionality
    assert llamacpp_model is not None
    # Now load the model we actually want
    lm = get_llama_with_batchsize(selected_model_info, batch_size)
    long_str = lm.engine.tokenizer.bos_token.decode("utf-8") * ((2 * batch_size) - 1)
    lm += long_str + gen(max_tokens=10)
    assert len(str(lm)) > len(long_str)

def test_llama_cpp_two_batches(llamacpp_model, selected_model_info):
    batch_size = 248
    # We use llamacpp_model for its 'skip' functionality
    assert llamacpp_model is not None
    # Now load the model we actually want
    lm = get_llama_with_batchsize(selected_model_info, batch_size)
    long_str = lm.engine.tokenizer.bos_token.decode("utf-8") * (2 * batch_size)
    lm += long_str + gen(max_tokens=10)
    assert len(str(lm)) > len(long_str)

def test_llama_cpp_more_than_two_batches(llamacpp_model, selected_model_info):
    batch_size = 248
    # We use llamacpp_model for its 'skip' functionality
    assert llamacpp_model is not None
    # Now load the model we actually want
    lm = get_llama_with_batchsize(selected_model_info, batch_size)
    long_str = lm.engine.tokenizer.bos_token.decode("utf-8") * ((2 * batch_size) + 1)
    lm += long_str + gen(max_tokens=10)
    assert len(str(lm)) > len(long_str)

def test_llama_with_temp(llamacpp_model: guidance.models.Model):
    lm = llamacpp_model
    lm += 'Here is a cute 5-line poem about cats and dogs:\n'
    for i in range(5):
        lm += f"LINE {i+1}: " + gen(temperature=0.8, suffix="\n")
    # we just want to make sure we don't crash the numpy sampler

def test_llama_with_temp2(llamacpp_model: guidance.models.Model):
    lm = llamacpp_model
    lm1 = lm + '2 + 2 =' + gen('answer', max_tokens=3)
    lm2 = lm + '2 + 2 =' + gen('answer', temperature=0.0000001, max_tokens=3)
    assert lm1["answer"] == lm2["answer"]

def test_max_tokens(llamacpp_model: guidance.models.Model):
    lm = llamacpp_model
    lm += "Who won the last Kentucky derby and by how much?"
    lm += "\n\n<<The last Kentucky Derby was held"
    lm += gen(max_tokens=2)
    assert str(lm)[-1] != "<" # the output should not end with "<" because that is coming from the stop sequence...
