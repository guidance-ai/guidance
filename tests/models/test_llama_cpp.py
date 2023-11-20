import guidance
from guidance import select, gen
from ..utils import get_model

def test_llama_cpp_gen():
    lm = get_model("llama_cpp:")
    lm = lm + "this is a test" + gen("test", max_tokens=10)
    assert len(str(lm)) > len("this is a test")

def test_llama_cpp_recursion_error():
    lm = get_model("llama_cpp:")

    # define a guidance program that adapts a proverb
    lm = lm + f"""Tweak this proverb to apply to model instructions instead.
    {gen('verse', max_tokens=2)}
    """
    assert len(str(lm)) > len("Tweak this proverb to apply to model instructions instead.\n\n")

def test_llama_cpp_select2():
    lm = get_model("llama_cpp:")
    lm += f'this is a test1 {select(["item1", "item2"])} and test2 {select(["item3", "item4"])}'
    assert str(lm) in [
        "this is a test1 item1 and test2 item3", 
        "this is a test1 item1 and test2 item4",
        "this is a test1 item2 and test2 item3", 
        "this is a test1 item2 and test2 item4"]

def test_repeat_calls():
    llama2 = get_model("llama_cpp:")
    a = []
    lm = llama2 + 'How much is 2 + 2? ' + gen(name='test', max_tokens=10)
    a.append(lm['test'])
    lm = llama2 + 'How much is 2 + 2? ' + gen(name='test',max_tokens=10, pattern=r'\d+')
    a.append(lm['test'])
    lm = llama2 + 'How much is 2 + 2? ' + gen(name='test', max_tokens=10)
    a.append(lm['test'])
    assert a[-1] == a[0]

def test_suffix():
    llama2 = get_model("llama_cpp:")
    lm = llama2 + '1. Here is a sentence ' + gen(name='bla', list_append=True, suffix='\n')
    assert (str(lm))[-1] == '\n'
    assert (str(lm))[-2] != '\n'

def test_subtoken_forced():
    llama2 = get_model("llama_cpp:")
    lm = llama2 + 'How much is 2 + 2? ' + gen(name='test', max_tokens=10, regex=r'\(')
    assert str(lm) == "How much is 2 + 2? ("

def test_llama_cpp_exactly_one_batch():
    lm = get_model("llama_cpp:", n_batch=9)
    long_str = lm.bos_token.decode("utf-8") * 9
    lm += long_str + gen(max_tokens=10)
    assert len(str(lm)) > len(long_str)

def test_llama_cpp_almost_two_batches():
    lm = get_model("llama_cpp:", n_batch=8)
    long_str = lm.bos_token.decode("utf-8") * 15
    lm += long_str + gen(max_tokens=10)
    assert len(str(lm)) > len(long_str)

def test_llama_cpp_two_batches():
    lm = get_model("llama_cpp:", n_batch=7)
    long_str = lm.bos_token.decode("utf-8") * 14
    lm += long_str + gen(max_tokens=10)
    assert len(str(lm)) > len(long_str)

def test_llama_cpp_more_than_two_batches():
    lm = get_model("llama_cpp:", n_batch=6)
    long_str = lm.bos_token.decode("utf-8") * 13
    lm += long_str + gen(max_tokens=10)
    assert len(str(lm)) > len(long_str)

