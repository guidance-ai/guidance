import guidance
from guidance import gen, capture
from ..utils import get_model

def test_gpt2():
    gpt2 = get_model("transformers:gpt2")
    lm = gpt2 + "this is a test" + gen("test", max_tokens=10)
    assert len(str(lm)) > len("this is a test")

def test_recursion_error():
    '''This is similar an infinite recursion error resulting from a terminal node at the root of a grammar.'''
    gpt2 = get_model("transformers:gpt2")

    # define a guidance program that adapts a proverb
    lm = gpt2 + f"""Tweak this proverb to apply to model instructions instead.
    {gen('verse', max_tokens=2)}
    """
    assert len(str(lm)) > len("Tweak this proverb to apply to model instructions instead.\n\n")