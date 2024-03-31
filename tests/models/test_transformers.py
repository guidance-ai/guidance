import pytest

from guidance import gen, select

from ..utils import get_model


def test_gpt2():
    gpt2 = get_model("transformers:gpt2")
    lm = gpt2 + "this is a test" + gen("test", max_tokens=10)
    assert len(str(lm)) > len("this is a test")


def test_recursion_error():
    """This checks for an infinite recursion error resulting from a terminal node at the root of a grammar."""
    gpt2 = get_model("transformers:gpt2")

    # define a guidance program that adapts a proverb
    lm = (
        gpt2
        + f"""Tweak this proverb to apply to model instructions instead.
    {gen('verse', max_tokens=2)}
    """
    )
    assert len(str(lm)) > len(
        "Tweak this proverb to apply to model instructions instead.\n\n"
    )


TRANSFORMER_MODELS = {
    "gpt2": {},
    # "microsoft/phi-2": {"trust_remote_code": True},
}


@pytest.mark.parametrize(["model_name", "model_kwargs"], TRANSFORMER_MODELS.items())
def test_transformer_smoke_gen(model_name, model_kwargs):
    my_model = get_model(f"transformers:{model_name}", **model_kwargs)

    prompt = "How many sides has a triangle?"
    lm = my_model + prompt + gen(name="answer", max_tokens=2)
    assert len(lm["answer"]) > 0, f"Output: {lm['answer']}"
    # Inexact, but at least make sure not too much was produced
    assert len(lm["answer"]) < 8, f"Output: {lm['answer']}"


@pytest.mark.parametrize(["model_name", "model_kwargs"], TRANSFORMER_MODELS.items())
def test_transformer_smoke_select(model_name, model_kwargs):
    my_model = get_model(f"transformers:{model_name}", **model_kwargs)

    prompt = """How many sides has a triangle?

p) 4
t) 3
w) 10"""
    lm = my_model + prompt + select(["p", "t", "w"], name="answer")
    assert lm["answer"] in ["p", "t", "w"]
