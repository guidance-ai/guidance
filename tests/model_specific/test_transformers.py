import pytest

from guidance import gen, select, models, assistant, system, user, guidance

from ..utils import get_model


@pytest.fixture(scope="module")
def phi3_model(selected_model, selected_model_name):
    if selected_model_name in ["transformers_phi3_mini_4k_instruct_cpu"]:
        return selected_model
    else:
        pytest.skip("Requires Phi3 model")


@pytest.fixture(scope="module")
def llama3_model(selected_model, selected_model_name):
    if selected_model_name in ["transformers_llama3_8b_cpu"] and selected_model is not None:
        return selected_model
    else:
        pytest.skip("Requires Llama3 model (needs HF_TOKEN to be set)")


def test_gpt2():
    gpt2 = get_model("transformers:gpt2")
    lm = gpt2 + "this is a test" + gen("test", max_tokens=10)

    assert len(str(lm)) > len("this is a test")

def test_gpt2_fastforward(): # TODO [HN]: figure out how all the get_model and fixture stuff works
    @guidance
    def ff_prompt(lm):
        big_opts = [
            "Lorem ipsum dolor sit amet",
            "Duis aute irure dolor "
        ]
        lm += "Example text: " + select(big_opts, name="choice")
        return lm

    # We should have significantly less output tokens in the fast-forwarded version (1 output)

    gpt2_noff = models.Transformers("gpt2", enable_backtrack=False, enable_ff_tokens=False)
    gpt2_noff += ff_prompt()
    str(gpt2_noff) # Trigger execution
    noff_count = gpt2_noff.engine.metrics.engine_output_tokens

    gpt2_nobt = models.Transformers("gpt2", enable_backtrack=False)
    gpt2_nobt += ff_prompt()
    str(gpt2_nobt) # Trigger execution
    nobt_count = gpt2_nobt.engine.metrics.engine_output_tokens

    gpt2_ff = models.Transformers("gpt2")
    gpt2_ff += ff_prompt()
    str(gpt2_ff) # Trigger execution
    ff_count = gpt2_ff.engine.metrics.engine_output_tokens

    assert nobt_count == 2
    assert ff_count == 2
    assert noff_count > ff_count






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
    assert len(str(lm)) > len("Tweak this proverb to apply to model instructions instead.\n\n")


TRANSFORMER_MODELS = {
    "gpt2": {},
    # "microsoft/phi-2": {"trust_remote_code": True},
}


@pytest.mark.parametrize(["model_name", "model_kwargs"], TRANSFORMER_MODELS.items())
def test_transformer_smoke_gen(model_name, model_kwargs):
    MAX_TOKENS = 2
    my_model = get_model(f"transformers:{model_name}", **model_kwargs)

    prompt = "How many sides has a triangle?"
    lm = my_model + prompt + gen(name="answer", max_tokens=MAX_TOKENS)
    assert len(lm["answer"]) > 0, f"Output: {lm['answer']}"

    # Make sure not too much was produced
    assert len(lm.engine.tokenizer.encode(lm["answer"].encode())) <= MAX_TOKENS, f"Output: {lm['answer']}"


@pytest.mark.parametrize(["model_name", "model_kwargs"], TRANSFORMER_MODELS.items())
def test_transformer_smoke_select(model_name, model_kwargs):
    my_model = get_model(f"transformers:{model_name}", **model_kwargs)

    prompt = """How many sides has a triangle?

p) 4
t) 3
w) 10"""
    lm = my_model + prompt + select(["p", "t", "w"], name="answer")

    assert lm["answer"] in ["p", "t", "w"]


# Phi-3 specific tests


@pytest.mark.skip("Don't overload the build machines")
def test_phi3_transformers_orig():
    import torch
    from transformers import AutoModelForCausalLM, pipeline, AutoTokenizer

    torch.random.manual_seed(0)
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/Phi-3-mini-4k-instruct",
        device_map="mps",
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )

    generation_args = {
        "max_new_tokens": 5,
        "return_full_text": True,
        "temperature": 0.0,
        "do_sample": False,
    }

    input_text = "You are a counting bot. Just keep counting numbers. 1,2,3,4"
    output = pipe(input_text, **generation_args)

    assert "5" in (output[0]["generated_text"])


def test_phi3_chat_basic(phi3_model: models.Model):
    lm = phi3_model

    with user():
        lm += "You are a counting bot. Just keep counting numbers."
    with assistant():
        lm += "1,2,3,4," + gen(name="five", max_tokens=20)

    assert "5" in lm["five"]


def test_phi3_chat_unrolled(phi3_model: models.Model):
    lm = phi3_model
    # Manually convert the chat format into completions style
    lm += f"""<|user|>\nYou are a counting bot. Just keep counting numbers.<|end|>\n<|assistant|>\n1,2,3,4,"""
    lm += gen("five", max_tokens=10)

    assert "5" in lm["five"]


def test_phi3_newline_chat(phi3_model: models.Model):
    lm = phi3_model

    lm += "You are a counting bot. Just keep counting numbers."
    with user():
        lm += "1\n2\n3\n4\n"
    with assistant():
        lm += "\n" + gen(name="five", max_tokens=1)
        lm += "\n" + gen(name="six", max_tokens=1)

    # This test would raise an exception earlier if we didn't fix the tokenizer.
    assert True


def test_phi3_unstable_tokenization(phi3_model: models.Model):
    lm = phi3_model

    lm += "You are a counting bot. Just keep counting numbers."
    with user():
        lm += "1,2,3,4,"
    with assistant():
        lm += "\n"  # comment and uncomment this line to get the error
        lm += gen(name="five", max_tokens=1)
        lm += "," + gen(name="six", max_tokens=1)

    assert True


def test_phi3_basic_completion_badtokens(phi3_model: models.Model):
    lm = phi3_model
    # Bad user tokens, but we should still generate /something/
    lm += f"""<|use\n\nYou are a counting bot. Just keep counting numbers.<|end|><|assistant|>1,2,3,4,"""
    lm += gen("five", max_tokens=10)

    assert len(lm["five"]) > 0
