import jinja2
import pytest

from guidance import assistant, gen, guidance, models, select, user

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


def test_gpt2_fastforward():  # TODO [HN]: figure out how all the get_model and fixture stuff works
    @guidance
    def ff_prompt(lm):
        big_opts = ["Lorem ipsum dolor sit amet", "Duis aute irure dolor "]
        lm += "Example text: " + select(big_opts, name="choice")
        return lm

    # We should have significantly less output tokens in the fast-forwarded version (1 output)

    gpt2_noff = models.Transformers("gpt2", enable_backtrack=False, enable_ff_tokens=False)
    gpt2_noff += ff_prompt()
    noff_count = gpt2_noff._get_usage().forward_passes

    gpt2_nobt = models.Transformers("gpt2", enable_backtrack=False)
    gpt2_nobt += ff_prompt()
    nobt_count = gpt2_nobt._get_usage().forward_passes

    gpt2_ff = models.Transformers("gpt2")
    gpt2_ff += ff_prompt()
    ff_count = gpt2_ff._get_usage().forward_passes

    # 3: 1 fp for the prompt, 1 for the first token of the select, 1 to get probs for the ff_tokens
    # To future devs, this should be 2 if we turn off the extra fp, e.g. when metrics are disabled
    assert nobt_count == 3
    assert ff_count == 3
    assert noff_count > ff_count


def test_recursion_error():
    """This checks for an infinite recursion error resulting from a terminal node at the root of a grammar."""
    gpt2 = get_model("transformers:gpt2")

    # define a guidance program that adapts a proverb
    lm = (
        gpt2
        + f"""Tweak this proverb to apply to model instructions instead.
    {gen("verse", max_tokens=2)}
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


def test_chat_format_smoke(transformers_model: models.Transformers):
    # Retrieve the template string
    if isinstance(transformers_model.engine.tokenizer._orig_tokenizer.chat_template, str):
        model_chat_template = transformers_model.engine.tokenizer._orig_tokenizer.chat_template
    else:
        pytest.skip("Chat template not available from Transformers object")

    messages = [
        {"role": "user", "content": "Good_day_to_you!"},
        {"role": "assistant", "content": "Hello!"},
    ]

    # Note that llama-cpp-python does provide a llama_chat_apply_template function
    # but details about its use are thin on the ground and according to
    # https://github.com/ggerganov/llama.cpp/wiki/Templates-supported-by-llama_chat_apply_template
    # it does its own thing internally
    jinja2_template = jinja2.Environment(loader=jinja2.BaseLoader()).from_string(model_chat_template)
    jinja2_render = jinja2_template.render(
        messages=messages,
        bos_token=transformers_model.engine.tokenizer.bos_token.decode(),
        eos_token=transformers_model.engine.tokenizer.eos_token.decode(),
    )

    lm = transformers_model
    with user():
        lm += "Good_day_to_you!"
    with assistant():
        lm += "Hello!"

    # Compare the tokenization of the strings, rather than the strings
    # themselves (e.g. `<|user|>` may tokenize the same as `<|user|>\n`)
    lm_tokens = lm._interpreter.engine.tokenizer.encode(str(lm).encode())
    jinja2_tokens = lm._interpreter.engine.tokenizer.encode(jinja2_render.encode())

    # Only check substring due to BOS/EOS tokens, unfinished closing tags
    diff = len(jinja2_tokens) - len(lm_tokens)
    assert diff >= 0
    for i in range(diff + 1):
        if jinja2_tokens[i : i + len(lm_tokens)] == lm_tokens:
            break
    else:
        raise AssertionError("lm mismatches jinja template", str(lm), str(jinja2_render))


def test_top_p_top_k_filtering():
    import numpy as np
    import torch
    from transformers.generation.logits_process import TopKLogitsWarper, TopPLogitsWarper

    from guidance._utils import apply_top_k_and_top_p_filter

    torch.random.manual_seed(0)
    logits = torch.randn((1, 1000))

    # apply top_k filtering
    top_k = 64
    top_k_warp = TopKLogitsWarper(top_k)
    transformers_logits = top_k_warp(None, logits)[0].numpy()
    guidance_logits = apply_top_k_and_top_p_filter(logits[0].numpy(), {"top_k": top_k})
    assert np.all(transformers_logits == guidance_logits), "Logits do not match after top_k filtering"

    # apply top_p filtering
    top_p = 0.9
    top_p_warp = TopPLogitsWarper(top_p)
    transformers_logits = top_p_warp(None, logits)[0].numpy()
    guidance_logits = apply_top_k_and_top_p_filter(logits[0].numpy(), {"top_p": top_p})
    assert np.all(transformers_logits == guidance_logits), "Logits do not match after top_p filtering"

    # apply both top_k and top_p filtering
    transformers_logits = top_p_warp(None, top_k_warp(None, logits))[0].numpy()
    guidance_logits = apply_top_k_and_top_p_filter(logits[0].numpy(), {"top_k": top_k, "top_p": top_p})
    assert np.all(transformers_logits == guidance_logits), "Logits do not match after top_k and top_p filtering"


def test_min_p_filtering():
    import numpy as np
    import torch
    from transformers.generation.logits_process import MinPLogitsWarper

    from guidance._utils import apply_min_p_filter

    torch.random.manual_seed(0)
    logits = torch.randn((1, 1000))

    # apply min_p filtering
    min_p = 0.1
    min_p_warp = MinPLogitsWarper(min_p)
    transformers_logits = min_p_warp(None, logits)[0].numpy()
    guidance_logits = apply_min_p_filter(logits[0].numpy(), {"min_p": min_p})
    assert np.all(transformers_logits == guidance_logits), "Logits do not match after min_p filtering"
