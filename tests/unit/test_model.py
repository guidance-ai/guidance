import pytest

import guidance
from guidance import gen, models


def test_call_embeddings():
    """This tests calls embedded in strings."""
    model = models.Mock()

    @guidance(dedent=False)
    def bla(lm, bla):
        lm += bla + "ae" + gen(max_tokens=10)
        return lm

    @guidance(dedent=False)
    def ble(lm):
        lm += f"""
    ae galera! {bla("33")}
    let's do more stuff!!""" + gen(max_tokens=10)
        return lm

    assert "{{G|" not in str(model + ble())


@pytest.mark.xfail(
    reason="llguidance currently emits an additional empty capture group when no explicit stop is provided"
)
def test_model_set():
    model = models.Mock()
    model = model.set("num", "4")
    assert "num" in model
    assert model["num"] == "4"
    assert model.log_prob("num") is not None

    model = model.set("list_num", ["1", "2"])
    assert "list_num" in model
    assert model["list_num"] == ["1", "2"]
    assert model.log_prob("list_num") is not None

    model += gen("list_num", max_tokens=10, list_append=True)
    assert len(model["list_num"]) == 3


def test_trace():
    from guidance import gen, models, system, user

    m0 = models.Mock()

    with system():
        m1 = m0 + "You are responsible for autocompleting a sentence."
    with user():
        m2 = m1 + "Roses are red and " + gen(name="suffix", regex="[A-Za-z]{2,5}", max_tokens=5)

    assert m2["suffix"] is not None


class _DefaultSystemPromptTemplate(guidance.chat.ChatMLTemplate):
    default_system_prompt = "You are a helpful assistant."


def _mock_with_template(template_cls):
    model = models.Mock()
    model._interpreter.chat_template = template_cls()
    return model


def test_default_system_prompt_inserted_when_missing():
    from guidance import user

    lm = _mock_with_template(_DefaultSystemPromptTemplate)
    with user():
        lm += "Hi"

    assert str(lm) == "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nHi"


def test_default_system_prompt_not_inserted_with_explicit_system():
    from guidance import system, user

    lm = _mock_with_template(_DefaultSystemPromptTemplate)
    with system():
        lm += "You are a cat expert."
    with user():
        lm += "Hi"

    assert "You are a helpful assistant." not in str(lm)
    assert str(lm).count("<|im_start|>system") == 1


def test_no_default_system_prompt_by_default():
    from guidance import user

    lm = _mock_with_template(guidance.chat.ChatMLTemplate)
    with user():
        lm += "Hi"

    assert str(lm) == "<|im_start|>user\nHi"


@pytest.mark.parametrize(
    "template_str", [guidance.chat.qwen2dot5_template, guidance.chat.qwen2dot5_it_template], ids=["base", "instruct"]
)
def test_default_system_prompt_matches_hf_template(template_str):
    import jinja2.sandbox

    from guidance import assistant, user

    lm = _mock_with_template(guidance.chat.load_template_class(template_str))
    with user():
        lm += "Hi"
    with assistant():
        lm += "Hello"

    env = jinja2.sandbox.ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    messages = [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}]
    expected = env.from_string(template_str).render(messages=messages, add_generation_prompt=False)
    # guidance doesn't emit the closing tag of the last role block
    assert str(lm) == expected.removesuffix("<|im_end|>\n")


def test_step_every_k_injection():
    import re

    lm = models.Mock(echo=False)

    calls = {"count": 0}

    def cb(ctx):
        calls["count"] += 1
        return {"injected_text": "[FIX]"}

    cfg = {
        "step_every_k": 4,
        "callback": cb,
    }
    lm = lm.with_step_config(cfg)

    lm = lm + gen(max_tokens=20, stop="\n", temperature=0.0)

    s = str(lm)
    # find all occurrences of [FIX] in s and their positions
    occurrences = [m.start() for m in re.finditer(r"\[FIX\]", s)]
    assert occurrences == [6, 18]
    assert calls["count"] == len(occurrences)


def test_step_stop_token_trigger_injection():
    lm = models.Mock(byte_patterns=[b"abc!\n"], echo=False)

    calls = {"count": 0}

    def cb(ctx):
        calls["count"] += 1
        return {"injected_text": "[FIX2]"}

    cfg = {
        "step_stop_tokens": {"ym"},
        "callback": cb,
    }
    lm = lm.with_step_config(cfg)

    lm = lm + gen(max_tokens=20, stop="\n", temperature=0.0)

    s = str(lm)
    assert "[FIX2]" in s and "ym" not in s
    assert calls["count"] == 1
