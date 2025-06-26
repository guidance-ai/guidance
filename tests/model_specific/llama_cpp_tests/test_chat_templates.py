import jinja2
import pytest

import guidance


def test_chat_format_smoke(llamacpp_model: guidance.models.LlamaCpp):
    # Retrieve the template string
    if (
        hasattr(llamacpp_model.engine.model_obj, "metadata")
        and "tokenizer.chat_template" in llamacpp_model.engine.model_obj.metadata
    ):
        model_chat_template = llamacpp_model.engine.model_obj.metadata["tokenizer.chat_template"]
    else:
        pytest.skip("Chat template not available from LlamaCpp object")

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
        bos_token=llamacpp_model.engine.tokenizer.bos_token.decode(),
        eos_token=llamacpp_model.engine.tokenizer.eos_token.decode(),
    )

    lm = llamacpp_model
    with guidance.user():
        lm += "Good_day_to_you!"
    with guidance.assistant():
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
