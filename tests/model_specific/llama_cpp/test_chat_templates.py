import jinja2
import llama_cpp

import guidance

from guidance.chat import CHAT_TEMPLATE_CACHE


def test_chat_format_smoke(llamacpp_model: guidance.models.LlamaCpp):
    # Retrieve the template string
    model_chat_template = llamacpp_model.engine.model_obj.metadata["tokenizer.chat_template"]

    lm = guidance.models.Mock("")
    lm.chat_template = CHAT_TEMPLATE_CACHE[model_chat_template]()

    messages = [
        {"role": "user", "content": "Good day to you!"},
        {"role": "assistant", "content": "Hello!"},
    ]

    # Note that llama-cpp-python does provide a llama_chat_apply_template function
    # but details about its use are thin on the ground and according to
    # https://github.com/ggerganov/llama.cpp/wiki/Templates-supported-by-llama_chat_apply_template
    # it does its own thing internally
    jinja2_template = jinja2.Environment(loader=jinja2.BaseLoader()).from_string(
        model_chat_template
    )
    jinja2_render = jinja2_template.render(messages=messages, eos_token=llamacpp_model.engine.tokenizer.eos_token.decode())

    with guidance.user():
        lm += "Good day to you!"
    with guidance.assistant():
        lm += "Hello!"
    # Only check substring due to BOS/EOS tokens
    assert str(lm) in jinja2_render
