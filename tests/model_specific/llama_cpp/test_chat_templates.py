import jinja2
import llama_cpp

import guidance

from guidance.chat import CHAT_TEMPLATE_CACHE


def test_chat_format_smoke(llamacpp_model: guidance.models.LlamaCpp):
    # Retrieve the template string
    model_chat_template = llamacpp_model.engine.tokenizer.chat_template

    lm = guidance.models.Mock("")
    lm.chat_template = CHAT_TEMPLATE_CACHE[model_chat_template]()

    messages = [
        {"role": "user", "content": "Good day to you!"},
        {"role": "assistant", "content": "Hello!"},
    ]

    jinja2_template = jinja2.Environment.from_string(model_chat_template)
    jinja2_render = jinja2_template.render(messages)

    with guidance.user():
        lm += "Good day to you!"
    with guidance.assistant():
        lm += "Hello!"
    # Only check substring due to BOS/EOS tokens
    assert str(lm) in jinja2_render
