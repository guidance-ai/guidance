import jinja2
import pytest

import guidance


def test_chat_format_smoke(llamacpp_model: guidance.models.LlamaCpp, selected_model_name):
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
    jinja2_template = jinja2.Environment(loader=jinja2.BaseLoader()).from_string(
        model_chat_template
    )
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
    # Only check substring due to BOS/EOS tokens
    if selected_model_name == "llamacpp_mistral_7b_cpu":
        # The templates extracted via Transformers and GGUF are somewhat
        # different for Mistral. This is showing up in slightly
        # different spacing (our template is putting in a few extra spaces)
        # so at least make sure the 'tags' are correct
        assert str(lm).replace(" ", "") in jinja2_render.replace(" ", "")
    else:
        assert str(lm) in jinja2_render
