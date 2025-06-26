import pytest
import transformers

import guidance
from guidance.chat import CHAT_TEMPLATE_CACHE

from ..utils import env_or_skip


@pytest.mark.parametrize(
    ("model_id", "should_pass"),
    [
        ("microsoft/Phi-3-mini-4k-instruct", True),  # Phi-3-Mini
        ("microsoft/Phi-3-small-8k-instruct", True),  # Phi-3-Small
        ("microsoft/Phi-3-medium-4k-instruct", True),  # Phi-3-Medium
        ("meta-llama/Meta-Llama-3-8B-Instruct", True),  # Llama-3
        ("meta-llama/Llama-2-7b-chat-hf", True),  # Llama-2
        ("mistralai/Mistral-7B-Instruct-v0.2", True),  # Mistral-7B-Instruct-v0.2
        ("google/gemma-2-9b-it", True),  # Gemma2
        ("HuggingFaceH4/zephyr-7b-beta", False),  # Have a test for model not in cache
        ("Qwen/Qwen2.5-0.5B", True),  # Qwen2.5-0.5B
        ("Qwen/Qwen2.5-0.5B-Instruct", True),  # Qwen2.5-0.5B-Instruct
    ],
)
def test_popular_models_in_cache(model_id: str, should_pass: bool):
    # This test simply checks to make sure the chat_templates haven't changed, and that they're still in our cache.
    # If this fails, the models have had their templates updated, and we need to fix the cache manually.
    hf_token = env_or_skip("HF_TOKEN")

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, token=hf_token, trust_remote_code=True)
    model_chat_template = tokenizer.chat_template
    if should_pass:
        assert model_chat_template in CHAT_TEMPLATE_CACHE
    else:
        # TODO: Expand this test to verify that a warning gets thrown when a model isn't in the cache and we have to default to chatML syntax
        assert model_chat_template not in CHAT_TEMPLATE_CACHE


# TODO: Expand testing to verify that tokenizer.apply_chat_template() produces same results as our ChatTemplate subclasses
# once I hook up the new ChatTemplate to guidance.models.Transformers and guidance.models.LlamaCPP, we can do this


@pytest.mark.skip(reason="Is this supposed to work still? See issue 1196")
@pytest.mark.parametrize(
    "model_id",
    [
        "microsoft/Phi-3-mini-4k-instruct",
        "microsoft/Phi-3-small-8k-instruct",
        "microsoft/Phi-3-medium-4k-instruct",
        "meta-llama/Meta-Llama-3-8B-Instruct",
        "meta-llama/Llama-2-7b-chat-hf",
        "mistralai/Mistral-7B-Instruct-v0.2",
        "google/gemma-2-9b-it",
    ],
)
def test_chat_format_smoke(model_id: str):
    hf_token = env_or_skip("HF_TOKEN")

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, token=hf_token, trust_remote_code=True)
    model_chat_template = tokenizer.chat_template

    lm = guidance.models.Mock("")
    lm.chat_template = CHAT_TEMPLATE_CACHE[model_chat_template]()

    messages = [
        {"role": "user", "content": "Good day to you!"},
        {"role": "assistant", "content": "Hello!"},
    ]
    tokeniser_render = tokenizer.apply_chat_template(messages, tokenize=False)

    with guidance.user():
        lm += "Good day to you!"
    with guidance.assistant():
        lm += "Hello!"
    # Only check substring due to BOS/EOS tokens
    assert str(lm) in tokeniser_render


@pytest.mark.skip(reason="Is this supposed to work still? See issue 1196")
@pytest.mark.parametrize(
    "model_id",
    [
        "microsoft/Phi-3-mini-4k-instruct",
        "meta-llama/Meta-Llama-3-8B-Instruct",
        "Qwen/Qwen2.5-0.5B",
        "Qwen/Qwen2.5-0.5B-Instruct",
        pytest.param(
            "meta-llama/Llama-2-7b-chat-hf",
            marks=pytest.mark.xfail(
                reason="Handling of system prompt highly constrained; does not work well with context blocks",
                raises=AssertionError,
            ),
        ),
    ],
)
def test_chat_format_smoke_with_system(model_id: str):
    hf_token = env_or_skip("HF_TOKEN")

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, token=hf_token, trust_remote_code=True)
    model_chat_template = tokenizer.chat_template

    lm = guidance.models.Mock("")
    lm.chat_template = CHAT_TEMPLATE_CACHE[model_chat_template]()

    messages = [
        {"role": "system", "content": "You are an LLM"},
        {"role": "user", "content": "Good day to you!"},
        {"role": "assistant", "content": "Hello!"},
    ]
    tokeniser_render = tokenizer.apply_chat_template(messages, tokenize=False)

    with guidance.system():
        lm += "You are an LLM"
    with guidance.user():
        lm += "Good day to you!"
    with guidance.assistant():
        lm += "Hello!"
    # Only check substring due to BOS/EOS tokens
    assert str(lm) in tokeniser_render
