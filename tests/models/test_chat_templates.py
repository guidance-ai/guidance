import pytest

from guidance._chat import load_template_class, CHAT_TEMPLATE_CACHE
import transformers

from ..utils import env_or_fail

def test_popular_models_in_cache():
    # This test simply checks to make sure the chat_templates haven't changed, and that they're still in our cache.
    # If this fails, the models have had their templates updated, and we need to fix the cache manually.
    hf_token = env_or_fail("HF_TOKEN")
    models_to_test = [
        "microsoft/Phi-3-mini-4k-instruct", # Phi-3
        "meta-llama/Meta-Llama-3-8B-Instruct", # Llama-3
        "meta-llama/Llama-2-7b-chat-hf", # Llama-2
        "mistralai/Mistral-7B-Instruct-v0.2", # Mistral-7B-Instruct-v0.2
    ]
    for model_id in models_to_test:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, token=hf_token)
        model_chat_template = tokenizer.chat_template
        assert model_chat_template in CHAT_TEMPLATE_CACHE

    # Extend tests to some random model not explicitly coded for in the cache
    not_in_cache_model = "HuggingFaceH4/zephyr-7b-beta"
    tokenizer = transformers.AutoTokenizer.from_pretrained(not_in_cache_model, token=hf_token)
    model_chat_template = tokenizer.chat_template
    assert model_chat_template not in CHAT_TEMPLATE_CACHE

# TODO: Expand testing to verify that tokenizer.apply_chat_template() produces same results as our ChatTemplate subclasses
# once I hook up the new ChatTemplate to guidance.models.Transformers and guidance.models.LlamaCPP, we can do this


