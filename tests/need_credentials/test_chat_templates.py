import pytest

from guidance.chat import load_template_class, CHAT_TEMPLATE_CACHE
import transformers

from ..utils import env_or_fail

@pytest.mark.needs_credentials
@pytest.mark.parametrize(
    "model_info",
    [
        ("microsoft/Phi-3-mini-4k-instruct", True), # Phi-3
        ("meta-llama/Meta-Llama-3-8B-Instruct", True), # Llama-3
        ("meta-llama/Llama-2-7b-chat-hf", True), # Llama-2
        ("mistralai/Mistral-7B-Instruct-v0.2", True), # Mistral-7B-Instruct-v0.2
        ("HuggingFaceH4/zephyr-7b-beta", False) # Have a test for model not in cache
    ],
)
def test_popular_models_in_cache(model_info):
    # This test simply checks to make sure the chat_templates haven't changed, and that they're still in our cache.
    # If this fails, the models have had their templates updated, and we need to fix the cache manually.
    hf_token = env_or_fail("HF_TOKEN")

    model_id, should_pass = model_info

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, token=hf_token)
    model_chat_template = tokenizer.chat_template
    if should_pass:
        assert model_chat_template in CHAT_TEMPLATE_CACHE
    else:
        # TODO: Expand this test to verify that a warning gets thrown when a model isn't in the cache and we have to default to chatML syntax
        assert model_chat_template not in CHAT_TEMPLATE_CACHE

    

# TODO: Expand testing to verify that tokenizer.apply_chat_template() produces same results as our ChatTemplate subclasses
# once I hook up the new ChatTemplate to guidance.models.Transformers and guidance.models.LlamaCPP, we can do this


