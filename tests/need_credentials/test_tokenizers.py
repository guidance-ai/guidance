import pytest

from tests.tokenizer_common import TOKENIZER_ROUND_TRIP_STRINGS, BaseTestTransformerTokenizers

# Since this is under 'need_credentials' we can assume that HF_TOKEN
# will be available when run


class TestAuthenticatedTransformerTokenizers(BaseTestTransformerTokenizers):
    TRANSFORMER_MODELS = [
        # "google/gemma-2-9b-it", # Works locally, fails in build
        "meta-llama/Meta-Llama-3-8B-Instruct",
    ]

    @pytest.mark.parametrize(
        "model_name",
        TRANSFORMER_MODELS,
    )
    def test_smoke(self, model_name: str):
        try:
            self.base_smoke(model_name)
        except OSError:
            pytest.skip("HuggingFace raises OSError if user is not authenticated.")

    @pytest.mark.parametrize("model_name", TRANSFORMER_MODELS)
    @pytest.mark.parametrize("target_string", TOKENIZER_ROUND_TRIP_STRINGS)
    def test_string_roundtrip(self, model_name: str, target_string: str):
        try:
            self.base_string_roundtrip(model_name, target_string)
        except OSError:
            pytest.skip("HuggingFace raises OSError if user is not authenticated.")
