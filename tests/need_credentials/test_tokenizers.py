import pytest

from guidance import models

from tests.tokenizer_common import TOKENIZER_ROUND_TRIP_STRINGS, BaseTestTransformerTokenizers

# Since this is under 'need_credentials' we can assume that HF_TOKEN
# will be available when run


class TestAuthenticatedTransformerTokenizers(BaseTestTransformerTokenizers):
    TRANSFORMER_MODELS = [
        "meta-llama/Meta-Llama-3-8B-Instruct",
    ]

    @pytest.mark.parametrize(
        "model_name",
        TRANSFORMER_MODELS,
    )
    def test_smoke(self, model_name: str):
        self.base_smoke(model_name)

    @pytest.mark.parametrize("model_name", TRANSFORMER_MODELS)
    @pytest.mark.parametrize("target_string", TOKENIZER_ROUND_TRIP_STRINGS)
    def test_string_roundtrip(self, model_name: str, target_string: str):
        self.base_string_roundtrip(model_name, target_string)
