import pytest
from tests.tokenizer_common import TOKENIZER_ROUND_TRIP_STRINGS, BaseTestTransformerTokenizers

# These are not _strictly_ unit tests, since they refer
# to specific tokenisers. However, tokenisers are small,
# so if the tokeniser can be loaded separately from the
# model, then this is a good place to have them live.

# The LlamaCpp tokenisers have tests in test_llamacpp.py
# since those tokenisers cannot be loaded separately from
# their models.

# The transformer tests have an authenticated version under
# need_credentials


class TestUnauthenticatedTransformerTokenizers(BaseTestTransformerTokenizers):
    TRANSFORMER_MODELS = [
        "gpt2",
        "microsoft/phi-2",
        "microsoft/Phi-3-small-8k-instruct",
        "microsoft/Phi-3-mini-4k-instruct",
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
