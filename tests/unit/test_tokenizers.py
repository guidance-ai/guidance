import pytest
import tiktoken

from guidance import models

from tests.constants import TOKENIZER_ROUND_TRIP_STRINGS

# These are not _strictly_ unit tests, since they refer
# to specific tokenisers. However, tokenisers are small,
# so if the tokeniser can be loaded separately from the
# model (as is the case for transformers and tiktoken),
# then this is a good place to have them live.

# The LlamaCpp tokenisers have tests in test_llamacpp.py
# since those tokenisers cannot be loaded separately from
# their models.


class TestTransformerTokenizers:
    TRANSFORMER_MODELS = [
        "gpt2",
        "microsoft/phi-2",
        "microsoft/Phi-3-small-8k-instruct",
        "microsoft/Phi-3-mini-4k-instruct",
        "meta-llama/Meta-Llama-3-8B-Instruct",
    ]

    @pytest.mark.parametrize(
        "model_name",
        TRANSFORMER_MODELS,
    )
    def test_smoke(self, model_name: str):
        my_tok = models.TransformersTokenizer(
            model=model_name, transformers_tokenizer=None, trust_remote_code=True
        )
        assert my_tok is not None

    @pytest.mark.parametrize("model_name", TRANSFORMER_MODELS)
    @pytest.mark.parametrize("target_string", TOKENIZER_ROUND_TRIP_STRINGS)
    def test_string_roundtrip(self, model_name: str, target_string: str):
        my_tok = models.TransformersTokenizer(
            model=model_name, transformers_tokenizer=None, trust_remote_code=True
        )

        encoded = my_tok.encode(target_string.encode())
        decoded = my_tok.decode(encoded)
        final_string = decoded.decode()

        assert final_string == target_string


class TestTiktoken:
    MODELS = ["gpt-3.5-turbo", "gpt-4"]

    @pytest.mark.parametrize("model_name", MODELS)
    @pytest.mark.parametrize("target_string", TOKENIZER_ROUND_TRIP_STRINGS)
    def test_string_roundtrip(self, model_name: str, target_string: str):
        my_tik = tiktoken.encoding_for_model(model_name)
        my_tok = models._grammarless.GrammarlessTokenizer(my_tik)

        encoded = my_tok.encode(target_string.encode())
        decoded = my_tok.decode(encoded)
        final_string = decoded.decode()

        assert final_string == target_string
