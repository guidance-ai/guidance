import pytest

from guidance import models

from tests.constants import TOKENIZER_ROUND_TRIP_STRINGS


class TestLlamaCppTokenizers:
    def test_smoke(self, llamacpp_model: models.LlamaCpp):
        my_tok = llamacpp_model.engine.tokenizer
        assert my_tok is not None

    @pytest.mark.parametrize("target_string", TOKENIZER_ROUND_TRIP_STRINGS)
    def test_string_roundtrip(self, llamacpp_model: models.LlamaCpp, target_string: str):
        my_tok = llamacpp_model.engine.tokenizer

        encoded = my_tok.encode(target_string.encode())
        decoded = my_tok.decode(encoded)
        final_string = decoded.decode()

        assert final_string == target_string
