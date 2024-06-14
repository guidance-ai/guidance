import pytest

from guidance import models


class TestTransformerTokenizers:
    TRANSFORMER_MODELS = [
        "gpt2",
        "microsoft/Phi-3-mini-4k-instruct",
        "microsoft/Phi-3-vision-128k-instruct",
        "microsoft/phi-2",
    ]

    @pytest.mark.parametrize(
        "model_name",
        TRANSFORMER_MODELS,
    )
    def test_smoke(self, model_name: str):
        my_tok = models.TransformersTokenizer(model=model_name, tokenizer=None)
        assert my_tok is not None

    @pytest.mark.parametrize("model_name", TRANSFORMER_MODELS)
    @pytest.mark.parametrize("target_string", ["hello", "’"])
    def test_string_roundtrip(self, model_name: str, target_string: str):
        my_tok = models.TransformersTokenizer(model=model_name, tokenizer=None)

        encoded = my_tok.encode(target_string.encode())
        decoded = my_tok.decode(encoded)
        final_string = decoded.decode()

        assert final_string == target_string
