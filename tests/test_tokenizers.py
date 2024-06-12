import pytest

from guidance import models


class TestTransformerTokenizers:
    @pytest.mark.parametrize(
        "model_name",
        [
            "gpt2",
            "microsoft/Phi-3-mini-4k-instruct",
            "microsoft/Phi-3-vision-128k-instruct",
        ],
    )
    def test_smoke(self, model_name: str):
        my_tok = models.TransformersTokenizer(model=model_name, tokenizer=None)
        assert my_tok is not None
