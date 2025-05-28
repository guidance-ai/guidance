from guidance import models

TOKENIZER_ROUND_TRIP_STRINGS = [
    "",
    " ",
    "hello",
    " hello",
    "two words",
    " two words",
    " two words ",
    "two words ",
    "’",
    "’•¶∂ƒ˙∆£Ħ爨ൠᅘ∰፨",
]


class BaseTestTransformerTokenizers:
    def base_smoke(self, model_name: str):
        my_tok = models.TransformersTokenizer.from_pretrained(
            model_name, trust_remote_code=True,
        )
        assert my_tok is not None

    def base_string_roundtrip(self, model_name: str, target_string: str):
        my_tok = models.TransformersTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )

        encoded = my_tok.encode(target_string.encode())
        decoded = my_tok.decode(encoded)
        final_string = decoded.decode()

        assert final_string == target_string

    def base_eos_bos_token_round_trip(
        self, model_name: str
    ):
        my_tok = models.TransformersTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )

        assert my_tok.eos_token == my_tok.decode([my_tok.eos_token_id])
        assert my_tok.encode(my_tok.eos_token) == [my_tok.eos_token_id]

        if my_tok.bos_token is not None:
            assert my_tok.bos_token == my_tok.decode([my_tok.bos_token_id])
            assert my_tok.encode(my_tok.bos_token) == [my_tok.bos_token_id]
