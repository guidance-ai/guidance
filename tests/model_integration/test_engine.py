import numpy as np

from guidance import models
from guidance.models._engine import Engine, LogitsOutput
from guidance.models._transformers import TransformersTokenizer


def test_chat_completion(selected_model: models.Model):
    engine = selected_model._interpreter.engine

    grammar = """%llguidance {}

start: "I choose the letter " myletter
myletter[capture="value"]: LETTER_CHOICE
LETTER_CHOICE: "A" | "B"
"""

    messages = [{"role": "user", "content": "Give me a capital letter"}]

    result, captures = engine.chat_completion(messages, grammar)

    assert len(captures) == 1
    assert captures["value"] in ["A", "B"]
    if captures["value"] == "A":
        assert result == "I choose the letter A"
    elif captures["value"] == "B":
        assert result == "I choose the letter B"


def test_chat_completion_tokenizer_without_bos_token():
    # Qwen2/Qwen3 tokenizers have no BOS token (bos_token_id is None), so the
    # chat template must render without one
    tokenizer = TransformersTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    assert tokenizer.bos_token_id is None
    assert tokenizer.bos_token is None

    class StubEngine(Engine):
        def get_logits(self, token_ids: list[int], include_all_uncached_tokens: bool = False) -> LogitsOutput:
            return {
                "logits": np.zeros((1, tokenizer._vocab_size)),
                "n_tokens": len(token_ids),
                "n_cached": 0,
            }

    engine = StubEngine(tokenizer)

    grammar = """%llguidance {}

start: "I choose the letter " myletter
myletter[capture="value"]: LETTER_CHOICE
LETTER_CHOICE: "A" | "B"
"""

    messages = [{"role": "user", "content": "Give me a capital letter"}]

    result, captures = engine.chat_completion(messages, grammar)

    assert captures["value"] in ["A", "B"]
    assert result == f"I choose the letter {captures['value']}"
