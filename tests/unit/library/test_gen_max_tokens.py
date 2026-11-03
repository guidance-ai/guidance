import pytest

from guidance import gen, models


@pytest.mark.parametrize("max_tokens", [0, -1, -5, 2.5, "3"])
def test_gen_rejects_invalid_max_tokens(max_tokens):
    """max_tokens must fail fast with a clear error instead of reaching the model."""
    lm = models.Mock()

    with pytest.raises(ValueError, match="max_tokens must be a positive integer"):
        lm += "Count: " + gen("text", max_tokens=max_tokens)


def test_gen_still_accepts_valid_max_tokens():
    lm = models.Mock()
    lm += "Count: " + gen("text", max_tokens=3)

    assert len(lm["text"]) > 0
