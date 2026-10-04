import re

import pytest

from guidance import gen, special_token
from guidance.models import Mock


def test_special_token():
    s = special_token("<s>")
    inner = gen(regex=r"\d{5,5}")
    outer = s + inner + s

    lm = Mock() + outer
    assert re.match(r"<s>\d{5,5}<s>", str(lm))


@pytest.mark.parametrize("token", ["<s>", "<think>", "</think>", "<|im_end|>", "<结束>"])
def test_special_token_preserves_complete_token(token):
    assert special_token(token).format() == token


@pytest.mark.parametrize("token", ["<s>text", "<s>\n", "<s>\r\n", "<s> ", "<s><s>", "<s>>"])
def test_special_token_rejects_trailing_content(token):
    with pytest.raises(ValueError, match="Only special tokens"):
        special_token(token)


@pytest.mark.parametrize("token", ["", "s", "<>", "<s", "s>", "prefix<s>", "<<s>>"])
def test_special_token_rejects_invalid_format(token):
    with pytest.raises(ValueError, match="Only special tokens"):
        special_token(token)
