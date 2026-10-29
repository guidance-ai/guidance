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


@pytest.mark.parametrize(
    "token",
    [
        "<s>text",
        "<s>\n",
        "<s>\r\n",
        "<s> ",
        "<s><s>",
        "<s>>",
        "text<s>",
        "s",
        "<s><e>",
    ],
)
def test_special_token_invalid_trailing_or_format(token: str):
    with pytest.raises(ValueError, match="Only special tokens that start and end with '<' and '>'"):
        special_token(token)


@pytest.mark.parametrize(
    "token",
    [
        "<s>",
        "<think>",
        "</think>",
        "<|im_end|>",
    ],
)
def test_special_token_valid(token: str):
    st = special_token(token)
    assert st.format() == token

