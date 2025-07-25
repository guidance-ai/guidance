import re

from guidance import gen, special_token
from guidance.models import Mock


def test_special_token():
    s = special_token("<s>")
    inner = gen(regex=r"\d{5,5}")
    outer = s + inner + s

    lm = Mock() + outer
    assert re.match(r"<s>\d{5,5}<s>", str(lm))
