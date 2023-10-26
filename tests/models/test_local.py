import guidance
from guidance import zero_or_more, byte_range
from ..utils import get_model

def test_token_healing():
    '''Tests a bug where the space is incorrectly forced as token 220, while it should be not forced it might be extended'''
    gpt2 = get_model("transformers:gpt2")
    lm = gpt2 + ("This is a story of 10 or 5 or " + zero_or_more(byte_range(b'0', b'9')))
    assert len(lm) > len("This is a story of 10 or 5 or ")