import guidance
from guidance import select, models, gen, zero_or_more, byte_range
from ..utils import get_model

def test_fstring():
    lm = get_model("transformers:gpt2")
    lm += f'this is a test {select(["item1", "item2"])}'
    assert str(lm) in ["this is a test item1", "this is a test item2"]

def test_fstring_custom():
    lm = get_model("transformers:gpt2")

    @guidance
    def my_function(lm):
        return lm + f'another {select(["item1", "item2"])}'

    lm += f'this is a test {my_function()}'
    assert str(lm) in ["this is a test another item1", "this is a test another item2"]

def test_token_count():
    lm = get_model("transformers:gpt2")
    lm2 = lm + ' 1 1 1 1 1' + gen(max_tokens=9) + gen(max_tokens=9)
    assert 18 <= lm2.token_count <= 20 # note we allow ourselves to be off by one because it is hard to know when we are continuing vs starting a new token in the parser

def test_call_embeddings():
    '''This tests calls embedded in strings.'''
    model = models.Mock()

    @guidance(dedent=False)
    def bla(lm, bla):
        lm += bla + 'ae' + gen(max_tokens=10)
        return lm
    
    @guidance(dedent=False)
    def ble(lm):
        lm += f'''
    ae galera! {bla('33')}
    let's do more stuff!!''' + gen(max_tokens=10)
        return lm
    
    assert "{{G|" not in str(model + ble())

def test_token_healing():
    '''Tests a bug where the space is incorrectly forced as token 220, while it should be not forced it might be extended'''
    gpt2 = get_model("transformers:gpt2")
    lm = gpt2 + ("This is a story of 10 or 5 or " + zero_or_more(byte_range(b'0', b'9')))
    assert len(lm) > len("This is a story of 10 or 5 or ")