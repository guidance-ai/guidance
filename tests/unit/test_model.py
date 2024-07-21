import guidance
from guidance import gen, models

import logging
logging.basicConfig(level=logging.DEBUG)

def test_call_embeddings():
    """This tests calls embedded in strings."""
    model = models.Mock()

    @guidance(dedent=False)
    def bla(lm, bla):
        lm += bla + "ae" + gen(max_tokens=10)
        return lm

    @guidance(dedent=False)
    def ble(lm):
        lm += f"""
    ae galera! {bla('33')}
    let's do more stuff!!""" + gen(
            max_tokens=10
        )
        return lm

    assert "{{G|" not in str(model + ble())

# def test_dedent():
#     """Test that dedent functionality in f-strings works across Python versions."""
#     @guidance(stateless=True, dedent=True)
#     def character_maker(lm):
#         lm += f"""\
#         {{
#             "name": "{1+1}",
#         }}"""
#         return lm

#     lm = guidance.models.Mock()
#     lm += character_maker()
#     assert str(lm).startswith("{")



def test_dedent():
    """Test that dedent functionality in f-strings works across Python versions."""
    @guidance(stateless=True, dedent=True)
    def character_maker(lm):
        lm += f"""\
        {{
            "name": "{1+1}",
            "age": "{gen('name', stop='"', max_tokens=1)}",
        }}"""
        return lm

    lm = guidance.models.Mock()
    result = lm + character_maker()
    # logging.debug(f"Result: {str(result)}")
    assert str(result).startswith("{")