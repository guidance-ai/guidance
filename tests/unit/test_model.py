import guidance

from guidance import gen, models

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
