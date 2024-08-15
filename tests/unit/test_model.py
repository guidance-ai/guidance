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


def test_model_set():
    model = models.Mock()
    model = model.set("num", "4")
    assert "num" in model
    assert model["num"] == "4"
    assert model.log_prob("num") is not None

    model = model.set("list_num", ['1', '2'])
    assert "list_num" in model
    assert model["list_num"] == ['1', '2']
    assert model.log_prob("list_num") is not None

    model += gen("list_num", max_tokens=10, list_append=True)
    assert len(model['list_num']) == 3