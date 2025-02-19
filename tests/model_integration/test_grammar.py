from guidance.models import Model, select


def test_select_simple(selected_model: Model.Model):
    lm = selected_model
    options = ["baad I think", "bad I think", "bad"]
    lm = lm + "Scott is quite " + select(name="bad", options=options)
    assert lm["bad"] in options
