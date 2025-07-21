from guidance import models, select


def test_select_simple(selected_model: models.Model):
    lm = selected_model
    options = ["baad I think", "bad I think", "bad"]
    lm = lm + "Scott is quite " + select(name="bad", options=options)
    assert lm["bad"] in options


def test_select_large(selected_model: models.Model):
    # From issue #1320
    lm = selected_model
    n = 2002

    lm += "Pick a number"
    lm += select(list(range(n)), name="response")

    result = int(lm["response"])
    assert 0 <= result < n
