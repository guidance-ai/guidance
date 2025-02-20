from guidance import gen, models, substring


def test_substring_equal_unconstrained(selected_model: models.Model):
    target_model = selected_model
    lm = target_model + "ae galera " + gen(max_tokens=10, name="test")
    lm2 = target_model + "ae galera " + substring(lm["test"], name="capture")
    assert lm2["capture"] in lm["test"]
