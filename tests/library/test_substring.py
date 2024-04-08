from guidance import gen, substring


def test_substring_equal_unconstrained(selected_model):
    target_model = selected_model
    lm = target_model + "ae galera " + gen(max_tokens=10, name="test")
    lm2 = target_model + "ae galera " + substring(lm["test"])
    assert str(lm) == str(lm2)
