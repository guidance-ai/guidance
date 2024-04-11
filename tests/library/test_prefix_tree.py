import pytest

from guidance import gen, models, prefix_tree


def test_smoke(selected_model: models.Model):
    KEY = "test_data"
    lm = (
        selected_model
        + "abc def\ndef ghi\nabc def\ndef "
        + prefix_tree(
            strings=["abc", "ghij"],
            name=KEY,
            partial_matches=False,
        )
    )
    print(
        selected_model
        + "abc def\ndef ghi\nabc def\ndef "
        + gen(max_tokens=3, stop="\n")
    )
    print("Done unconstrained")

    print(str(lm))
    assert lm[KEY] == "ghi"
