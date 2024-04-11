import pytest

from guidance import gen, models, prefix_tree


def test_smoke(selected_model: models.Model):
    KEY = "test_data"
    long_string = "abcdef"
    lm = (
        selected_model
        + "The alphabet begins "
        + prefix_tree(strings=[long_string], name=KEY, partial_matches=False)
    )

    assert lm[KEY] == "1"
