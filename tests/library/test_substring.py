import pytest

from guidance import gen, models, substring


def test_substring_equal_unconstrained(selected_model):
    target_model = selected_model
    lm = target_model + "ae galera " + gen(max_tokens=10, name="test")
    lm2 = target_model + "ae galera " + substring(lm["test"], name="capture")
    assert lm2["capture"] in lm["test"]
    assert str(lm) == str(lm2)


@pytest.mark.parametrize(
    ("mock_string", "target_string", "expected_string"),
    [
        ("abc", "abc", "abc"),
        ("ab", "abc", "ab"),
        ("bc", "abc", "bc"),
        ("a", "abc", "a"),
        ("b", "abc", "b"),
        ("c", "abc", "c"),
        ("abc", "def", ""),  # This is a 'failure' case
        (
            "long string",
            "This is long string, only take part of this long string",
            "long string",
        ),
    ],
)
def test_mocked_substring(mock_string, target_string, expected_string):
    m = models.Mock(f"<s>{mock_string}")

    lm = m + substring(target_string, name="result")
    assert lm["result"] == expected_string
