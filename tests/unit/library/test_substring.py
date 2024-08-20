import pytest

from guidance import models, substring


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
    m = models.Mock(f"<s>{mock_string}<s>")

    lm = m + substring(target_string, name="result")
    assert lm["result"] == expected_string
