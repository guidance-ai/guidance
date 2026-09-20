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

    lm = m + substring(target_string, chunk="character", name="result")
    assert lm["result"] == expected_string


@pytest.mark.parametrize(
    "chunk_function",
    [list, tuple, lambda s: iter(s), lambda s: (c for c in s)],
    ids=["list", "tuple", "iterator", "generator"],
)
def test_substring_chunk_function_returns_any_iterable(chunk_function):
    # `chunk` is typed as Callable[[str], Iterable[str]], so one-shot iterables are valid returns
    target_string = "long string"

    grammar = substring(target_string, chunk=chunk_function)
    assert "".join(grammar.value.chunks) == target_string

    m = models.Mock(f"<s>{target_string}<s>")
    lm = m + substring(target_string, chunk=chunk_function, name="result")
    assert lm["result"] == target_string
