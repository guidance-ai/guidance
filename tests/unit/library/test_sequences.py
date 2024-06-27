import pytest

from guidance._grammar import Join, select, Byte
from guidance.library import sequence, exactly_n_repeats, at_most_n_repeats


from ...utils import check_match_failure, check_match_success_with_guards


class TestExactlynRepeats:
    def test_smoke(self):
        grammar = Join(["AAA", exactly_n_repeats("b", 4), "BBB"])

        matched = grammar.match(b"AAAbbbbBBB", raise_exceptions=True)
        assert matched is not None

    @pytest.mark.parametrize("test_string", ["aa", "ab", "ba", "bb"])
    def test_with_select(self, test_string):
        grammar = exactly_n_repeats(select(["a", "b"]), 2)
        check_match_success_with_guards(grammar, test_string)

    @pytest.mark.parametrize(
        ["bad_string", "good_bytes", "failure_byte", "allowed_bytes"],
        [
            ("bbb", b"bbb", b"B", set([Byte(b"b")])),
            ("bbbbb", b"bbbb", b"b", set([Byte(b"B")])),
            ("aaaa", b"", b"a", set([Byte(b"b")])),
        ],
    )
    def test_bad_repeats(self, bad_string: str, good_bytes, failure_byte, allowed_bytes):
        PREFIX = "AAA"
        SUFFIX = "BBB"
        grammar = Join([PREFIX, exactly_n_repeats("b", 4), SUFFIX])
        check_match_failure(
            PREFIX + bad_string + SUFFIX,
            PREFIX.encode() + good_bytes,
            failure_byte,
            allowed_bytes,
            grammar=grammar,
        )


class TestAtMostnRepeats:
    def test_smoke(self):
        grammar = Join(["AAA", at_most_n_repeats("a", 3), "BBB"])

        matched = grammar.match("AAAaBBB".encode(), raise_exceptions=True)
        assert matched is not None

    @pytest.mark.parametrize("n_repeats", range(3))
    def test_check_repeats(self, n_repeats: int):
        grammar = at_most_n_repeats("b", 2)
        check_match_success_with_guards(grammar, "b" * n_repeats)

    @pytest.mark.parametrize("test_string", ["", "a", "b", "aa", "ab", "ba", "bb"])
    def test_with_select(self, test_string):
        grammar = at_most_n_repeats(select(["a", "b"]), 2)
        check_match_success_with_guards(grammar, test_string)

    @pytest.mark.parametrize(
        ["bad_string", "good_bytes", "failure_byte", "allowed_bytes"],
        [
            ("bbbbb", b"bbbb", b"b", set([Byte(b"B")])),
            ("aaaa", b"", b"a", set([Byte(b"b"), Byte(b"B")])),
        ],
    )
    def test_bad_repeats(self, bad_string: str, good_bytes, failure_byte, allowed_bytes):
        PREFIX = "AAA"
        SUFFIX = "BBB"
        grammar = Join([PREFIX, at_most_n_repeats("b", 4), SUFFIX])
        check_match_failure(
            PREFIX + bad_string + SUFFIX,
            PREFIX.encode() + good_bytes,
            failure_byte,
            allowed_bytes,
            grammar=grammar,
        )
