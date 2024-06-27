import pytest

from guidance._grammar import Byte, Join, select
from guidance.library import at_most_n_repeats, exactly_n_repeats, sequence

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


class TestSequence:
    def test_smoke(self):
        grammar = Join(["AAA", sequence("a"), "BBB"])

        matched = grammar.match("AAAaaaaaaaBBB".encode(), raise_exceptions=True)
        assert matched is not None

    @pytest.mark.parametrize("test_string", ["", "a", "aaa", "aaaaaaaaaaaaaaa"])
    def test_unconstrained(self, test_string: str):
        grammar = sequence("a")
        check_match_success_with_guards(grammar, test_string)

    @pytest.mark.parametrize("test_string", ["a", "aa", "aaa"])
    def test_min_length(self, test_string):
        grammar = sequence("a", min_length=1)
        check_match_success_with_guards(grammar, test_string)

    @pytest.mark.parametrize("test_string", ["", "a", "a", "aaa"])
    def test_min_length_zero(self, test_string):
        grammar = sequence("a", min_length=0)
        check_match_success_with_guards(grammar, test_string)

    @pytest.mark.parametrize(
        ["bad_string", "good_bytes", "failure_byte", "allowed_bytes"],
        [
            ("bbb", b"bbb", b"B", set([Byte(b"b")])),
            ("aaaa", b"", b"a", set([Byte(b"b")])),
            ("bbbba", b"bbbb", b"a", set([Byte(b"b"), Byte(b"B")])),
            ("bbbbbba", b"bbbbbb", b"a", set([Byte(b"b"), Byte(b"B")])),
        ],
    )
    def test_bad_repeats_min_length(
        self, bad_string: str, good_bytes, failure_byte, allowed_bytes
    ):
        PREFIX = "AAA"
        SUFFIX = "BBB"
        grammar = Join([PREFIX, sequence("b", min_length=4), SUFFIX])
        check_match_failure(
            PREFIX + bad_string + SUFFIX,
            PREFIX.encode() + good_bytes,
            failure_byte,
            allowed_bytes,
            grammar=grammar,
        )

    @pytest.mark.parametrize("n_repeats", range(3))
    def test_ax_length(self, n_repeats):
        grammar = sequence("a", max_length=2)
        test_string = "a" * n_repeats
        assert len(test_string) == n_repeats
        check_match_success_with_guards(grammar, test_string)

    def test_max_length_zero(self):
        grammar = sequence("a", max_length=0)
        check_match_success_with_guards(grammar, "")

    @pytest.mark.parametrize(
        ["bad_string", "good_bytes", "failure_byte", "allowed_bytes"],
        [
            ("bbb", b"bb", b"b", set([Byte(b"B")])),
            ("aa", b"", b"a", set([Byte(b"b"), Byte(b"B")])),
        ],
    )
    def test_bad_repeats_max_length(
        self, bad_string: str, good_bytes, failure_byte, allowed_bytes
    ):
        PREFIX = "AAA"
        SUFFIX = "BBB"
        grammar = Join([PREFIX, sequence("b", max_length=2), SUFFIX])
        check_match_failure(
            PREFIX + bad_string + SUFFIX,
            PREFIX.encode() + good_bytes,
            failure_byte,
            allowed_bytes,
            grammar=grammar,
        )
