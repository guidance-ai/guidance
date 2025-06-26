import pytest

from guidance._grammar import select
from guidance.library import at_most_n_repeats, exactly_n_repeats, sequence

from ...utils import check_match_failure, check_match_success_with_guards


class TestExactlynRepeats:
    def test_smoke(self):
        grammar = "AAA" + exactly_n_repeats("b", 4) + "BBB"

        matched = grammar.match(b"AAAbbbbBBB", raise_exceptions=True)
        assert matched is not None

    @pytest.mark.parametrize("test_string", ["aa", "ab", "ba", "bb"])
    def test_with_select(self, test_string):
        grammar = exactly_n_repeats(select(["a", "b"]), 2)
        check_match_success_with_guards(grammar, test_string)

    @pytest.mark.parametrize(
        ["bad_string", "good_bytes", "failure_byte", "allowed_bytes"],
        [
            ("bbb", b"bbb", b"B", {b"b"}),
            ("bbbbb", b"bbbb", b"b", {b"B"}),
            ("aaaa", b"", b"a", {b"b"}),
        ],
    )
    def test_bad_repeats(self, bad_string: str, good_bytes, failure_byte, allowed_bytes):
        PREFIX = "AAA"
        SUFFIX = "BBB"
        grammar = PREFIX + exactly_n_repeats("b", 4) + SUFFIX
        check_match_failure(
            bad_string=PREFIX + bad_string + SUFFIX,
            good_bytes=PREFIX.encode() + good_bytes,
            failure_byte=failure_byte,
            allowed_bytes=allowed_bytes,
            grammar=grammar,
        )


class TestAtMostnRepeats:
    def test_smoke(self):
        grammar = "AAA" + at_most_n_repeats("a", 3) + "BBB"

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
            ("bbbbb", b"bbbb", b"b", {b"B"}),
            ("aaaa", b"", b"a", {b"b", b"B"}),
        ],
    )
    def test_bad_repeats(self, bad_string: str, good_bytes, failure_byte, allowed_bytes):
        PREFIX = "AAA"
        SUFFIX = "BBB"
        grammar = PREFIX + at_most_n_repeats("b", 4) + SUFFIX
        check_match_failure(
            bad_string=PREFIX + bad_string + SUFFIX,
            good_bytes=PREFIX.encode() + good_bytes,
            failure_byte=failure_byte,
            allowed_bytes=allowed_bytes,
            grammar=grammar,
        )


class TestSequence:
    def test_smoke(self):
        grammar = "AAA" + sequence("a") + "BBB"

        matched = grammar.match("AAAaaaaaaaBBB".encode(), raise_exceptions=True)
        assert matched is not None

    @pytest.mark.parametrize(
        "test_string",
        [
            "",
            "a",
            "b",
            "ab",
            "ba",
            "aaa",
            "aaaaaaaaaaaaaaa",
            "bbbbbbbbbbb",
            "abababab",
            "aaaaabbbbbbbbb",
        ],
    )
    def test_unconstrained(self, test_string: str):
        grammar = sequence(select(["a", "b"]))
        check_match_success_with_guards(grammar, test_string)

    @pytest.mark.parametrize(
        ["bad_string", "good_bytes", "failure_byte", "allowed_bytes"],
        [
            ("ba", b"b", b"a", {b"b", b"B"}),
            ("a", b"", b"a", {b"b", b"B"}),
        ],
    )
    def test_bad_repeats_unconstrained(self, bad_string: str, good_bytes, failure_byte, allowed_bytes):
        PREFIX = "AAA"
        SUFFIX = "BBB"
        grammar = PREFIX + sequence("b") + SUFFIX
        check_match_failure(
            bad_string=PREFIX + bad_string + SUFFIX,
            good_bytes=PREFIX.encode() + good_bytes,
            failure_byte=failure_byte,
            allowed_bytes=allowed_bytes,
            grammar=grammar,
        )

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
            ("bbb", b"bbb", b"B", {b"b"}),
            ("aaaa", b"", b"a", {b"b"}),
            ("bbbba", b"bbbb", b"a", {b"b", b"B"}),
            ("bbbbbba", b"bbbbbb", b"a", {b"b", b"B"}),
        ],
    )
    def test_bad_repeats_min_length(self, bad_string: str, good_bytes, failure_byte, allowed_bytes):
        PREFIX = "AAA"
        SUFFIX = "BBB"
        grammar = PREFIX + sequence("b", min_length=4) + SUFFIX
        check_match_failure(
            bad_string=PREFIX + bad_string + SUFFIX,
            good_bytes=PREFIX.encode() + good_bytes,
            failure_byte=failure_byte,
            allowed_bytes=allowed_bytes,
            grammar=grammar,
        )

    @pytest.mark.parametrize("n_repeats", range(3))
    def test_max_length(self, n_repeats):
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
            ("bbb", b"bb", b"b", {b"B"}),
            ("aa", b"", b"a", {b"b", b"B"}),
        ],
    )
    def test_bad_repeats_max_length(self, bad_string: str, good_bytes, failure_byte, allowed_bytes):
        PREFIX = "AAA"
        SUFFIX = "BBB"
        grammar = PREFIX + sequence("b", max_length=2) + SUFFIX
        check_match_failure(
            bad_string=PREFIX + bad_string + SUFFIX,
            good_bytes=PREFIX.encode() + good_bytes,
            failure_byte=failure_byte,
            allowed_bytes=allowed_bytes,
            grammar=grammar,
        )

    @pytest.mark.parametrize("n_repeats", range(1, 4))
    def test_min_max_length(self, n_repeats):
        grammar = sequence("a", min_length=1, max_length=3)
        test_string = "a" * n_repeats
        assert len(test_string) == n_repeats
        check_match_success_with_guards(grammar, test_string)

    @pytest.mark.parametrize("n_repeats", range(0, 4))
    def test_min_max_length_zero(self, n_repeats):
        grammar = sequence("a", min_length=0, max_length=3)
        test_string = "a" * n_repeats
        assert len(test_string) == n_repeats
        check_match_success_with_guards(grammar, test_string)

    @pytest.mark.parametrize("c0", ["a", "b"])
    @pytest.mark.parametrize("c1", ["", "a", "b"])
    @pytest.mark.parametrize("c2", ["", "a", "b"])
    def test_min_max_length_select(self, c0, c1, c2):
        test_string = c0 + c1 + c2
        grammar = sequence(select(["a", "b"]), min_length=1, max_length=3)
        assert len(test_string) >= 1 and len(test_string) <= 3
        check_match_success_with_guards(grammar, test_string)

    @pytest.mark.parametrize("c0", ["", "a", "b"])
    @pytest.mark.parametrize("c1", ["", "a", "b"])
    @pytest.mark.parametrize("c2", ["", "a", "b"])
    def test_min_max_length_select_zero(self, c0, c1, c2):
        test_string = c0 + c1 + c2
        grammar = sequence(select(["a", "b"]), min_length=0, max_length=3)
        assert len(test_string) >= 0 and len(test_string) <= 3
        check_match_success_with_guards(grammar, test_string)

    def test_min_max_length_equal_zero(self):
        # Call this out as a special case of next
        grammar = sequence("a", min_length=0, max_length=0)
        check_match_success_with_guards(grammar, "")

    @pytest.mark.parametrize("n_repeats", range(1, 4))
    def test_min_max_length_equal(self, n_repeats):
        grammar = sequence("a", min_length=n_repeats, max_length=n_repeats)
        test_string = "a" * n_repeats
        assert len(test_string) == n_repeats
        check_match_success_with_guards(grammar, test_string)

    @pytest.mark.parametrize(
        ["bad_string", "good_bytes", "failure_byte", "allowed_bytes"],
        [
            ("", b"", b"B", {b"b"}),
            ("bbb", b"bb", b"b", {b"B"}),
            ("aa", b"", b"a", {b"b"}),
            ("ba", b"b", b"a", {b"b", b"B"}),
            ("bba", b"bb", b"a", {b"B"}),
        ],
    )
    def test_bad_repeats_min_max_length(self, bad_string: str, good_bytes, failure_byte, allowed_bytes):
        PREFIX = "AAA"
        SUFFIX = "BBB"
        grammar = PREFIX + sequence("b", min_length=1, max_length=2) + SUFFIX
        check_match_failure(
            bad_string=PREFIX + bad_string + SUFFIX,
            good_bytes=PREFIX.encode() + good_bytes,
            failure_byte=failure_byte,
            allowed_bytes=allowed_bytes,
            grammar=grammar,
        )
