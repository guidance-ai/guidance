import pytest
from functools import partial

from guidance import regex
from guidance._grammar import Byte, ByteRange

from ..utils import check_match_failure, generate_and_check


class TestCharacterClasses:
    @pytest.mark.parametrize(
        "pattern, string, stop_char",
        [
            (r"[abc]+", "cbabbaccabc", chr(7)),
            (r"[a-z]+", "thequickbrownfoxjumpsoverthelazydog", chr(7)),
            (r"[0-9]+", "9876543210", chr(7)),
            (r"[b-y]+", "by", chr(7)),  # range is left and right inclusive
            (r"[a-f0-9]+", "abcdef0123456789", chr(7)),
            (r"[abcA-Z]+", "abcABCXYZ", chr(7)),
            (r"[a-z\d]+", "abc123", chr(7)),
            (r"[^abc]+", "ABCxyz8743-!@#$%^&*()_+", "a"),
            (r"[^\d]+", "abcXYZ-!@#$%^&*()_+", "8"),
            (r"[^B-Z]+", "qwertyA", "B"),
            (r"[^a-z\d]+", "ABCDEF-!@#$%^&*()_+", "a"),
            (r"[^\n]+", "ABCxyz8743-!@#$%^&*()_+", "\n"),
        ],
    )
    def test_good(self, pattern, string, stop_char):
        grammar_callable = partial(regex, pattern=pattern)
        generate_and_check(grammar_callable, string, stop_char=stop_char)

    @pytest.mark.parametrize(
        "pattern, string, good_bytes, failure_byte, allowed_bytes",
        [
            (
                r"[abc]+",
                "cbabbaccabcx",
                b"cbabbaccabc",
                b"x",
                {Byte(b"a"), Byte(b"b"), Byte(b"c")},
            ),
            (
                r"[a-z]+",
                "thequickbrownfoxjumpsoverthelazydogX",
                b"thequickbrownfoxjumpsoverthelazydog",
                b"X",
                {ByteRange((b"az"))},
            ),
            (
                r"[0-9]+",
                "9876543210x",
                b"9876543210",
                b"x",
                {ByteRange((b"09"))},
            ),
            (
                r"[b-y]+",
                "bya",
                b"by",
                b"a",
                {ByteRange(b"by")},
            ),  # range doesn't overflow left
            (
                r"[b-y]+",
                "byz",
                b"by",
                b"z",
                {ByteRange(b"by")},
            ),  # range doesn't overflow right
            (
                r"[a-f0-9]+",
                "abcdef0123456789x",
                b"abcdef0123456789",
                b"x",
                {ByteRange(b"af"), ByteRange(b"09")},
            ),
            (
                r"[abcA-Z]+",
                "abcABCXYZx",
                b"abcABCXYZ",
                b"x",
                {Byte(b"a"), Byte(b"b"), Byte(b"c"), ByteRange(b"AZ")},
            ),
            (
                r"[a-z\d]+",
                "abc123@",
                b"abc123",
                b"@",
                {ByteRange(b"az"), ByteRange(b"09")},
            ),
            (
                r"[^abc]+",
                "ABCxyz8743-!@#$%^&*()_+a",
                b"ABCxyz8743-!@#$%^&*()_+",
                b"a",
                {ByteRange(b"\x00`"), ByteRange(b"d\x7f")},
            ),
            (
                r"[^\d]+",
                "abcXYZ-!@#$%^&*()_+6",
                b"abcXYZ-!@#$%^&*()_+",
                b"6",
                {ByteRange(b"\x00/"), ByteRange(b":\x7f")},
            ),
            (
                r"[^B-Z]+",
                "qwertyAB",
                b"qwertyA",
                b"B",
                {ByteRange(b"\x00A"), ByteRange(b"[\x7f")},
            ),
            (
                r"[^a-z\d]+",
                "ABCDEF-!@#$%^&*()_+x",
                b"ABCDEF-!@#$%^&*()_+",
                b"x",
                {ByteRange(b"\x00/"), ByteRange(b":`"), ByteRange(b"{\x7f")},
            ),
            (
                r"[^\n]+",
                "ABCxyz8743-!@#$%^&*()_+\n",
                b"ABCxyz8743-!@#$%^&*()_+",
                b"\n",
                {ByteRange(b"\x00\t"), ByteRange(b"\x0b\x7f")},
            ),
        ],
    )
    def test_bad(self, pattern, string, good_bytes, failure_byte, allowed_bytes):
        check_match_failure(
            bad_string=string,
            good_bytes=good_bytes,
            failure_byte=failure_byte,
            allowed_bytes=allowed_bytes,
            grammar=regex(pattern),
        )


class TestQuantifiers:
    @pytest.mark.parametrize(
        "pattern, string",
        [
            (r"a*b", "b"),
            (r"a*b", "ab"),
            (r"a*b", "aaaaab"),
            (r"a?b", "b"),
            (r"a?b", "ab"),
            (r"a+b", "ab"),
            (r"a+b", "aaaaab"),
        ],
    )
    def test_kleene_star_optional_one_or_more(self, pattern, string):
        grammar_callable = partial(regex, pattern=pattern)
        generate_and_check(grammar_callable, string)

    @pytest.mark.parametrize(
        "pattern, string",
        [
            (r"a{3}", "aaa"),
            (r"(ab){2}", "abab"),
        ],
    )
    def test_exact_repetitions(self, pattern, string):
        grammar_callable = partial(regex, pattern=pattern)
        generate_and_check(grammar_callable, string)

    @pytest.mark.parametrize(
        "pattern, string",
        [
            (r"(a{2,3}b){2}", "aabaaab"),
            (r"(a{2,3}){2}", "aaaaaa"),
            (r"a{2}b{2,3}", "aabbb"),
            (r"(ab){1,3}c{2,4}", "ababccc"),
            (r"(ab){1,3}c{2,4}", "abababcccc"),
        ],
    )
    def test_nested_quantifiers(self, pattern, string):
        grammar_callable = partial(regex, pattern=pattern)
        generate_and_check(grammar_callable, string)

    @pytest.mark.parametrize(
        "pattern, string, good_bytes, failure_byte, allowed_bytes",
        [
            (
                r"a*b",
                "axb",
                b"a",
                b"x",
                {Byte(b"a"), Byte(b"b")},
            ),  # 'x' disrupts the match
            (
                r"a+b",
                "b",
                b"",
                b"b",
                {Byte(b"a")},
            ),  # 'a+' requires at least one 'a' before 'b'
            (
                r"a?b",
                "x",
                b"",
                b"x",
                {Byte(b"a"), Byte(b"b")},
            ),  # 'a?' requires zero or one 'a' before 'b'
            (
                r"a?b",
                "axb",
                b"a",
                b"x",
                {Byte(b"b")},
            ),  # 'x' disrupts the match
            (
                r"a?b",
                "aab",
                b"a",
                b"a",
                {Byte(b"b")},
            ),  # Second 'a' is too many
            (
                r"(xyz)?abc",
                "xyabc",
                b"xy",
                b"a",
                {Byte(b"z")},
            ),  # Expected 'z'
            (
                r"(xyz)?abc",
                "abcx",
                b"abc",
                b"x",
                set(),
            ),  # Extra character after 'abc'
            (
                r"a{2,4}",
                "aaaaa",
                b"aaaa",
                b"a",
                set(),
            ),  # Too many 'a's
            (
                r"(ab){2,3}",
                "abababab",
                b"ababab",
                b"a",
                set(),
            ),  # 'ab' four times, more than the maximum
            (
                r"a{3,5}b",
                "aab",
                b"aa",
                b"b",
                {Byte(b"a")},
            ),  # Less than the minimum 'a's before 'b'
            (
                r"a{3,5}b",
                "aaaaaab",
                b"aaaaa",
                b"a",
                {Byte(b"b")},
            ),  # More than the maximum 'a's before 'b'
        ],
    )
    def test_quantifiers_failure(
        self, pattern, string, good_bytes, failure_byte, allowed_bytes
    ):
        check_match_failure(
            bad_string=string,
            good_bytes=good_bytes,
            failure_byte=failure_byte,
            allowed_bytes=allowed_bytes,
            grammar=regex(pattern),
        )


class TestAlternations:
    @pytest.mark.parametrize(
        "pattern, string",
        [
            (r"a|b", "a"),
            (r"a|b", "b"),
            (r"cat|dog", "cat"),
            (r"cat|dog", "dog"),
        ],
    )
    def test_simple_alternations(self, pattern, string):
        grammar_callable = partial(regex, pattern=pattern)
        generate_and_check(grammar_callable, string)

    @pytest.mark.parametrize(
        "pattern, string",
        [
            (r"apple|orange", "apple"),
            (r"apple|orange", "orange"),
            (r"100|200", "100"),
            (r"100|200", "200"),
        ],
    )
    def test_alternations_multiple_characters(self, pattern, string):
        grammar_callable = partial(regex, pattern=pattern)
        generate_and_check(grammar_callable, string)

    @pytest.mark.parametrize(
        "pattern, string",
        [
            (r"(a|b)c|d", "ac"),
            (r"(a|b)c|d", "bc"),
            (r"(a|b)c|d", "d"),
        ],
    )
    def test_nested_alternations(self, pattern, string):
        grammar_callable = partial(regex, pattern=pattern)
        generate_and_check(grammar_callable, string)

    @pytest.mark.parametrize(
        "pattern, string",
        [
            (r"(a|b)+", "aaa"),
            (r"(a|b)+", "bbb"),
            (r"(a|b)+", "abab"),
            (r"(dog|cat)s?", "dog"),
            (r"(dog|cat)s?", "cats"),
        ],
    )
    def test_alternations_with_quantifiers(self, pattern, string):
        grammar_callable = partial(regex, pattern=pattern)
        generate_and_check(grammar_callable, string)

    @pytest.mark.parametrize(
        "pattern, string, good_bytes, failure_byte, allowed_bytes",
        [
            (
                r"a|b",
                "c",
                b"",
                b"c",
                {Byte(b"a"), Byte(b"b")},
            ),  # Neither 'a' nor 'b'
            (
                r"apple|orange",
                "banana",
                b"",
                b"b",
                {Byte(b"a"), Byte(b"o")},
            ),  # Neither 'apple' nor 'orange'
            (
                r"100|200",
                "300",
                b"",
                b"3",
                {Byte(b"1"), Byte(b"2")},
            ),  # Neither '100' nor '200'
            (
                r"(a|b)c|d",
                "ae",
                b"a",
                b"e",
                {Byte(b"c"), Byte(b"c")},
            ),  # Neither 'ac' nor 'bc' nor 'd'
            (
                r"(a|b)+",
                "abbaabbabc",
                b"abbaabbab",
                b"c",
                {Byte(b"a"), Byte(b"b")},
            ),  # 'c' does not match pattern '(a|b)+'
            (
                r"cat|dog",
                "car",
                b"ca",
                b"r",
                {Byte(b"t")},
            ),  # 't' should be forced
            (
                r"(dog|cat)s?",
                "cars",
                b"ca",
                b"r",
                {Byte(b"t")},
            ),  # 't' should be forced
        ],
    )
    def test_alternations_failures(
        self, pattern, string, good_bytes, failure_byte, allowed_bytes
    ):
        check_match_failure(
            bad_string=string,
            good_bytes=good_bytes,
            failure_byte=failure_byte,
            allowed_bytes=allowed_bytes,
            grammar=regex(pattern),
        )


class TestDot:
    @pytest.mark.parametrize(
        "pattern, string",
        [
            (r".+", "ABCxyz8743-!@#$%^&*()_+ \t"),
        ],
    )
    def test_dot(self, pattern, string):
        grammar_callable = partial(regex, pattern=pattern)
        generate_and_check(grammar_callable, string, stop_char="\n")

    @pytest.mark.parametrize(
        "pattern, string, good_bytes, failure_byte, allowed_bytes",
        [
            (
                r".+",
                "ABCxyz8743-!@#$%^&*()_+\n",
                b"ABCxyz8743-!@#$%^&*()_+",
                b"\n",
                {ByteRange(b"\x00\t"), ByteRange(b"\x0b\x7f")},
            ),
        ],
    )
    def test_dot_failures(
        self, pattern, string, good_bytes, failure_byte, allowed_bytes
    ):
        check_match_failure(
            bad_string=string,
            good_bytes=good_bytes,
            failure_byte=failure_byte,
            allowed_bytes=allowed_bytes,
            grammar=regex(pattern),
        )


class TestSpecialCharacters:
    @pytest.mark.parametrize(
        "pattern, string, stop_char",
        [
            (r"\d+", "1234567890", chr(7)),
            (r"[^\D]+", "1234567890", chr(7)),
            (r"\D+", "ABCxyz-!@#$%^&*()_+", "9"),
            (r"[^\d]+", "ABCxyz-!@#$%^&*()_+", "9"),
            (r"\w+", "abcABC123_", chr(7)),
            (r"[^\W]+", "abcABC123_", chr(7)),
            (r"\W+", " -!@#$%^&*()+", "9"),
            (r"[^\w]+", "-!@#$%^&*()+", "9"),
            (r"\s+", " \t\n\r\f\v", chr(7)),
            (r"[^\S]+", " \t\n\r\f\v", chr(7)),
            (r"\S+", "ABCxyz8743-!@#$%^&*()_+", " "),
            (r"[^\s]+", "ABCxyz8743-!@#$%^&*()_+", " "),
        ],
    )
    def test_good(self, pattern, string, stop_char):
        grammar_callable = partial(regex, pattern=pattern)
        generate_and_check(grammar_callable, string, stop_char=stop_char)

    @pytest.mark.parametrize(
        "pattern, string, good_bytes, failure_byte, allowed_bytes",
        [
            (
                r"\d+",
                "0123456789x",
                b"0123456789",
                b"x",
                {ByteRange(b"09")},
            ),
            (
                r"\D+",
                "ABCxyz-!@#$%^&*()_+1",
                b"ABCxyz-!@#$%^&*()_+",
                b"1",
                {ByteRange(b"\x00/"), ByteRange(b":\x7f")},
            ),
            (
                r"\w+",
                "abcABC123_@",
                b"abcABC123_",
                b"@",
                {ByteRange(b"az"), ByteRange(b"AZ"), ByteRange(b"09"), Byte(b"_")},
            ),
            (
                r"\W+",
                " -!@#$%^&*()+a",
                b" -!@#$%^&*()+",
                b"a",
                {
                    ByteRange(b"\x00/"),
                    ByteRange(b":@"),
                    ByteRange(b"[^"),
                    Byte(b"`"),
                    ByteRange(b"{\x7f"),
                },
            ),
            (
                r"\s+",
                " \t\n\r\f\v8",
                b" \t\n\r\f\v",
                b"8",
                {
                    Byte(b" "),
                    Byte(b"\t"),
                    Byte(b"\n"),
                    Byte(b"\r"),
                    Byte(b"\f"),
                    Byte(b"\v"),
                },
            ),
            (
                r"\S+",
                "abcABC123_ ",
                b"abcABC123_",
                b" ",
                {ByteRange(b"\x00\x08"), ByteRange(b"\x0e\x1f"), ByteRange(b"!\x7f")},
            ),
        ],
    )
    def test_bad(self, pattern, string, good_bytes, failure_byte, allowed_bytes):
        check_match_failure(
            bad_string=string,
            good_bytes=good_bytes,
            failure_byte=failure_byte,
            allowed_bytes=allowed_bytes,
            grammar=regex(pattern),
        )
