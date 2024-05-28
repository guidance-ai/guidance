import pytest
from guidance._parser import ParserException
from guidance import regex


class TestCharacterClasses:
    @pytest.mark.parametrize(
        "pattern, string",
        [
            (r"[abc]", "a"),
            (r"[abc]", "b"),
            (r"[^abc]", "d"),
            (r"[a-z]", "m"),
            (r"[0-9]", "5"),
        ],
    )
    def test_good(self, pattern, string):
        assert regex(pattern).match(string) is not None

    @pytest.mark.parametrize(
        "pattern, string, failure_byte",
        [
            (r"[abc]+", "bx", b"x"),  # Bad character not in [abc]
            (r"[^abc]+", "xb", b"b"),  # Negated but matched 'b'
            (r"[0-9a-f]+", "3bz", b"z"),  # Character outside the range
            (r"[a-z]+", "g1", b"1"),  # Digit where a letter is expected
        ],
    )
    def test_bad(self, pattern, string, failure_byte):
        with pytest.raises(ParserException) as pe:
            regex(pattern).match(string, raise_exceptions=True)
        assert pe.value.current_byte == failure_byte


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
        assert regex(pattern).match(string) is not None

    @pytest.mark.parametrize(
        "pattern, string",
        [
            (r"a{3}", "aaa"),
            (r"(ab){2}", "abab"),
        ],
    )
    def test_exact_repetitions(self, pattern, string):
        assert regex(pattern).match(string) is not None

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
        assert regex(pattern).match(string) is not None

    @pytest.mark.parametrize(
        "pattern, string, failure_byte",
        [
            (r"a*b", "axb", b"x"),  # 'x' disrupts the match
            (r"a+b", "b", b"b"),  # 'a+' requires at least one 'a' before 'b'
            (r"a?b", "axb", b"x"),  # 'x' disrupts the match
            (r"a?b", "aab", b"a"),  # Second 'a' is too many
            (r"(xyz)?abc", "xyabc", b"a"),  # Expected 'z'
            (r"(xyz)?abc", "abcx", b"x"),  # Extra character after 'abc'
            (r"a{2,4}", "aaaaa", b"a"),  # Too many 'a's
            (r"(ab){2,3}", "abababab", b"a"),  # 'ab' four times, more than the maximum
            (r"a{3,5}b", "aab", b"b"),  # Less than the minimum 'a's before 'b'
            (r"a{3,5}b", "aaaaaab", b"a"),  # More than the maximum 'a's before 'b'
        ],
    )
    def test_quantifiers_failure(self, pattern, string, failure_byte):
        with pytest.raises(ParserException) as pe:
            regex(pattern).match(string, raise_exceptions=True)
        assert pe.value.current_byte == failure_byte


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
        assert regex(pattern).match(string) is not None

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
        assert regex(pattern).match(string) is not None

    @pytest.mark.parametrize(
        "pattern, string",
        [
            (r"(a|b)c|d", "ac"),
            (r"(a|b)c|d", "bc"),
            (r"(a|b)c|d", "d"),
        ],
    )
    def test_nested_alternations(self, pattern, string):
        assert regex(pattern).match(string) is not None

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
        assert regex(pattern).match(string) is not None

    @pytest.mark.parametrize(
        "pattern, string, failure_byte",
        [
            (r"a|b", "c", b"c"),  # Neither 'a' nor 'b'
            (r"cat|dog", "car", b"r"),  # Neither 'cat' nor 'dog'
            (r"apple|orange", "banana", b"b"),  # Neither 'apple' nor 'orange'
            (r"100|200", "300", b"3"),  # Neither '100' nor '200'
            (r"(a|b)c|d", "ae", b"e"),  # Neither 'ac' nor 'bc' nor 'd'
            (r"(a|b)+", "abbaabbabc", b"c"),  # 'c' does not match pattern '(a|b)+'
            (
                r"(dog|cat)s?",
                "cars",
                b"r",
            ),  # 'cars' does not match 'dog' or 'cat' with optional 's'
        ],
    )
    def test_alternations_failures(self, pattern, string, failure_byte):
        with pytest.raises(ParserException) as pe:
            regex(pattern).match(string, raise_exceptions=True)
        assert pe.value.current_byte == failure_byte
