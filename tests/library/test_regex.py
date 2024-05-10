import pytest

from guidance import regex


class TestCharacterClasses:
    @pytest.mark.parametrize(
        "pattern, string",
        [
            (r"[abc]", "a"),
            (r"[abc]", "b"),
            (r"[^abc]", "d"),  # Negation
        ],
    )
    def test_character_classes(self, pattern, string):
        assert regex(pattern).match(string) is not None

    @pytest.mark.parametrize(
        "pattern, string",
        [
            (r"[a-z]", "m"),
            (r"[0-9]", "5"),
        ],
    )
    def test_ranges(self, pattern, string):
        assert regex(pattern).match(string) is not None


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
