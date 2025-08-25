import pytest

from guidance import gen, models


def test_basic():
    lm = models.Mock()
    lm += "Write a number: " + gen("text", max_tokens=3)
    assert len(lm["text"]) > 0


def test_stop_string():
    lm = models.Mock(b"<s>Count to 10: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10")
    lm += "Count to 10: 1, 2, 3, 4, 5, 6, 7, " + gen("text", stop=", 9")
    assert lm["text"] == "8"


def test_stop_char():
    lm = models.Mock(b"<s>Count to 10: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10")
    lm += "Count to 10: 1, 2, 3, 4, 5, 6, 7, " + gen("text", stop=",")
    assert lm["text"] == "8"


def test_save_stop():
    lm = models.Mock(b"<s>Count to 10: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10")
    lm += "Count to 10: 1, 2, 3, 4, 5, 6, 7, " + gen("text", stop=",", save_stop_text="stop_text")
    assert lm["stop_text"] == ","


def test_gsm8k():
    lm = models.Mock()
    (
        lm
        + """Question: Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?
Answer: """
        + gen(max_tokens=30)
    )
    assert True


def test_pattern_optional():
    lm = models.Mock(b"<s>12342333")
    pattern = ".?233"
    lm2 = lm + "123" + gen(name="numbers", regex=pattern, max_tokens=10)
    assert lm2["numbers"] == "4233"
    lm = models.Mock(b"<s>1232333")
    pattern = ".?233"
    lm2 = lm + "123" + gen(name="numbers", regex=pattern, max_tokens=10)
    assert lm2["numbers"] == "233"
    pattern = r"(Scott is bad)?(\d+)?o"
    lm = models.Mock(b"<s>John was a little man full of things")
    lm2 = lm + "J" + gen(name="test", regex=pattern, max_tokens=30)
    assert lm2["test"] == "o"


def test_pattern_stops_when_fulfilled():
    lm = models.Mock(b"<s>123abc")
    lm += gen(regex=r"\d+", max_tokens=10, name="test")
    assert lm["test"] == "123"


def test_pattern_star():
    # lm = models.Mock(b"<s>1234233234<s>") # commented out because it is not a valid test
    # patterns = ['\d+233', '\d*233', '.+233', '.*233']
    # for pattern in patterns:
    #     lm2 = lm + '123' + gen(name='numbers', regex=pattern, max_tokens=10)
    #     assert lm2['numbers'] == '4233'
    lm = models.Mock(b"<s>123233")
    patterns = [r"\d*233", ".*233"]
    for pattern in patterns:
        lm2 = lm + "123" + gen(name="numbers", regex=pattern, max_tokens=10)
        assert lm2["numbers"].startswith("233")
    pattern = ".*(\n|little)"
    lm = models.Mock(b"<s>John was a little")
    lm2 = lm + "J" + gen(name="test", regex=pattern, max_tokens=30)
    assert lm2["test"].startswith("ohn was a little")
    lm = models.Mock(b"<s>John was a litt\n")
    lm2 = lm + "J" + gen(name="test", regex=pattern, max_tokens=30)
    assert lm2["test"].startswith("ohn was a litt\n")


def test_stop_regex():
    lm = models.Mock(b"<s>123a3233")
    lm2 = lm + "123" + gen(name="test", stop_regex=r"\d233", max_tokens=10)
    assert lm2["test"] == "a"
    lm = models.Mock(b"<s>123aegalera3233")
    lm2 = lm + "123" + gen(name="test", stop_regex=r"\d", max_tokens=30)
    assert lm2["test"] == "aegalera"


def test_stop_regex_star():
    lm = models.Mock(b"<s>123a3233")
    pattern = r"\d+233"
    lm2 = lm + "123" + gen(name="test", stop_regex=pattern, max_tokens=10)
    assert lm2["test"] == "a"


def test_empty_pattern():
    pattern = r"(Scott is bad)?(\d+)?"
    lm = models.Mock(b"<s>J<s>")
    lm2 = lm + "J" + gen(name="test", regex=pattern, max_tokens=30)
    assert lm2["test"] == ""


def test_list_append():
    """This tests is list append works across grammar appends."""
    lm = models.Mock(b"<s>bababababa")
    lm += "<s>"
    for _ in range(3):
        lm += gen("my_list", list_append=True, stop="a") + "a"
    assert isinstance(lm["my_list"], list)
    assert len(lm["my_list"]) == 3


@pytest.mark.xfail(
    reason="llguidance currently emits an additional empty capture group when no explicit stop is provided"
)
def test_list_append_no_explicit_stop():
    model = models.Mock("<s>bbbbbbb<s>")
    model += gen("list", list_append=True)
    assert model["list"][-1] == "bbbbbbb"
    assert len(model["list"]) == 1


def test_list_append_in_grammar():
    """This tests is list append works within the same grammar."""
    lm = models.Mock(b"<s>bababababa")
    lm += "<s>"
    lm += (
        gen("my_list", list_append=True, stop="a")
        + "a"
        + gen("my_list", list_append=True, stop="a")
        + "a"
        + gen("my_list", list_append=True, stop="a")
    )
    assert isinstance(lm["my_list"], list)
    assert len(lm["my_list"]) == 3


def test_one_char_suffix_and_regex():
    model = models.Mock(b"<s>this is\na test")
    model += gen(regex=".*", suffix="\n", max_tokens=20)
    assert str(model) == "this is\n"


def test_one_char_stop_and_regex():
    model = models.Mock(b"<s>this is\na test")
    model += gen(regex=".*", stop="\n", max_tokens=20)
    assert str(model) == "this is"


def test_multiline():
    model = models.Mock(b"<s>this\nis\na\ntest<s>")
    model += gen(max_tokens=20)
    assert str(model) == "this\nis\na\ntest"
