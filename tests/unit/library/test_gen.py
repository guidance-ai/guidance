import pytest
from collections import defaultdict
import guidance
from guidance import gen, models


def test_basic():
    lm = models.Mock()
    try:
        lm += "Write a number: " + gen("text", max_tokens=3)
        assert len(lm["text"]) > 0
    finally:
        lm.close()


def test_stop_string():
    lm = models.Mock(b"<s>Count to 10: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10")
    try:
        lm += "Count to 10: 1, 2, 3, 4, 5, 6, 7, " + gen("text", stop=", 9")
        assert lm["text"] == "8"
    finally:
        lm.close()


def test_stop_char():
    lm = models.Mock(b"<s>Count to 10: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10")
    try:
        lm += "Count to 10: 1, 2, 3, 4, 5, 6, 7, " + gen("text", stop=",")
        assert lm["text"] == "8"
    finally:
        lm.close()


def test_save_stop():
    lm = models.Mock(b"<s>Count to 10: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10")
    try:
        lm += "Count to 10: 1, 2, 3, 4, 5, 6, 7, " + gen(
            "text", stop=",", save_stop_text="stop_text"
        )
        assert lm["stop_text"] == ","
    finally:
        lm.close()


def test_gsm8k():
    lm = models.Mock()
    try:
        (
            lm
            + """Question: Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?
    Answer: """
            + gen(max_tokens=30)
        )
        assert True
    finally:
        lm.close()


def test_pattern_optional():
    lm = models.Mock(b"<s>12342333")
    try:
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
    finally:
        lm.close()


def test_pattern_stops_when_fulfilled():
    lm = models.Mock(b"<s>123abc")
    try:
        lm += gen(regex=r"\d+", max_tokens=10, name="test")
        assert lm["test"] == "123"
    finally:
        lm.close()


def test_pattern_star():
    # lm = models.Mock(b"<s>1234233234<s>") # commented out because it is not a valid test
    # patterns = ['\d+233', '\d*233', '.+233', '.*233']
    # for pattern in patterns:
    #     lm2 = lm + '123' + gen(name='numbers', regex=pattern, max_tokens=10)
    #     assert lm2['numbers'] == '4233'
    lm = models.Mock(b"<s>123233")
    try:
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
    finally:
        lm.close()


def test_stop_regex():
    lm = models.Mock(b"<s>123a3233")
    try:
        lm2 = lm + "123" + gen(name="test", stop_regex=r"\d233", max_tokens=10)
        assert lm2["test"] == "a"
        lm = models.Mock(b"<s>123aegalera3233")
        lm2 = lm + "123" + gen(name="test", stop_regex=r"\d", max_tokens=30)
        assert lm2["test"] == "aegalera"
    finally:
        lm.close()


def test_stop_regex_star():
    lm = models.Mock(b"<s>123a3233")
    try:
        pattern = r"\d+233"
        lm2 = lm + "123" + gen(name="test", stop_regex=pattern, max_tokens=10)
        assert lm2["test"] == "a"
    finally:
        lm.close()


def test_empty_pattern():
    pattern = r"(Scott is bad)?(\d+)?"
    lm = models.Mock(b"<s>J<s>")
    try:
        lm2 = lm + "J" + gen(name="test", regex=pattern, max_tokens=30)
        assert lm2["test"] == ""
    finally:
        lm.close()


def test_list_append():
    """This tests is list append works across grammar appends."""
    lm = models.Mock(b"<s>bababababa")
    try:
        lm += "<s>"
        for _ in range(3):
            lm += gen("my_list", list_append=True, stop="a") + "a"
        assert isinstance(lm["my_list"], list)
        assert len(lm["my_list"]) == 3
    finally:
        lm.close()

@pytest.mark.xfail(
    reason="llguidance currently emits an additional empty capture group when no explicit stop is provided"
)
def test_list_append_no_explicit_stop():
    model = models.Mock("<s>bbbbbbb<s>")
    try:
        model += gen("list", list_append=True)
        assert model["list"][-1] == "bbbbbbb"
        assert len(model["list"]) == 1
    finally:
        model.close()

def test_list_append_in_grammar():
    """This tests is list append works within the same grammar."""
    lm = models.Mock(b"<s>bababababa")
    try:
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
    finally:
        lm.close()


def test_one_char_suffix_and_regex():
    model = models.Mock(b"<s>this is\na test")
    try:
        model += gen(regex=".*", suffix="\n", max_tokens=20)
        assert str(model) == "this is\n"
    finally:
        model.close()


def test_one_char_stop_and_regex():
    model = models.Mock(b"<s>this is\na test")
    try:
        model += gen(regex=".*", stop="\n", max_tokens=20)
        assert str(model) == "this is"
    finally:
        model.close()


def test_multiline():
    model = models.Mock(b"<s>this\nis\na\ntest<s>")
    try:
        model += gen(max_tokens=20)
        assert str(model) == "this\nis\na\ntest"
    finally:
        model.close()


def test_tool_call():
    called_args = []

    @guidance(dedent=False)
    def square(lm, x):
        called_args.append(x)
        return lm + str(int(x)**2)

    model = models.Mock(b"<s>Three squared is square(3)9<s>")
    try:
        model += gen(tools=[square], max_tokens=30)
        assert str(model) == "Three squared is square(3)9"
        assert called_args == ["3"]
    finally:
        model.close()


def test_tool_call_hidden():
    called_args = []

    @guidance(dedent=False)
    def square(lm, x):
        called_args.append(x)
        return lm + str(int(x)**2)

    model = models.Mock([
        b"<s>Three squared is square(3)",
        b"<s>Three squared is 9<s>"
    ])
    try:
        model += gen(tools=[square], hide_tool_call=True, max_tokens=30)
        assert str(model) == "Three squared is 9"
        assert called_args == ["3"]
    finally:
        model.close()


def test_tool_call_multi():
    called_args = defaultdict(list)

    @guidance(dedent=False)
    def square(lm, x):
        called_args['square'].append(x)
        return lm + str(int(x)**2)

    @guidance(dedent=False)
    def cube(lm, x):
        called_args['cube'].append(x)
        return lm + str(int(x)**3)

    model = models.Mock(
        b"<s>Three squared is square(3)9, which cubed is cube(9)729. Good job me.<s>",
    )
    try:
        model += gen(tools=[square, cube], hide_tool_call=False, max_tokens=50)
        assert str(model) == "Three squared is square(3)9, which cubed is cube(9)729. Good job me."
        assert called_args["square"] == ["3"]
        assert called_args["cube"] == ["9"]
    finally:
        model.close()


def test_tool_call_multi_hidden():
    called_args = defaultdict(list)

    @guidance(dedent=False)
    def square(lm, x):
        called_args['square'].append(x)
        return lm + str(int(x)**2)

    @guidance(dedent=False)
    def cube(lm, x):
        called_args['cube'].append(x)
        return lm + str(int(x)**3)

    model = models.Mock([
        b"<s>Three squared is square(3)",
        b"<s>Three squared is 9, which cubed is cube(9)",
        b"<s>Three squared is 9, which cubed is 729. Good job me.<s>"
    ])
    try:
        model += gen(tools=[square, cube], hide_tool_call=True, max_tokens=50)
        assert str(model) == "Three squared is 9, which cubed is 729. Good job me."
        assert called_args["square"] == ["3"]
        assert called_args["cube"] == ["9"]
    finally:
        model.close()
