import pytest
import guidance
from guidance import gen

def test_dedent_basic():
    """Test that dedent functionality in f-strings works across Python versions."""
    @guidance(stateless=True, dedent=True)
    def character_maker(lm):
        lm += f"""\
        {{
            "name": "{1+1}",
            "age": "{gen('name', stop='"', max_tokens=1)}",
        }}"""
        return lm

    lm = guidance.models.Mock()
    result = lm + character_maker()
    assert str(result).startswith("{")


def test_basic_multiline_fstring():
    """Test a simple multiline f-string."""
    @guidance(stateless=True, dedent=True)
    def character_maker(lm):
        lm += f"""\
        {{
            "name": "{'har' + 'sha'}",
            "age": "{314}",
        }}"""
        return lm

    lm = guidance.models.Mock()
    result = lm + character_maker()
    assert str(result) == '{\n    "name": "harsha",\n    "age": "314",\n}'

def test_mixed_content():
    """Test mixed f-strings and regular strings."""
    @guidance(stateless=True, dedent=True)
    def mixed_content(lm):
        s = "Regular string\n"
        s += f"""\
        {{
            "name": "{'har' + 'sha'}",
            "age": "{314}",
        }}"""
        lm += s
        return lm

    lm = guidance.models.Mock()
    result = lm + mixed_content()
    assert str(result) == 'Regular string\n{\n    "name": "harsha",\n    "age": "314",\n}'

def test_non_fstring_multiline():
    """Test multiline strings that are not f-strings."""
    @guidance(stateless=True, dedent=True)
    def non_fstring_multiline(lm):
        s = """\
        Line 1
        Line 2
        Line 3
        """
        lm += s
        return lm

    lm = guidance.models.Mock()
    result = lm + non_fstring_multiline()
    assert str(result) == 'Line 1\nLine 2\nLine 3\n'

def test_empty_strings():
    """Test empty strings."""
    @guidance(stateless=True, dedent=True)
    def empty_string(lm):
        s = f"""\
        {""}"""
        lm += s
        return lm

    lm = guidance.models.Mock()
    result = lm + empty_string()
    assert str(result) == ''

def test_inconsistent_indentation():
    """Test strings with inconsistent indentation."""
    @guidance(stateless=True, dedent=True)
    def inconsistent_indentation(lm):
        s = f"""\
        {{
        "name": "{'har' + 'sha'}",
          "age": "{314}",
        "weapon": "{'sword'}"
        }}"""
        lm += s
        return lm
    
    lm = guidance.models.Mock()
    result = lm + inconsistent_indentation()
    assert str(result) == '{\n"name": "harsha",\n  "age": "314",\n"weapon": "sword"\n}'

# NOTE [HN]: The following two tests currently don't work, but they're fairly special/rare cases.
# Some implementation thoughts for the future:
# Nested f-strings: try creating a custom handler for ast.FormattedValue in the handler?
# closure functions: storing and rebinding vars referenced in func globals when recompiling?
# def test_nested_fstrings():
#     """Test nested f-strings."""
#     @guidance(stateless=True, dedent=True)
#     def nested_fstring(lm):
#         lm += f"""\
#         Outer {{
#             "inner": f"{{
#                 "value": {1+1}
#             }}"
#         }}
#         """
#         return lm

#     lm = guidance.models.Mock()
#     result = lm + nested_fstring()
#     assert str(result) == 'Outer {\n    "inner": "{\n    "value": 2\n}"\n}'

# def test_closure_function():
#     """Test function with closures referring to outer variables."""
#     @guidance(stateless=True, dedent=True)
#     def outer_function(lm):
#         outer_var = "outer_value"
        
#         def inner_function():
#             inner_var = f"""\
#             Inner function variable:
#                 outer_var: {outer_var}
#                 """
#             return inner_var
#         lm += inner_function()
#         return lm

#     lm = guidance.models.Mock()
#     result = lm + outer_function()
#     assert result == "Inner function variable:\nouter_var: outer_value\n"

def test_exception_on_repeat_calls():
    @guidance(stateless=True, dedent=False)
    def raises(lm):
        assert False
    with pytest.raises(AssertionError):
        raises()
    with pytest.raises(AssertionError):
        # Test against failure to reset the grammar function;
        # improper handling may not raise and may instead return
        # a Placeholder grammar node
        raises()
