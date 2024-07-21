import guidance
from guidance import gen, models

import logging
logging.basicConfig(level=logging.DEBUG)

def test_call_embeddings():
    """This tests calls embedded in strings."""
    model = models.Mock()

    @guidance(dedent=False)
    def bla(lm, bla):
        lm += bla + "ae" + gen(max_tokens=10)
        return lm

    @guidance(dedent=False)
    def ble(lm):
        lm += f"""
    ae galera! {bla('33')}
    let's do more stuff!!""" + gen(
            max_tokens=10
        )
        return lm

    assert "{{G|" not in str(model + ble())

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
    # logging.debug(f"Result: {str(result)}")
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

def test_nested_fstrings():
    """Test nested f-strings."""
    @guidance(stateless=True, dedent=True)
    def nested_fstring(lm):
        lm += f"""\
        Outer {{
            "inner": f"{{
                "value": {1+1}
            }}"
        }}
        """
        return lm

    lm = guidance.models.Mock()
    result = lm + nested_fstring()
    assert str(result) == 'Outer {\n    "inner": "{\n    "value": 2\n}"\n}'

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