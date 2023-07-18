import guidance
from ..utils import get_llm

def test_hidden_block():
    """ Test the behavior of generic `block`.
    """

    prompt = guidance("""This is a test {{#block hidden=True}}example{{/block}}""")
    out = prompt()
    assert out.text == "This is a test "

def test_empty_block():
    """ Test the behavior of a completely empty `block`.
    """

    prompt = guidance(
        "{{#block}}{{#if nonempty}}{{nonempty}}{{/if}}{{/block}}",
    )
    out = prompt(nonempty=False)
    assert out.text == ''

def test_name_capture():
    prompt = guidance(
        "This is a block: {{#block 'my_block'}}text inside block{{/block}}",
    )
    out = prompt()
    assert out["my_block"] == 'text inside block'

def test_name_capture_whitespace():
    prompt = guidance(
        "This is a block: {{#block 'my_block'}} text inside block {{/block}}",
    )
    out = prompt()
    assert out["my_block"] == ' text inside block '