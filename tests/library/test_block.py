import guidance

def test_hidden_block():
    """ Test the behavior of `if` with an `else` clause.
    """

    prompt = guidance("""This is a test {{#block hidden=True}}example{{/block}}""")
    out = prompt()
    assert out.text == "This is a test "