import guidance

def test_equal():
    """ Test the behavior of `equal`.
    """

    program = guidance("""{{#if (equal val 5)}}are equal{{else}}not equal{{/if}}""")
    assert str(program(val=4)) == "not equal"
    assert str(program(val=5)) == "are equal"
    assert str(program(val="5")) == "not equal"

def test_equal_with_symbol():
    """ Test the behavior of `equal` written as `==`.
    """

    program = guidance("""{{#if (== val 5)}}are equal{{else}}not equal{{/if}}""")
    assert str(program(val=4)) == "not equal"
    assert str(program(val=5)) == "are equal"
    assert str(program(val="5")) == "not equal"