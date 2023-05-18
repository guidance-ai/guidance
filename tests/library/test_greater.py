import guidance

def test_greater():
    """ Test the behavior of `greater`.
    """

    program = guidance("""{{#if (greater val 5)}}greater{{else}}not greater{{/if}}""")
    assert str(program(val=4)) == "not greater"
    assert str(program(val=6)) == "greater"
    assert str(program(val=5.3)) == "greater"

def test_greater_with_symbol():
    """ Test the behavior of `greater` used as `>`.
    """

    program = guidance("""{{#if (> val 5)}}greater{{else}}not greater{{/if}}""")
    assert str(program(val=4)) == "not greater"
    assert str(program(val=6)) == "greater"
    assert str(program(val=5.3)) == "greater"