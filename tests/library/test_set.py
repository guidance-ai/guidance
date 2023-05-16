import guidance

def test_set():
    """ Test the behavior of `set`.
    """

    program = guidance("""{{set 'output' 234}}{{output}}""")
    assert str(program()) == "234234"
    
    program = guidance("""{{set 'output' 234 hidden=True}}{{output}}""")
    assert str(program()) == "234"