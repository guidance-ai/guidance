import guidance

def test_strip():
    """ Test the behavior of `strip`.
    """

    program = guidance("""{{strip ' this is '}}""")
    assert str(program()) == "this is"