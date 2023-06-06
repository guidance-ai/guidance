import guidance

def test_parse():
    """ Test the basic behavior of `parse`.
    """

    program = guidance("""This is parsed: {{parse template}}""")
    assert str(program(template="My name is {{name}}", name="Bob")) == "This is parsed: My name is Bob"