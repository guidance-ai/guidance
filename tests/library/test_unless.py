import guidance

def test_unless():
    """ Test the behavior of `unless`.
    """

    program = guidance("""Answer: {{#unless flag}}Yes{{/unless}}""")

    for flag in [True, 1, "random text"]:
        out = program(flag=flag)
        assert str(out) == "Answer: "
    
    for flag in [False, 0, ""]:
        out = program(flag=flag)
        assert str(out) == "Answer: Yes"