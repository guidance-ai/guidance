import guidance

def test_subtract():
    """ Basic test of `subtract`.
    """

    program = guidance("""Write a number: {{set 'user_response' (subtract 20 variable)}}""")
    assert program(variable=10)["user_response"] == 10
    assert abs(program(variable=20.1)["user_response"] + 0.1) < 1e-5