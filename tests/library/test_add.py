import guidance

def test_add():
    """ Basic test of `add`.
    """

    program = guidance("""Write a number: {{set 'user_response' (add 20 variable)}}""")
    assert program(variable=10)["user_response"] == 30
    assert program(variable=20.1)["user_response"] == 40.1

def test_add_multi():
    """ Test more than 2 arguments for `add`.
    """

    program = guidance("""Write a number: {{set 'user_response' (add 20 5 variable)}}""")
    assert program(variable=10)["user_response"] == 35
    assert program(variable=20.1)["user_response"] == 45.1