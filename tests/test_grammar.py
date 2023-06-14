import guidance

def test_variable_interpolation():
    """ Test variable interpolation in prompt
    """

    prompt = guidance("Hello, {{name}}!")
    assert str(prompt(name="Guidance")) == "Hello, Guidance!"

def test_command_call():
    prompt = guidance("Hello, {{add 1 2}}!")
    assert str(prompt(name="Guidance")) == "Hello, 3!"

def test_paren_command_call():
    prompt = guidance("Hello, {{add(1, 2)}}!")
    assert str(prompt(name="Guidance")) == "Hello, 3!"

def test_nested_command_call():
    prompt = guidance("Hello, {{add (add 1 2) 3}}!")
    assert str(prompt(name="Guidance")) == "Hello, 6!"

def test_nested_paren_command_call():
    prompt = guidance("Hello, {{add add(1, 2) 3}}!")
    assert str(prompt(name="Guidance")) == "Hello, 6!"

def test_infix_plus():
    prompt = guidance("Hello, {{1 + 2}}!")
    assert str(prompt()) == "Hello, 3!"

def test_infix_plus_nested():
    prompt = guidance("Hello, {{set 'variable' 1 + 2}}!")
    assert prompt()["variable"] == 3

def test_comment():
    prompt = guidance("Hello, {{! this is a comment}}Bob!")
    assert str(prompt()) == "Hello, Bob!"

def test_long_comment():
    prompt = guidance("Hello, {{!-- this is a comment --}}Bob!")
    assert str(prompt()) == "Hello, Bob!"

def test_long_comment_ws_strip():
    prompt = guidance("Hello, {{~!-- this is a comment --~}} Bob!")
    assert str(prompt()) == "Hello,Bob!"

def test_comment_ws_strip():
    prompt = guidance("Hello, {{~! this is a comment ~}} Bob!")
    assert str(prompt()) == "Hello,Bob!"

def test_escape_command():
    prompt = guidance("Hello, \{{command}} Bob!")
    assert str(prompt()) == "Hello, {{command}} Bob!"