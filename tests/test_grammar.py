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

def test_comment2():
    prompt = guidance("Hello, {{! this is a comment}}Bob!{{@prefix}}")
    assert str(prompt()) == "Hello, Bob!Hello, Bob!"

def test_long_comment():
    prompt = guidance("Hello, {{!-- this is a comment --}}Bob!{{@prefix}}")
    assert str(prompt()) == "Hello, Bob!Hello, Bob!"

def test_long_comment_ws_strip():
    prompt = guidance("Hello, {{~!-- this is a comment --~}} Bob!{{@prefix}}")
    assert str(prompt()) == "Hello,Bob!Hello,Bob!"

def test_comment_ws_strip():
    prompt = guidance("Hello, {{~! this is a comment ~}} Bob!{{@prefix}}")
    assert str(prompt()) == "Hello,Bob!Hello,Bob!"

def test_escape_command():
    prompt = guidance("Hello, \{{command}} Bob!")
    assert str(prompt()) == "Hello, {{command}} Bob!"

def test_indexing():
    prompt = guidance("Hello, {{arr[0]}} Bob!")
    assert str(prompt(arr=["there"])) == "Hello, there Bob!"

def test_special_var():
    prompt = guidance("{{#each arr}}Hello, {{@index}}-{{this}}!{{/each}}")
    assert str(prompt(arr=["there"])) == "Hello, 0-there!"

    prompt = guidance("{{#geneach 'arr' num_iterations=1}}Hello, {{@index}}!{{/each}}")
    assert str(prompt(arr=["there"])) == "Hello, 0!"

def test_special_var_index():
    prompt = guidance("{{#each arr}}{{arr[@index]}}{{/each}}!")
    assert str(prompt(arr=["there"])) == "there!"
    prompt = guidance("{{#geneach 'out' num_iterations=1}}{{arr[@index]}}{{/each}}!")
    assert str(prompt(arr=["there"])) == "there!"