import guidance

def test_each():
    """ Test an each loop.
    """

    prompt = guidance("Hello, {{name}}!{{#each names}} {{this}}{{/each}}")
    assert str(prompt(name="Guidance", names=["Bob", "Sue"])) == "Hello, Guidance! Bob Sue"

def test_each_with_objects():
    """ Test an each loop with objects.
    """

    prompt = guidance("Hello, {{name}}!{{#each names}} {{this.name}}{{/each}}")
    out = prompt(
        name="Guidance",
        names=[{"name": "Bob"}, {"name": "Sue"}]
    )
    assert str(out) == "Hello, Guidance! Bob Sue"

def test_missing_list():
    prompt = guidance('''List of ideas:{{#each ideas}}test{{this}}{{/each}}''', await_missing=True)
    assert str(prompt()) == "List of ideas:{{#each ideas}}test{{this}}{{/each}}"
    # try:
    #     out = prompt()
    # except KeyError:
    #     return
    # assert False, "An error should have been raised because the list is missing!"