import guidance

def test_variable_interpolation():
    """Test variable interpolation in prompt
    """

    prompt = guidance.Prompt("Hello, {{name}}!")
    assert str(prompt(name="Guidance")) == "Hello, Guidance!"

def test_each():
    """Test each loops.
    """

    prompt = guidance.Prompt("Hello, {{name}}!{{#each names}} {{this}}{{/each}}")
    assert str(prompt(name="Guidance", names=["Bob", "Sue"])) == "Hello, Guidance! Bob Sue"

def test_each_with_objects():
    """Test each loops.
    """

    prompt = guidance.Prompt("Hello, {{name}}!{{#each names}} {{this.name}}{{/each}}")
    out = prompt(
        name="Guidance",
        names=[{"name": "Bob"}, {"name": "Sue"}]
    )
    assert str(out) == "Hello, Guidance! Bob Sue"

def test_generate():
    """Test that LM geneation works.
    """

    prompt = guidance.Prompt("Hello my name is{{generate 'name' max_tokens=5}}")
    out = prompt()
    assert len(out["name"]) > 1

def test_select():
    """Test that LM select works.
    """

    prompt = guidance.Prompt("Answer 'Yes' or 'No': '{{#select 'name'}}Yes{{or}}No{{/select}}")
    out = prompt()
    assert out["name"] in ["Yes", "No"]