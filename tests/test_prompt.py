import guidance

def test_variable_interpolation():
    """ Test variable interpolation in prompt
    """

    prompt = guidance("Hello, {{name}}!")
    assert str(prompt(name="Guidance")) == "Hello, Guidance!"

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

def test_generate():
    """ Test that LM geneation works.
    """
    llm = guidance.llms.OpenAI("text-curie-001")

    prompt = guidance("Hello my name is{{generate 'name' max_tokens=5}}", llm=llm)
    out = prompt()
    assert len(out["name"]) > 1

def test_select():
    """ Test the behavior of `select`.
    """

    llm = guidance.llms.OpenAI("text-curie-001")
    prompt = guidance("Is Everest very tall?\nAnswer 'Yes' or 'No': '{{#select 'name'}}Yes{{or}}No{{/select}}", llm=llm)
    out = prompt()
    assert out["name"] in ["Yes", "No"]

def test_await():
    """ Test the behavior of `await`.
    """

    llm = guidance.llms.OpenAI("text-curie-001")
    prompt = guidance("""Is Everest very tall?
User response: '{{set 'user_response' (await 'user_response')}}'
Answer 'Yes' or 'No': '{{#select 'name'}}Yes{{or}}No{{/select}}""", llm=llm)
    waiting_prompt = prompt()
    assert str(waiting_prompt) == "Is Everest very tall?\nUser response: '{{set 'user_response' (await 'user_response')}}'\nAnswer 'Yes' or 'No': '{{#select 'name'}}Yes{{or}}No{{/select}}"
    out = waiting_prompt(user_response="Yes")
    assert str(out).startswith("Is Everest very tall?\nUser response: 'Yes'\nAnswer 'Yes' or 'No': '")
    assert out["name"] in ["Yes", "No"]

def test_if():
    """ Test the behavior of `if`.
    """

    llm = guidance.llms.OpenAI("text-curie-001")
    prompt = guidance("""Answer: {{#if flag}}Yes{{/if}}""", llm=llm)

    for flag in [True, "Yes", "yes", "YES", 1, 1.0, "True", "true", "TRUE"]:
        out = prompt(flag=flag)
        assert str(out) == "Answer: Yes"
    
    for flag in [False, "No", "no", "NO", 0, 0.0, "False", "false", "FALSE"]:
        out = prompt(flag=flag)
        assert str(out) == "Answer: "

def test_if_else():
    """ Test the behavior of `if` with an `else` clause.
    """

    llm = guidance.llms.OpenAI("text-curie-001")
    prompt = guidance("""Answer 'Yes' or 'No': '{{#if flag}}Yes{{else}}No{{/if}}'""", llm=llm)

    for flag in [True, "Yes", "yes", "YES", 1, 1.0, "True", "true", "TRUE"]:
        out = prompt(flag=flag)
        assert str(out) == "Answer 'Yes' or 'No': 'Yes'"
    
    for flag in [False, "No", "no", "NO", 0, 0.0, "False", "false", "FALSE"]:
        out = prompt(flag=flag)
        assert str(out) == "Answer 'Yes' or 'No': 'No'"