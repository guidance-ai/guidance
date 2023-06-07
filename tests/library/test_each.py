import guidance
from ..utils import get_llm

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

def test_each_after_await():
    """Test an each loop when we are not executing."""

    prompt = guidance("Hello, {{name}}!{{await 'some_var'}}{{#each names}} {{this}}{{/each}}")
    assert str(prompt(name="Guidance", names=["Bob", "Sue"])) == "Hello, Guidance!{{await 'some_var'}}{{#each names}} {{this}}{{/each}}"

def test_each_over_an_await():
    """Test an each loop when we are not executing."""

    program = guidance("Hello, {{name}}!{{#each (await 'names')}} {{this}}{{/each}}")
    partial_execution = program(name="Guidance")
    assert str(partial_execution) == "Hello, Guidance!{{#each (await 'names')}} {{this}}{{/each}}"
    full_execution = partial_execution(names=["Bob", "Sue"])
    assert str(full_execution) == "Hello, Guidance! Bob Sue"

def test_each_parallel():
    """Test an each loop run in parallel."""

    program = guidance("Hello, {{name}}!{{#each names parallel=True hidden=True}} {{this}}{{/each}}")
    executed_program = program(name="Guidance", names=["Bob", "Sue", "Sam"])
    assert str(executed_program) == "Hello, Guidance!"

def test_each_parallel_with_gen():
    """Test an each loop run in parallel with generations inside."""

    llm = guidance.llms.Mock(["Pizza", "Burger", "Salad"])

    program = guidance("""Hello, {{name}}! Here are 5 names and their favorite food:
{{#each names parallel=True hidden=True}}{{this}}: {{gen 'foods' list_append=True}}
{{/each}}""", llm=llm)
    executed_program = program(name="Guidance", names=["Bob", "Sue", "Sam"])
    assert str(executed_program) == "Hello, Guidance! Here are 5 names and their favorite food:\n"
    for food in executed_program["foods"]:
        assert food in ["Pizza", "Burger", "Salad"]

def test_each_parallel_with_gen_openai():
    """Test an each loop run in parallel with generations inside using OpenAI."""

    llm = get_llm("openai:text-curie-001")

    program = guidance("""Hello, {{name}}! Here are 5 names and their favorite food:
{{#each names parallel=True hidden=True}}{{this}}: {{gen 'foods' list_append=True}}
{{/each}}""", llm=llm)
    executed_program = program(name="Guidance", names=["Bob", "Sue", "Sam"])
    assert str(executed_program) == "Hello, Guidance! Here are 5 names and their favorite food:\n"
    assert len(executed_program["foods"]) == 3

# def test_with_stop():
#     """ Test an each loop when we are not executing.
#     """

#     token_count = 0
#     def token_limit(item, variables, template_context):
#         nonlocal token_count
#         tokenizer = template_context["@tokenizer"]
#         token_count += len(tokenizer.encode(item))
#         return token_count > 3

#     program = guidance("""This is a list of names:
# {{set 'token_start' (len (tokenize prefix))~}}
# {{#each names stop=token_limit}} {{this}}
# {{~if (len (tokenize prefix)) - token_start > 100}}{{break}}{{/if~}}
# {{/each}}""", token_limit=token_limit)

#     program = guidance("Hello, {{name}}!{{#each names)}} {{this}}{{/each}}")
#     executed_program = program(name="Guidance", names=["Bob", "Sue", "Sam"])
#     assert str(executed_program) == "Hello, Guidance! Bob Sue Sam"