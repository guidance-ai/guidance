import guidance

def test_select():
    """ Test the behavior of `select`.
    """

    llm = guidance.llms.OpenAI("text-curie-001", caching=False)
    program = guidance("Is Everest very tall?\nAnswer 'Yes' or 'No': '{{#select 'name'}}Yes{{or}}No{{/select}}", llm=llm)
    out = program()
    assert out["name"] in ["Yes", "No"]

def test_select_longtext():
    """ Test the behavior of `select`.
    """

    llm = guidance.llms.OpenAI("text-curie-001", caching=False)
    program = guidance("""Is Everest very tall?\nAnswer:
{{#select 'name'}}No because of all the other ones.{{or}}Yes because I saw it.{{/select}}""", llm=llm)
    out = program()
    assert out["name"] in ["No because of all the other ones.", "Yes because I saw it."]

def test_select_longtext_transformers():
    """ Test the behavior of `select`.
    """

    llm = guidance.llms.Transformers("gpt2", caching=False)
    program = guidance("""Is Everest very tall?\nAnswer:
{{#select 'name'}}No because of all the other ones.{{or}}Yes because I saw it.{{/select}}""", llm=llm)
    out = program()
    assert out["name"] in ["No because of all the other ones.", "Yes because I saw it."]

def test_select_with_list():
    """ Test the behavior of `select` in non-block mode.
    """

    # llm = guidance.llms.Mock("Yes")
    llm = guidance.llms.OpenAI("text-curie-001", caching=False)
    program = guidance("Is Everest very tall?\nAnswer 'Yes' or 'No': '{{select 'name' options=options}}", llm=llm)
    out = program(options=["Yes", "No"])
    assert out["name"] in ["Yes", "No"]