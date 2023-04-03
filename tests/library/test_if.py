import guidance

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