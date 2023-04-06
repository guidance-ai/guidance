import guidance

def test_if():
    """ Test the behavior of `if`.
    """

    llm = guidance.llms.OpenAI("text-curie-001")
    prompt = guidance("""Answer: {{#if flag}}Yes{{/if}}""", llm=llm)

    for flag in [True, 1, "random text"]:
        out = prompt(flag=flag)
        assert str(out) == "Answer: Yes"
    
    for flag in [False, 0, ""]:
        out = prompt(flag=flag)
        assert str(out) == "Answer: "

def test_if_else():
    """ Test the behavior of `if` with an `else` clause.
    """

    llm = guidance.llms.OpenAI("text-curie-001")
    prompt = guidance("""Answer 'Yes' or 'No': '{{#if flag}}Yes{{else}}No{{/if}}'""", llm=llm)

    for flag in [True, 1, "random text"]:
        out = prompt(flag=flag)
        assert str(out) == "Answer 'Yes' or 'No': 'Yes'"
    
    for flag in [False, 0, ""]:
        out = prompt(flag=flag)
        assert str(out) == "Answer 'Yes' or 'No': 'No'"