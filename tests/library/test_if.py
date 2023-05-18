import guidance

def test_if():
    """ Test the behavior of `if`.
    """

    prompt = guidance("""Answer: {{#if flag}}Yes{{/if}}""")

    for flag in [True, 1, "random text"]:
        out = prompt(flag=flag)
        assert str(out) == "Answer: Yes"
    
    for flag in [False, 0, ""]:
        out = prompt(flag=flag)
        assert str(out) == "Answer: "

def test_if_else():
    """ Test the behavior of `if` with an `else` clause.
    """

    prompt = guidance("""Answer 'Yes' or 'No': '{{#if flag}}Yes{{else}}No{{/if}}'""")

    for flag in [True, 1, "random text"]:
        out = prompt(flag=flag)
        assert str(out) == "Answer 'Yes' or 'No': 'Yes'"
    
    for flag in [False, 0, ""]:
        out = prompt(flag=flag)
        assert str(out) == "Answer 'Yes' or 'No': 'No'"