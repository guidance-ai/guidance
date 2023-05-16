import guidance

def test_break_each():
    """ Test the behavior of `break` in an `each` loop.
    """

    llm = guidance.llms.Mock()
    program = guidance("""Loop to ten:
{{~#each list}}
{{this}}
{{~#if (equal this 5)}}{{break}}{{/if~}}
{{/each}}""", llm=llm)
    out = program(list=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    assert out.text == "Loop to ten:\n1\n2\n3\n4\n5"