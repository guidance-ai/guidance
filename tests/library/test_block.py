import guidance

def test_hidden_block():
    """ Test the behavior of `if` with an `else` clause.
    """

    llm = guidance.llms.OpenAI("text-curie-001")
    prompt = guidance("""This is a test {{#block hidden=True}}example{{/block}}""", llm=llm)
    out = prompt()
    assert out.text == "This is a test "