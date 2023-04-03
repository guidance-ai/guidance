import guidance

def test_select():
    """ Test the behavior of `select`.
    """

    llm = guidance.llms.OpenAI("text-curie-001")
    prompt = guidance("Is Everest very tall?\nAnswer 'Yes' or 'No': '{{#select 'name'}}Yes{{or}}No{{/select}}", llm=llm)
    out = prompt()
    assert out["name"] in ["Yes", "No"]