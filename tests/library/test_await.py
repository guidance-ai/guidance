import guidance

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