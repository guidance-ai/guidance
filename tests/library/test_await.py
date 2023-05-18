import guidance

def test_await():
    """ Test the behavior of `await`.
    """

    prompt = guidance("""Is Everest very tall?
User response: '{{set 'user_response' (await 'user_response')}}'""")
    waiting_prompt = prompt()
    assert str(waiting_prompt) == "Is Everest very tall?\nUser response: '{{set 'user_response' (await 'user_response')}}'"
    out = waiting_prompt(user_response="Yes")
    assert str(out) == "Is Everest very tall?\nUser response: 'Yes'"