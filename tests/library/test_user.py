import pytest

@pytest.fixture
def mock_llm():
    return guidance.llms.Mock("the output")

def test_user(mock_llm):
    """ Test behavior of `user`.
    """

    llm = mock_llm

    program = guidance("""
    {{~#system}}You are fake.{{/system}}
    {{#user}}You are real.{{/user}}
    {{#assistant}}{{gen 'output' save_prompt='prompt'}}{{/assistant}}""", llm=llm)
    out = program()

    assert 'user\nYou are real.' in str(out)
