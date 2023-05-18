import guidance
import pytest

def get_openai_llm(model_name, caching=False):
    """ Test the behavior of `stream=True` for an openai chat endpoint.
    """

    llm = guidance.llms.OpenAI(model_name, caching=caching)

    if llm.token is None:
        pytest.skip("OpenAI token not found")

    return llm