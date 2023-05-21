import guidance
import pytest
from ..utils import get_llm

@pytest.mark.parametrize("llm", ["transformers:gpt2", "openai:text-curie-001"])
def test_select(llm):
    """ Test the behavior of `select`.
    """

    llm = get_llm(llm)
    program = guidance("Is Everest very tall?\nAnswer 'Yes' or 'No': '{{#select 'name'}}Yes{{or}}No{{/select}}", llm=llm)
    out = program()
    assert out["name"] in ["Yes", "No"]

@pytest.mark.parametrize("llm", ["transformers:gpt2", "openai:text-curie-001"])
def test_select_longtext(llm):
    """ Test the behavior of `select`.
    """

    llm = get_llm(llm)
    program = guidance("""Is Everest very tall?\nAnswer:
{{#select 'name'}}No because of all the other ones.{{or}}Yes because I saw it.{{/select}}""", llm=llm)
    out = program()
    assert out["name"] in ["No because of all the other ones.", "Yes because I saw it."]

@pytest.mark.parametrize("llm", ["transformers:gpt2", "openai:text-curie-001"])
def test_select_with_list(llm):
    """ Test the behavior of `select` in non-block mode.
    """

    llm = get_llm(llm)
    program = guidance("Is Everest very tall?\nAnswer 'Yes' or 'No': '{{select 'name' options=options logprobs='logprobs'}}", llm=llm)
    out = program(options=["Yes", "No", "Maybe", "I don't know"])
    assert out["name"] in ["Yes", "No", "Maybe", "I don't know"]
    for k in out["logprobs"]:
        assert out["logprobs"][k] <= 0

@pytest.mark.parametrize("llm", ["transformers:gpt2", "openai:text-curie-001"])
def test_select_list_append(llm):
    """ Test the behavior of `select` with list_append=True.
    """

    llm = get_llm(llm)
    program = guidance("Is Everest very tall?\n{{select 'name' options=options list_append=True}}\n{{select 'name' options=options list_append=True}}", llm=llm)
    out = program(options=["Yes", "No"])
    assert len(out["name"]) == 2
    for v in out["name"]:
        assert v in ["Yes", "No"]

@pytest.mark.parametrize("llm", ["transformers:gpt2", "openai:text-curie-001"])
def test_select_names(llm):
    """ Test the behavior of `select` with hard to guess terms.
    """

    llm = get_llm(llm)
    out = guidance(
        "Hello, {{#select 'name'}}Alice{{or}}Bob{{/select}}",
        llm=llm
    )()
    assert out["name"] in ["Alice", "Bob"]

@pytest.mark.parametrize("llm", ["transformers:gpt2", "openai:text-curie-001"])
def test_select_odd_spacing(llm):
    """ Test the behavior of `select` with list_append=True.
    """

    llm = get_llm(llm)
    prompt = guidance('''Is the following sentence offensive? Please answer with a single word, either "Yes", "No", or "Maybe".
    Sentence: {{example}}
    Answer: {{#select "answer" logprobs='logprobs'}} Yes{{or}} Nein{{or}} Maybe{{/select}}''', llm=llm)
    prompt = prompt(example='I hate tacos.')
    assert prompt["answer"] in [" Yes", " Nein", " Maybe"]

@pytest.mark.parametrize("llm", ["transformers:gpt2", "openai:text-curie-001"])
def test_overlapping_options(llm):
    """ Test the behavior of `select` when one option is a prefix of another.
    """

    llm = get_llm(llm)
    options = ['a', 'aa']
    program = guidance("'{{select options=options}}", llm=llm)
    out = program(options=options)
    assert out["selected"] in options

# TODO: fix this next
# def test_unexpected_tokens():
#     """ Test the behavior of `select` when the next tokens are hard to predict.
#     """

#     llm = get_transformers_llm("gpt2")
#     options = ['a', 'b']
#     program = guidance("some word xy{{select options=options}}", llm=llm)
#     out = program(options=options)
#     assert out["selected"] in options

# TODO: test when we have few starting tokens