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

def test_select_multi_path():
    """ Test the behavior of `select` and confirm the returns probability distribution sums to 1.
    """
    import numpy as np

    options = [
        "This is one sentence about fish and dogs.",
        "This is another sentence about fish and dogs.",
        "Sure, here is a sentence about cats.",
        "Sure thing, here is a sentence about cats.",
        "This is one"
    ]

    llm = get_llm("transformers:gpt2")
    out = guidance(
        "Hello, write me a sentence. {{select 'sentence' logprobs='probs' options=options}}",
        llm=llm
    )(options=options)
    assert abs(1 - np.exp([l for l in out["probs"].values()]).sum()) < 1e-5
    assert out["sentence"] in options

def test_select_multi_path_with_suffix():
    """ Test the behavior of `select` and confirm the returns probability distribution sums to 1.
    """
    import numpy as np

    options = [
        "This is one sentence about fish and dogs.",
        "This is another sentence about fish and dogs.",
        "Sure, here is a sentence about cats.",
        "Sure thing, here is a sentence about cats.",
        "This is one"
    ]

    llm = get_llm("transformers:gpt2")
    out = guidance(
        "Hello, write me a sentence. {{select 'sentence' logprobs='probs' options=options}} And this is the suffix.",
        llm=llm
    )(options=options)
    assert abs(1 - np.exp([l for l in out["probs"].values()]).sum()) < 1e-5
    assert out["sentence"] in options

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

@pytest.mark.parametrize("llm", ["transformers:gpt2", "openai:text-curie-001"])
def test_non_greedy_tokenize(llm):
    """ Test the behavior of `select` when the GPT tokenizer is not greedy (odd space handling).
    """

    llm = get_llm(llm)
    program = guidance('''Is the following sentence offensive? Please answer with a single word, either "Yes", "No", or "Maybe".
Sentence: {{example}}
Answer:{{#select "answer" logprobs='logprobs'}} 
    Yes{{or}} 
    No{{or}} 
    Maybe
{{/select}}''', llm=llm)
    executed_program = program(example='I hate tacos')
    assert executed_program["answer"] in [" \n    Yes", " \n    No", " \n    Maybe\n"]

import guidance

guidance.llm = guidance.llms.OpenAI("text-davinci-003")
# the {{#select}} command allows you to use the LLM to select from a set of options


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