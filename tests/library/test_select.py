import guidance
from ..utils import get_openai_llm, get_transformers_llm

def test_select():
    """ Test the behavior of `select`.
    """

    llm = get_openai_llm("text-curie-001")
    program = guidance("Is Everest very tall?\nAnswer 'Yes' or 'No': '{{#select 'name'}}Yes{{or}}No{{/select}}", llm=llm)
    out = program()
    assert out["name"] in ["Yes", "No"]

def test_select_longtext():
    """ Test the behavior of `select`.
    """

    llm = get_openai_llm("text-curie-001")
    program = guidance("""Is Everest very tall?\nAnswer:
{{#select 'name'}}No because of all the other ones.{{or}}Yes because I saw it.{{/select}}""", llm=llm)
    out = program()
    assert out["name"] in ["No because of all the other ones.", "Yes because I saw it."]

def test_select_longtext_transformers():
    """ Test the behavior of `select`.
    """

    llm = get_transformers_llm("gpt2")
    program = guidance("""Is Everest very tall?\nAnswer:
{{#select 'name'}}No because of all the other ones.{{or}}Yes because I saw it.{{/select}}""", llm=llm)
    out = program()
    assert out["name"] in ["No because of all the other ones.", "Yes because I saw it."]

def test_select_with_list():
    """ Test the behavior of `select` in non-block mode.
    """

    llm = get_openai_llm("text-curie-001")
    program = guidance("Is Everest very tall?\nAnswer 'Yes' or 'No': '{{select 'name' options=options}}", llm=llm)
    out = program(options=["Yes", "No"])
    assert out["name"] in ["Yes", "No"]

def test_select_list_append():
    """ Test the behavior of `select` with list_append=True.
    """

    llm = get_openai_llm("text-curie-001")
    program = guidance("Is Everest very tall?\n{{select 'name' options=options list_append=True}}\n{{select 'name' options=options list_append=True}}", llm=llm)
    out = program(options=["Yes", "No"])
    assert len(out["name"]) == 2
    for v in out["name"]:
        assert v in ["Yes", "No"]

def test_select_odd_spacing():
    """ Test the behavior of `select` with list_append=True.
    """

    llm = get_openai_llm("text-curie-001")
    prompt = guidance('''Is the following sentence offensive? Please answer with a single word, either "Yes", "No", or "Maybe".
    Sentence: {{example}}
    Answer: {{#select "answer" logprobs='logprobs'}} Yes{{or}} Nein{{or}} Maybe{{/select}}''', llm=llm)
    prompt = prompt(example='I hate tacos.')
    assert prompt["answer"] in [" Yes", " Nein", " Maybe"]