import guidance

def test_role():
    """ Test the behavior of `role`.
    """

    llm = guidance.llms.Mock()
    prompt = guidance("""
{{#role 'system'~}}
You are an assistant.
{{~/role}}

{{#role 'user'~}}
What is the weather?
{{~/role}}

{{#role 'assistant'~}}
{{gen}}
{{~/role}}
""", llm=llm)

    out = prompt()
    assert str(out) == '\n<|im_start|>system\nYou are an assistant.<|im_end|>\n\n<|im_start|>user\nWhat is the weather?<|im_end|>\n\n<|im_start|>assistant\nmock output 0<|im_end|>\n'

def test_short_roles():
    """ Test the behavior of the shorthand versions of `role`.
    """

    llm = guidance.llms.Mock()
    prompt = guidance("""
{{#system~}}
You are an assistant.
{{~/system}}

{{#user~}}
What is the weather?
{{~/user}}

{{#assistant~}}
{{gen}}
{{~/assistant}}
""", llm=llm)

    out = prompt(test="asdfa")
    assert str(out) == '\n<|im_start|>system\nYou are an assistant.<|im_end|>\n\n<|im_start|>user\nWhat is the weather?<|im_end|>\n\n<|im_start|>assistant\nmock output 0<|im_end|>\n'
    