import guidance

def test_gen():
    """ Test that LM geneation works.
    """
    llm = guidance.llms.OpenAI("text-curie-001")

    prompt = guidance("Hello my name is{{gen 'name' max_tokens=5}}", llm=llm)
    out = prompt()
    assert len(out["name"]) > 1

def test_gen_n_greater_than_one():
    """Test agentes, calling prompt twice"""
    llm = guidance.llms.OpenAI("text-curie-001")
    prompt = guidance('''The best thing about the beach is{{gen 'best' n=3 temperature=0.7 max_tokens=5}}''', llm=llm)
    a = prompt()
    assert len(a["best"]) == 3

def test_gen_n_greater_than_one():
    llm = guidance.llms.OpenAI("text-curie-001")
    prompt = guidance('''The best thing about the beach is{{gen 'best' n=3 temperature=0.7 max_tokens=5}}''', llm=llm)
    a = prompt()
    assert len(a["best"]) == 3