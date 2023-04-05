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
    llm = guidance.llms.Mock()
    prompt = guidance('''The best thing about the beach is{{gen 'best' n=3 temperature=0.7 max_tokens=5}}''', llm=llm)
    a = prompt()
    assert "\n".join(a["best"]) == 'mock output 0\nmock output 1\nmock output 2'

def test_gen_n_greater_than_one_hidden():
    llm = guidance.llms.Mock()

    def aggregate(best):
        return '\n'.join(['- ' + x for x in best])
    prompt = guidance('''The best thing about the beach is{{gen 'best' temperature=0.7 n=3 hidden=True}}
{{aggregate best}}''', llm=llm)
    a = prompt(aggregate=aggregate)
    assert str(a) == 'The best thing about the beach is\n- mock output 0\n- mock output 1\n- mock output 2'