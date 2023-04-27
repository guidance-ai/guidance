import guidance

def test_geneach():
    """ Test a geneach loop.
    """
    llm = guidance.llms.OpenAI("text-curie-001")
    prompt = guidance('''<instructions>Generate a list of three names</instructions>
<list>{{#geneach 'names' stop="</list>"}}
<item index="{{@index}}">{{gen 'this'}}</item>{{/geneach}}</list>"''', llm=llm)
    out = prompt()
    assert len(out["names"]) == 3
