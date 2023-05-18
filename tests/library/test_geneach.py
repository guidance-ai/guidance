import guidance

def test_geneach():
    """ Test a geneach loop.
    """

    llm = guidance.llms.Mock({
        'Joe</item>': {"text": '</list>', "finish_reason": "stop"},
        '</item>': {"text": '\n<item', "finish_reason": "length"},
        '">' : ["Bob", "Sue", "Joe"],
    })
    prompt = guidance('''<instructions>Generate a list of three names</instructions>
<list>{{#geneach 'names' stop="</list>"}}
<item index="{{@index}}">{{gen 'this'}}</item>{{/geneach}}</list>"''', llm=llm)
    out = prompt()
    assert len(out["names"]) == 3
