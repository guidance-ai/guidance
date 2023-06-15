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
<item index="{{@index}}">{{gen 'this'}}</item>{{/geneach}}</list>''', llm=llm)
    out = prompt()
    assert len(out["names"]) == 3
    assert out["names"] == ["Bob", "Sue", "Joe"]
    assert str(out) == """<instructions>Generate a list of three names</instructions>
<list>
<item index="0">Bob</item>
<item index="1">Sue</item>
<item index="2">Joe</item></list>"""

def test_geneach_with_join():
    """ Test a geneach loop.
    """

    llm = guidance.llms.Mock({
        'Joe</item>': {"text": '</list>', "finish_reason": "stop"},
        '</item>': {"text": '\n<item', "finish_reason": "length"},
        '">' : ["Bob", "Sue", "Joe"],
    })
    prompt = guidance('''<instructions>Generate a list of three names</instructions>
<list>{{#geneach 'names' join="<mark>" stop="</list>"}}
<item index="{{@index}}">{{gen 'this'}}</item>{{/geneach}}</list>''', llm=llm)
    out = prompt()
    assert len(out["names"]) == 3
    assert out["names"] == ["Bob", "Sue", "Joe"]
    assert str(out) == """<instructions>Generate a list of three names</instructions>
<list>
<item index="0">Bob</item><mark>
<item index="1">Sue</item><mark>
<item index="2">Joe</item></list>"""

def test_geneach_single_call():
    """ Test a geneach loop.
    """

    llm = guidance.llms.Mock('''
<item index="0">Bob</item>
<item index="1">Sue</item>
<item index="2">Jow</item>
</list>''')
    prompt = guidance('''<instructions>Generate a list of three names</instructions>
<list>{{#geneach 'names' single_call=True stop="</list>"}}
<item index="{{@index}}">{{gen 'this'}}</item>{{/geneach}}</list>"''', llm=llm)
    out = prompt()
    assert len(out["names"]) == 3

def test_geneach_with_index():
    """ Test a geneach loop.
    """

    llm = guidance.llms.Mock(["Qs", "A1", "A2", "A3", "A4", "A5"])
    program = guidance('''
{{~#system~}}You are a teacher.{{~/system~}}

{{~#user~}}
Make a list of questions.
{{~/user~}}

{{~#assistant~}}
{{gen 'qmap' temperature=1.0 max_tokens=50}}
{{~/assistant~}}

{{#geneach 'answers' num_iterations=5}}"
{{#user~}}
answer The following question: {{questions[@index]}}
{{~/user}}
{{#assistant~}}
{{gen 'this' temperature=0.7}}
{{~/assistant}}"
{{/geneach}}''', llm=llm)

    executed_program = program(questions=["Q1", "Q2", "Q3", "Q4", "Q5"])