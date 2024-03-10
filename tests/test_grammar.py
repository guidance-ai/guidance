import guidance
from guidance import models, select, gen, optional
from .utils import get_model

def test_select_reset_pos():
    model = models.Mock()
    model += 'This is' + select(options=['bad', 'quite bad'])
    assert str(model) in ["This isbad", "This isquite bad"]

def test_select_simple():
    lm = get_model("transformers:gpt2")
    options = ['baad I think', 'bad I think', 'bad']
    lm = lm + 'Scott is quite ' + select(name='bad', options=options)
    assert lm['bad'] in options

def test_select_longer():
    '''This tests to ensure that the grammar is extended greedily.'''
    lm = models.Mock(b"<s>Scott is a very nice man.")
    lm += "Scott is a very " + select(name='text', options=['nice', 'nice man.'])
    assert lm["text"] == 'nice man.'

def test_grammar_plus_fstring():
    @guidance(stateless=True, dedent=False)
    def test(lm):
        val = 4
        lm += f"the value of {val} is best! {gen(max_tokens=1)}"
        return lm

    lm = models.Mock()
    lm += test()
    assert "{{G|" not in str(lm)


class TestRecursion:

    def test_simple_recursion(self):

        @guidance(stateless=True, dedent=False)
        def grammar(lm):
            return lm + 'x' + optional(grammar())

        grammar()

    def test_mutual_recursion(self):

        @guidance(stateless=True, dedent=False)
        def grammar1(lm):
            return lm + 'x' + grammar2()

        @guidance(stateless=True, dedent=False)
        def grammar2(lm):
            return lm + 'y' + optional(grammar1())

        grammar1()
        grammar2()

    def test_multiple_mutual_recursion(self):
        @guidance(stateless=True, dedent=False)
        def grammar1(lm):
            return lm + 'x' + grammar2()

        @guidance(stateless=True, dedent=False)
        def grammar2(lm):
            return lm + 'y' + grammar3()

        @guidance(stateless=True, dedent=False)
        def grammar3(lm):
            return lm + 'z' + optional(grammar1())

        grammar1()
        grammar2()
        grammar3()

    def test_branching_mutual_recursion(self):
        @guidance(stateless=True, dedent=False)
        def grammar1(lm):
            return lm + 'x' + grammar2()

        @guidance(stateless=True, dedent=False)
        def grammar2(lm):
            return lm + 'y' + select([grammar1(), grammar3()])

        @guidance(stateless=True, dedent=False)
        def grammar3(lm):
            return lm + 'z' + optional(grammar1())

        grammar1()
        grammar2()
        grammar3()
