import pytest
import guidance
from guidance import models, select, gen, optional, one_or_more
from guidance._parser import ParserException

def test_select_reset_pos():
    model = models.Mock()
    model += 'This is' + select(options=['bad', 'quite bad'])
    assert str(model) in ["This isbad", "This isquite bad"]

def test_select_simple(selected_model):
    lm = selected_model
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


@pytest.mark.parametrize('prefix', ['', 'abc'])
@pytest.mark.parametrize('suffix', ['', '123'])
@pytest.mark.parametrize('stop',   ['@', '@@'])
def test_gen_stop_excluded(prefix, suffix, stop):
    matchstr = f"{prefix}{stop}{suffix}"
    grammar = gen(stop=stop)
    match = grammar.match(matchstr, allow_partial=True)
    assert match is None

@pytest.mark.parametrize('prefix', ['', 'abc'])
@pytest.mark.parametrize('suffix', ['', '123'])
@pytest.mark.parametrize('stop',   ['@', '@@'])
def test_gen_stop_continues(prefix, suffix, stop):
    matchstr = f"{prefix}{stop}{suffix}"
    grammar = gen(stop=stop) + stop + suffix
    match = grammar.match(matchstr, allow_partial=True)
    assert match is not None
    assert not match.partial

@pytest.mark.parametrize('prefix', ['', 'abc'])
@pytest.mark.parametrize('suffix', ['', '123'])
@pytest.mark.parametrize('stop',   ['@', '@@'])
def test_gen_stop_repeat(prefix, suffix, stop):
    matchstr = f"{prefix}{stop}{stop}{suffix}"
    grammar = gen(stop=stop) + stop + suffix
    match = grammar.match(matchstr, allow_partial=True)
    assert match is None

@pytest.mark.parametrize('stop',   ['@', '@@'])
def test_multiple_gen_stops(stop):
    grammar = one_or_more(gen(stop=stop) + stop)
    matchstr = 'abc' + stop + '123' + stop + 'xyz' + stop
    match = grammar.match(matchstr, allow_partial=True)
    assert match is not None


# Cartesian product of stop token (in grammar, not in grammar) and (in matchstr, not in matchstr)

def test_stop_no_grammar_no_matchstr():
    grammar = gen(regex=r'a*', stop='@') + gen(regex=r'b*')
    matchstr = 'aaaaaabbbbb'
    grammar.match(matchstr, raise_exceptions=True)

def test_stop_no_grammar_yes_matchstr():
    # Somehow a '@' made it into our string even though it should have been hidden.
    grammar = gen(regex=r'a*', stop='@') + gen(regex=r'b*')
    matchstr = 'aaaaaa@bbbbb'
    with pytest.raises(ParserException):
        # TODO: should be a HiddenCommitPointException
        grammar.match(matchstr, allow_partial=True, raise_exceptions=True)

def test_stop_yes_grammar_no_matchstr():
    # We got a 'b' when we were expecting an 'a' or an '@'
    grammar = gen(regex=r'a*', stop='@') + '@' + gen(regex=r'b*')
    matchstr = 'aaaaaabbbbb'
    with pytest.raises(ParserException):
        grammar.match(matchstr, allow_partial=True, raise_exceptions=True)

def test_stop_yes_grammar_yes_matchstr():
    grammar = gen(regex=r'a*', stop='@') + '@' + gen(regex=r'b*')
    matchstr = 'aaaaaa@bbbbb'
    grammar.match(matchstr, raise_exceptions=True)


def test_ambiguous_hidden_stop_bad():
    # Cannot be a match because once we see an "@" we know that no numbers can appear 
    # (since an "@" can't appear in the first pattern and numbers can't appear in the second).
    grammar = gen(regex=r'.*', stop='@') + gen(regex=r'[^0-9]*', name="end")
    matchstr = 'ajdkjck@sk9dfjjsdfjk'
    with pytest.raises(ParserException):
        grammar.match(matchstr, allow_partial=True, raise_exceptions=True)

def test_ambiguous_hidden_stop_good():
    grammar = gen(regex=r'.*', stop='@') + gen(regex=r'[^0-9]*', name="end")
    matchstr = 'ajdkjcksk9dfjjsdfjk'
    grammar.match(matchstr, raise_exceptions=True)