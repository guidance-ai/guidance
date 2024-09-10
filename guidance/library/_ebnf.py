from collections import defaultdict
from typing import Callable, Union, Optional, Sequence

from lark import Lark
from lark.grammar import NonTerminal, Rule, Terminal
from lark.load_grammar import FromPackageLoader

from .._grammar import GrammarFunction, Join
from .._guidance import guidance
from ._subgrammar import subgrammar, lexeme
from . import capture, select


class EBNF:
    @classmethod
    def open(cls, path: str, **kwargs):
        parser = Lark.open(path, **kwargs)
        return cls(parser)

    @classmethod
    def from_package(
        cls, package: str, grammar_path: str, search_paths: Sequence[str] = [""], **kwargs
    ):
        parser = Lark.open_from_package(package, grammar_path, search_paths, **kwargs)
        return cls(parser)

    @classmethod
    def from_grammar_string(cls, grammar: str, **kwargs):
        parser = Lark(grammar, **kwargs)
        return cls(parser)

    def __init__(self, parser: Lark):
        self.parser = parser
        self.start = self.parser.options.options["start"]

        # grammars for nonterminals -- regex seems to be the simplest solution
        self.terminal_regexes: dict[Terminal, str] = {
            Terminal(terminal.name): terminal.pattern.to_regexp()
            for terminal in self.parser.terminals
        }
        self.terminal_grammars: dict[Terminal, GrammarFunction] = {
            Terminal(terminal.name): lexeme(terminal.pattern.to_regexp())
            for terminal in self.parser.terminals
        }

        # Collect rules by nonterminal such that we can easily `select` over
        # the corresponding grammars
        self.rules_by_nonterminal: dict[NonTerminal, list[Rule]] = defaultdict(list)
        for rule in self.parser.rules:
            self.rules_by_nonterminal[rule.origin].append(rule)

        # Callables to produce grammars for nonterminals
        # They need to be callables rather than literal grammars to avoid
        # infinite recursion (taking advantage of late binding)
        self.nonterminal_grammar_callables: dict[Terminal, Callable[[], GrammarFunction]] = {}

    def build_term(self, term: Union[Terminal, NonTerminal]) -> GrammarFunction:
        if isinstance(term, Terminal):
            return self.terminal_grammars[term]
        if isinstance(term, NonTerminal):
            grammar_callable = self.nonterminal_grammar_callables.setdefault(
                term, self.build_nonterminal(term)
            )
            return grammar_callable()
        raise TypeError(f"term must be one of type Union[Terminal, NonTerminal], got {type(term)}")

    def build_rule(self, rule: Rule) -> GrammarFunction:
        terms = [self.build_term(term) for term in rule.expansion]
        if len(terms) == 1 and rule.alias is None:
            # Unwrap unnamed singletons
            return terms[0]
        else:
            return Join(terms, name=rule.alias)

    def build_nonterminal(self, nonterminal: NonTerminal) -> Callable[[], GrammarFunction]:
        # No-arg function to be wrapped in guidance decorator.
        #   - Associated with exactly one nonterminal
        #   - Needs to be no-arg to allow for mutual recursion via `Placeholder`s
        #   - Wrap in guidance decorator later so that we can set the __name__ first
        def inner(lm):
            # Options to select over (one for each rule associated with a nonterminal)
            options = [self.build_rule(rule) for rule in self.rules_by_nonterminal[nonterminal]]
            return lm + select(options)

        # Set name and wrap
        inner.__name__ = nonterminal.name
        return guidance(inner, stateless=True, dedent=False, cache=True)

    def build(self, name=None) -> GrammarFunction:
        # Trigger recursive build of grammar using start nonterminal
        body = select([self.build_term(NonTerminal(s)) for s in self.start])
        ignore_rx = "|".join(
            self.terminal_regexes[Terminal(name)] for name in self.parser.ignore_tokens
        )
        return subgrammar(name=name, body=body, skip_regex=ignore_rx)


def ebnf():
    pass
