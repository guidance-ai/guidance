import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Union, cast

# to support the embedding of guidance functions inside Python f-strings we use tags with these delimiters
tag_start = "{{G|"  # start of a call tag
tag_end = "|G}}"  # end of a call tag
_tag_pool: dict[str, Union["Function", "GrammarNode"]] = (
    {}
)  # the Functions or GrammarNodes associated with the tags
_tag_pattern = re.compile(
    re.escape(tag_start) + r"([^\|]+)" + re.escape(tag_end)
)  # the pattern for matching call tags


class Tagged:
    def __str__(self):
        """Creates a string tag that can be used to retrieve this object."""

        # save the call in our call pool, ready to be run when it is attached to an LM object
        str_id = str(id(self))
        if str_id not in _tag_pool:
            _tag_pool[str_id] = self

        # return a string representation of this call so it can be combined with other strings/calls
        return tag_start + str_id + tag_end


class StatefulException(Exception):
    """This is raised when we try and use the state of a grammar object like it was a live model.

    Note that eventually it would be nice to support stateful parser/grammar constructs directly, but
    such "parser combinators" cannot be run effciently in Python. So we use a traditional parser and
    grammar separation (hence the need for this exception)."""

    pass


@dataclass(slots=True)
class Function(Tagged):
    name: str = field(init=False)
    f: Callable
    args: tuple[Any, ...]
    kwargs: dict[str, Any]

    def __post_init__(self):
        self.name = self.f.__name__

    def __call__(self, model):
        model = self.f(model, *self.args, **self.kwargs)
        if model is None:
            raise Exception(
                f"The guidance function `{self.f.__name__}` did not return a model object! You need to return an updated model object at the end of your guidance function."
            )
        return model

    def __add__(self, other):
        if not isinstance(other, (str, GrammarNode, Function)):
            return NotImplemented

        if isinstance(other, str):
            other = extract_tags(other)

        def __add__(model):
            return self(model) + other

        return Function(__add__, [], {})

    def __radd__(self, other):
        if not isinstance(other, (str, GrammarNode, Function)):
            return NotImplemented

        if isinstance(other, str):
            other = extract_tags(other)

        def __radd__(model):
            return self(model + other)

        return Function(__radd__, [], {})


@dataclass(slots=True, eq=False)
class GrammarNode(ABC, Tagged):
    @abstractmethod
    def __repr__(self) -> str:
        pass

    def is_atomic(self) -> bool:
        return True

    def is_terminal(self) -> bool:
        return all(child.is_terminal() for child in self.children())

    def top_str(self) -> str:
        return repr(self)

    def simplify(self) -> "GrammarNode":
        return self

    def children(self) -> list["GrammarNode"]:
        return []

    def __add__(self, other) -> "GrammarNode":
        if not isinstance(other, (str, GrammarNode)):
            return NotImplemented

        if isinstance(other, str):
            other = extract_tags(other)

        if isinstance(other, Function):
            return other.__radd__(self)

        return JoinNode([self, other])

    def __radd__(self, other) -> "GrammarNode":
        if not isinstance(other, (str, GrammarNode)):
            return NotImplemented

        if isinstance(other, str):
            other = extract_tags(other)

        if isinstance(other, Function):
            return other.__add__(self)

        return JoinNode([other, self])

    def lark_serialize(self) -> str:
        return lark_serialize(self)


@dataclass(slots=True, eq=False)
class RuleNode(GrammarNode):
    name: str
    value: GrammarNode
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    capture_name: Optional[str] = None
    rule_is_terminal: bool = False

    def children(self) -> list["GrammarNode"]:
        return [self.value]

    def __repr__(self) -> str:
        rep = self.name
        attrs = {}
        for attr in ["temperature", "max_tokens", "capture_name"]:
            if getattr(self, attr) is not None:
                attrs[attr] = getattr(self, attr)
        if attrs:
            rep += f"[{', '.join(f'{k}={v}' for k, v in attrs.items())}]"
        rep += f": {self.value.top_str()}"
        return rep


@dataclass(slots=True, eq=False)
class RuleRefNode(GrammarNode):
    target: Optional[RuleNode] = None

    def is_terminal(self) -> bool:
        if self.target is None:
            return False
        else:
            return self.target.rule_is_terminal

    def __repr__(self) -> str:
        if self.target is None:
            # Error in repr is bad. Maybe we shouldn't be using repr.
            raise ValueError("RuleRefNode has no target")
        else:
            return self.target.name


@dataclass(slots=True, eq=False)
class LiteralNode(GrammarNode):
    value: str

    def __repr__(self) -> str:
        # TODO: escape?
        return f'"{self.value}"'


@dataclass(slots=True, eq=False)
class RegexNode(GrammarNode):
    regex: str

    def __repr__(self) -> str:
        return f"/{self.regex}/"


@dataclass(slots=True, eq=False)
class SelectNode(GrammarNode):
    alternatives: list[GrammarNode]

    def __post_init__(self):
        self.alternatives = [
            RuleRefNode(alt) if isinstance(alt, RuleNode) else alt for alt in self.alternatives
        ]

    def top_str(self) -> str:
        return "\n     | ".join(repr(alt) for alt in self.alternatives)

    def __repr__(self) -> str:
        return "(" + " | ".join(repr(alt) for alt in self.alternatives) + ")"

    def is_atomic(self) -> bool:
        # Not really atomic, but we already wrap it in parentheses
        return True

    def simplify(self) -> "GrammarNode":
        for i in range(len(self.alternatives)):
            self.alternatives[i] = self.alternatives[i].simplify()
        if len(self.alternatives) == 1:
            return self.alternatives[0]
        return self

    def children(self) -> list["GrammarNode"]:
        return self.alternatives


@dataclass(slots=True, eq=False)
class JoinNode(GrammarNode):
    nodes: list[GrammarNode]

    def __post_init__(self):
        self.nodes = [RuleRefNode(n) if isinstance(n, RuleNode) else n for n in self.nodes]

    def __repr__(self) -> str:
        if not self.nodes:
            return '""'
        else:
            return " ".join(repr(node) for node in self.nodes)

    def is_atomic(self) -> bool:
        return False

    def simplify(self) -> "GrammarNode":
        for i in range(len(self.nodes)):
            self.nodes[i] = self.nodes[i].simplify()
        if len(self.nodes) == 1:
            return self.nodes[0]
        return self

    def children(self) -> list["GrammarNode"]:
        return self.nodes


@dataclass(slots=True, eq=False)
class RepeatNode(GrammarNode):
    node: GrammarNode
    min: int
    max: Optional[int]

    def __post_init__(self):
        self.node = RuleRefNode(self.node) if isinstance(self.node, RuleNode) else self.node

    def children(self) -> list["GrammarNode"]:
        return [self.node]

    def simplify(self) -> GrammarNode:
        self.node = self.node.simplify()

    def __repr__(self) -> str:
        inner = repr(self.node)
        if not self.node.is_atomic():
            inner = f"({inner})"
        match (self.min, self.max):
            case (0, None):
                return f"{inner}*"
            case (1, None):
                return f"{inner}+"
            case (0, 1):
                return f"{inner}?"
            case (min, None):
                return f"{inner}{{{min},}}"
            case (min, max):
                return f"{inner}{{{min},{max}}}"


def extract_tags(s: str) -> Union[GrammarNode, Function]:
    parts = cast(list[str], _tag_pattern.split(s))
    obj = string(parts.pop(0))
    is_tag = True
    for part in parts:
        if is_tag:
            obj += _tag_pool[part]
        else:
            obj += string(part)
        is_tag = not is_tag
    return obj


def string(s: str) -> GrammarNode:
    return LiteralNode(s)


def resolve(node: GrammarNode) -> dict[str, RuleNode]:
    rules: dict[str, RuleNode] = {}
    seen: set[GrammarNode] = set()

    def add_node(n: GrammarNode):
        if n in seen:
            return
        seen.add(n)

        if isinstance(n, RuleNode):
            name = n.name
            if name in rules:
                i = 1
                while f"{name}_{i}" in rules:
                    i += 1
                name = f"{name}_{i}"
            rules[name] = n

        elif isinstance(n, RuleRefNode):
            if n.target is None:
                raise ValueError("RuleRefNode has no target")
            add_node(n.target)

        for child in n.children():
            if isinstance(child, RuleNode):
                raise ValueError("Child RuleNodes should always be wrapped in RuleRefNodes")
            add_node(child)

    if isinstance(node, RuleNode) and node.name == "start":
        add_node(node)
    else:
        if isinstance(node, RuleNode):
            target = RuleRefNode(node)
        else:
            target = node
        add_node(RuleNode("start", target))

    num_fix = 1
    while num_fix > 0:
        num_fix = 0
        for r in rules.values():
            if r.name != "start" and not r.rule_is_terminal and r.value.is_terminal():
                r.rule_is_terminal = True
                num_fix += 1

    for name, r in rules.items():
        new_name = name.replace("-", "_")
        # convert fooBar_Baz to foo_bar_baz
        new_name = re.sub(r"([a-z])([A-Z])", r"\1_\2", new_name).lower()
        if r.rule_is_terminal:
            new_name = new_name.upper()
        else:
            new_name = new_name.lower()
        if r.name != new_name:
            r.name = new_name

    return rules


def test():
    number = RuleNode("NUMBER", RegexNode("[0-9]+"))
    expr_ref = RuleRefNode()
    factor = RuleNode("factor", SelectNode(["(" + expr_ref + ")", number]))
    term_ref = RuleRefNode()
    term = RuleNode("term", SelectNode([term_ref + "*" + factor, factor]))
    term_ref.target = term
    expr = RuleNode("expr", SelectNode([expr_ref + "+" + term, term]))
    expr_ref.target = expr
    return expr.lark_serialize()


def lark_serialize(node: GrammarNode) -> str:
    rules = resolve(node)
    res = "%llguidance {}\n\n"
    prev_nl = True
    for r in rules.values():
        s = repr(r)
        if not prev_nl and "\n" in s:
            res += "\n"
        res += s + "\n"
        prev_nl = "\n" in s
        if prev_nl:
            res += "\n"
    return res
