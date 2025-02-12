import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Union, cast

from ._parser import ByteParser, ByteParserException
from ._schema import LarkGrammar, LLGrammar

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


class Match:
    def __init__(self, captures, log_probs, partial):
        self.captures = captures
        self.log_probs = log_probs
        self.partial = partial

    def __getitem__(self, key):
        return self.captures[key]

    def __len__(self):
        return len(self.captures)

    def __bool__(self):
        return True

    def __str__(self):
        return str(self.captures)

    def __repr__(self):
        return (
            "<guidance.Match object; captures="
            + str(self.captures)
            + "; partial="
            + str(self.partial)
            + ">"
        )


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
            other = parse_tags(other)

        def __add__(model):
            return self(model) + other

        return Function(__add__, [], {})

    def __radd__(self, other):
        if not isinstance(other, (str, GrammarNode, Function)):
            return NotImplemented

        if isinstance(other, str):
            other = parse_tags(other)

        def __radd__(model):
            return self(model + other)

        return Function(__radd__, [], {})


@dataclass(slots=True, eq=False)
class GrammarNode(ABC, Tagged):
    @abstractmethod
    def lark_str(self, top: bool = False) -> str:
        pass

    @property
    def is_atomic(self) -> bool:
        return True

    @property
    def is_null(self) -> bool:
        return False

    @property
    def is_terminal(self) -> bool:
        return all(child.is_terminal for child in self.children())

    def simplify(self) -> "GrammarNode":
        return self

    def children(self) -> list["GrammarNode"]:
        return []

    def __repr__(self) -> str:
        return self.lark_str()

    def __add__(self, other) -> "GrammarNode":
        if not isinstance(other, (str, GrammarNode)):
            return NotImplemented

        if isinstance(other, str):
            other = parse_tags(other)

        if isinstance(other, Function):
            return other.__radd__(self)

        return JoinNode([self, other])

    def __radd__(self, other) -> "GrammarNode":
        if not isinstance(other, (str, GrammarNode)):
            return NotImplemented

        if isinstance(other, str):
            other = parse_tags(other)

        if isinstance(other, Function):
            return other.__add__(self)

        return JoinNode([other, self])

    def __getitem__(self, key):
        raise StatefulException("GrammarNodes can't access state!")

    def match(
        self,
        byte_string: Union[str, bytes],
        allow_partial: bool = False,
        raise_exceptions: bool = False,
    ) -> Union[Match, None]:
        if isinstance(byte_string, str):
            byte_string = byte_string.encode()
        parser = ByteParser(self.ll_grammar())

        try:
            parser.consume_bytes(byte_string)
            if not allow_partial:
                parser.force_done()
        except ByteParserException:
            if raise_exceptions:
                raise
            else:
                return None

        if not allow_partial and not parser.matched():
            return None

        if parser.matched():
            parser.force_done()

        return Match(*parser.get_captures(), partial=not parser.matched())  # type: ignore[misc]

    def forced_prefix(self) -> str:
        parser = ByteParser(self.ll_grammar())
        return parser.bytes.decode("utf-8", errors="ignore")

    def ll_grammar(self) -> LLGrammar:
        return LLGrammar(grammars=[LarkGrammar(lark_grammar=lark_serialize(self))])


@dataclass(slots=True, eq=False)
class LiteralNode(GrammarNode):
    value: str

    @property
    def is_null(self) -> bool:
        return self.value == ""

    def lark_str(self, top: bool = False) -> str:
        # TODO: escape?
        return f'"{self.value}"'


@dataclass(slots=True, eq=False)
class RegexNode(GrammarNode):
    regex: str

    @property
    def is_null(self) -> bool:
        return self.regex == ""

    def lark_str(self, top: bool = False) -> str:
        return f"/{self.regex}/"


@dataclass(slots=True, eq=False)
class SelectNode(GrammarNode):
    alternatives: list[GrammarNode]

    @property
    def is_null(self) -> bool:
        return all(alt.is_null for alt in self.alternatives)

    def lark_str(self, top: bool = False) -> str:
        if self.is_null:
            return '""'
        else:
            if top:
                return "\n     | ".join(
                    alt.lark_str() for alt in self.alternatives if not alt.is_null
                )
            else:
                return (
                    "("
                    + " | ".join(alt.lark_str() for alt in self.alternatives if not alt.is_null)
                    + ")"
                )

    @property
    def is_atomic(self) -> bool:
        # Not really atomic, but we already wrap it in parentheses
        return True

    def simplify(self) -> "GrammarNode":
        nullable = any(alt.is_null for alt in self.alternatives)
        self.alternatives = [alt.simplify() for alt in self.alternatives if not alt.is_null]
        if nullable:
            self.alternatives.append(LiteralNode(""))
        if len(self.alternatives) == 1:
            return self.alternatives[0]
        return self

    def children(self) -> list["GrammarNode"]:
        return self.alternatives


@dataclass(slots=True, eq=False)
class JoinNode(GrammarNode):
    nodes: list[GrammarNode]

    @property
    def is_null(self) -> bool:
        return all(node.is_null for node in self.nodes)

    def lark_str(self, top: bool = False) -> str:
        if self.is_null:
            return '""'
        else:
            return " ".join(node.lark_str() for node in self.nodes if not node.is_null)

    @property
    def is_atomic(self) -> bool:
        return False

    def simplify(self) -> "GrammarNode":
        self.nodes = [node.simplify() for node in self.nodes if not node.is_null]
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

    @property
    def is_null(self) -> bool:
        return self.node.is_null or self.min == self.max == 0

    def __post_init__(self):
        if self.min < 0:
            raise ValueError("min must be >= 0")
        if self.max is not None and self.max < self.min:
            raise ValueError("max must be >= min")

    def children(self) -> list["GrammarNode"]:
        return [self.node]

    def simplify(self) -> GrammarNode:
        self.node = self.node.simplify()
        return self

    def lark_str(self, top: bool = False) -> str:
        inner = self.node.lark_str()
        if not self.node.is_atomic:
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
        raise RuntimeError("Unreachable")


@dataclass(slots=True, eq=False)
class JsonNode(GrammarNode):
    schema: Union[bool, dict[str, Any]]

    @property
    def is_terminal(self) -> bool:
        return False

    def lark_str(self, top: bool = False) -> str:
        if top:
            indent = 2
        else:
            indent = None
        return f"%json {json.dumps(self.schema, indent=indent)}"


@dataclass(slots=True, eq=False)
class SubstringNode(GrammarNode):
    chunks: list[str]

    @property
    def is_terminal(self) -> bool:
        # TODO: true? technically a regex...
        return False

    def lark_str(self, top: bool = False) -> str:
        if top:
            indent = 2
        else:
            indent = None
        return f'%regex {json.dumps({"substring_chunks": self.chunks}, indent=indent)}'


@dataclass(slots=True, eq=False)
class RuleNode(GrammarNode):
    name: str
    value: GrammarNode
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    capture_name: Optional[str] = None
    _is_terminal: bool = field(default=False, init=False)
    _acyclic: bool = field(default=True, init=False)

    @property
    def is_terminal(self) -> bool:
        return self._is_terminal

    @is_terminal.setter
    def is_terminal(self, value: bool):
        if value and (
            self.temperature is not None
            or self.max_tokens is not None
            or self.capture_name is not None
        ):
            raise ValueError(
                "RuleNode has a temperature, max_tokens, or capture_name, so it cannot be terminal"
            )
        self._is_terminal = value

    def children(self) -> list["GrammarNode"]:
        return [self.value]

    def _attrs(self) -> set[str]:
        return {"temperature", "max_tokens", "capture_name"}

    def lark_str(self, top: bool = False) -> str:
        rep = self.name
        if top:
            attrs = {}
            for attr in self._attrs():
                if getattr(self, attr) is not None:
                    attrs[attr] = getattr(self, attr)
            if attrs:
                rep += f"[{', '.join(f'{k}={json.dumps(v)}' for k, v in attrs.items())}]"
            rep += f": {self.value.lark_str(top=not isinstance(self.value, RuleNode))}"
        return rep


@dataclass(slots=True, eq=False)
class GenNode(RuleNode):
    value: RegexNode
    stop: RegexNode = RegexNode("")
    save_stop_text: bool = False

    @property
    def is_terminal(self) -> bool:
        return True

    def _attrs(self) -> set[str]:
        return super()._attrs() | {"stop", "save_stop_text"}


@dataclass(slots=True, eq=False)
class RuleRefNode(GrammarNode):
    target: Optional[RuleNode] = None

    @property
    def is_terminal(self) -> bool:
        if self.target is None:
            return False
        else:
            return self.target.is_terminal

    def lark_str(self, top: bool = False) -> str:
        if self.target is None:
            raise ValueError("RuleRefNode has no target")
        else:
            return self.target.name

    def __repr__(self) -> str:
        try:
            return self.lark_str()
        except ValueError:
            return super().__repr__()


def parse_tags(s: str) -> Union[GrammarNode, Function]:
    parts = cast(list[str], _tag_pattern.split(s))
    obj: GrammarNode = LiteralNode(parts.pop(0))
    is_tag = True
    for part in parts:
        if is_tag:
            obj += _tag_pool[part]
        else:
            obj += LiteralNode(part)
        is_tag = not is_tag
    return obj


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
            n.value = n.value.simplify()
            rules[name] = n

        elif isinstance(n, RuleRefNode):
            if n.target is None:
                raise ValueError("RuleRefNode has no target")
            add_node(n.target)

        for child in n.children():
            add_node(child)

    if isinstance(node, RuleNode) and node.name == "start":
        add_node(node)
    else:
        add_node(RuleNode("start", node))

    num_fix = 1
    while num_fix > 0:
        num_fix = 0
        for r in rules.values():
            if r.name != "start" and not r.is_terminal and r.value.is_terminal:
                try:
                    r.is_terminal = True
                except ValueError:
                    pass
                else:
                    num_fix += 1

    for name, r in rules.items():
        new_name = name.replace("-", "_")
        # convert fooBar_Baz to foo_bar_baz
        new_name = re.sub(r"([a-z])([A-Z])", r"\1_\2", new_name).lower()
        if r.is_terminal:
            new_name = new_name.upper()
        else:
            new_name = new_name.lower()
        if r.name != new_name:
            r.name = new_name

    return rules


def lark_serialize(node: GrammarNode) -> str:
    rules = resolve(node)
    res = "%llguidance {}\n\n"
    prev_nl = True
    for r in rules.values():
        s = r.lark_str(top=True)
        if not prev_nl and "\n" in s:
            res += "\n"
        res += s + "\n"
        prev_nl = "\n" in s
        if prev_nl:
            res += "\n"
    return res
