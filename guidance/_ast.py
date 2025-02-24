import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Union, cast

from ._parser import ByteParser, ByteParserException
from ._schema import JSONGrammar, LarkGrammar, LLGrammar

# to support the embedding of guidance functions inside Python f-strings we use tags with these delimiters
tag_start = "{{G|"  # start of a call tag
tag_end = "|G}}"  # end of a call tag
_tag_pool: dict[str, Union["Function", "GrammarNode"]] = (
    {}
)  # the Functions or GrammarNodes associated with the tags
_tag_pattern = re.compile(
    re.escape(tag_start) + r"([^\|]+)" + re.escape(tag_end)
)  # the pattern for matching call tags


def parse_tags(s: str) -> Union["GrammarNode", "Function"]:
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


@dataclass
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

        if isinstance(other, GrammarNode) and other.is_null:
            return self

        def __add__(model):
            return self(model) + other

        return Function(__add__, [], {})

    def __radd__(self, other):
        if not isinstance(other, (str, GrammarNode, Function)):
            return NotImplemented

        if isinstance(other, str):
            other = parse_tags(other)

        if isinstance(other, GrammarNode) and other.is_null:
            return self

        def __radd__(model):
            return self(model + other)

        return Function(__radd__, [], {})


@dataclass(eq=False)
class GrammarNode(Tagged):

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

        if self.is_null:
            return other

        if isinstance(other, Function):
            return other.__radd__(self)

        if other.is_null:
            return self

        return JoinNode([self, other])

    def __radd__(self, other) -> "GrammarNode":
        if not isinstance(other, (str, GrammarNode)):
            return NotImplemented

        if isinstance(other, str):
            other = parse_tags(other)

        if self.is_null:
            return other

        if isinstance(other, Function):
            return other.__add__(self)

        if other.is_null:
            return self

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
        return LLSerializer().serialize(self)


@dataclass(eq=False)
class LiteralNode(GrammarNode):
    value: str

    @property
    def is_null(self) -> bool:
        return self.value == ""


@dataclass(eq=False)
class RegexNode(GrammarNode):
    regex: str

    @property
    def is_null(self) -> bool:
        return self.regex == ""


@dataclass(eq=False)
class SelectNode(GrammarNode):
    alternatives: list[GrammarNode]

    @property
    def is_null(self) -> bool:
        return all(alt.is_null for alt in self.alternatives)

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


@dataclass(eq=False)
class JoinNode(GrammarNode):
    nodes: list[GrammarNode]

    @property
    def is_null(self) -> bool:
        return all(node.is_null for node in self.nodes)

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


@dataclass(eq=False)
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


@dataclass(eq=False)
class SubstringNode(GrammarNode):
    chunks: list[str]

    @property
    def is_terminal(self) -> bool:
        # TODO: true? technically a regex...
        return False


@dataclass(eq=False)
class RuleNode(GrammarNode):
    name: str
    value: GrammarNode
    capture: Optional[str] = None
    list_append: bool = False
    _temperature: Optional[float] = field(init=False, default=None)
    _max_tokens: Optional[int] = field(init=False, default=None)

    @property
    def temperature(self) -> Optional[float]:
        return self._temperature

    @temperature.setter
    def temperature(self, value: Optional[float]):
        if value is not None and not (
            self.value.is_terminal or isinstance(self.value, BaseSubgrammarNode)
        ):
            raise ValueError("RuleNode is not terminal, so it cannot have a temperature")
        self._temperature = value

    @property
    def max_tokens(self) -> Optional[int]:
        return self._max_tokens

    @max_tokens.setter
    def max_tokens(self, value: Optional[int]):
        if value is not None and not (
            self.value.is_terminal or isinstance(self.value, BaseSubgrammarNode)
        ):
            raise ValueError("RuleNode is not terminal, so it cannot have a max_tokens")
        self._max_tokens = value

    @property
    def is_terminal(self) -> bool:
        return (
            (self.capture is None and self._temperature is None and self._max_tokens is None)
            and self.value.is_terminal
            and not isinstance(self.value, BaseSubgrammarNode)
        )

    def children(self) -> list["GrammarNode"]:
        return [self.value]


@dataclass(eq=False)
class GenNode(RuleNode):
    value: RegexNode
    stop_regex: str = ""
    save_stop_text: Optional[str] = None

    def __post_init__(self):
        if self.save_stop_text is not None:
            raise NotImplementedError(
                "save_stop_text is currently unimplemented (will add back if there is demand)"
            )

    @property
    def is_terminal(self) -> bool:
        return (
            super(GenNode, self).is_terminal and self.stop_regex == "" and not self.save_stop_text
        )


@dataclass(eq=False)
class RuleRefNode(GrammarNode):
    target: Optional[RuleNode] = None

    @property
    def is_terminal(self) -> bool:
        # RuleRefNode should only ever be used to enable recursive rule definitions,
        # so it should never be terminal.
        return False


@dataclass(eq=False)
class BaseSubgrammarNode(GrammarNode):
    name: str

    @property
    def is_terminal(self) -> bool:
        return False


@dataclass(eq=False)
class SubgrammarNode(BaseSubgrammarNode):
    body: GrammarNode
    skip_regex: Optional[str] = None


@dataclass(eq=False)
class JsonNode(BaseSubgrammarNode):
    schema: Union[bool, dict[str, Any]]


class LLSerializer:
    def __init__(self):
        self.grammars: dict[str, Union[JSONGrammar, LarkGrammar]] = {}
        self.names: dict[BaseSubgrammarNode, str] = {}

    def serialize(self, node: GrammarNode) -> LLGrammar:
        if isinstance(node, BaseSubgrammarNode):
            self.visit(node)
        else:
            self.visit(SubgrammarNode("main", node))
        return LLGrammar(grammars=[self.grammars[name] for name in self.names.values()])

    def visit(self, node: BaseSubgrammarNode) -> str:
        if node in self.names:
            return self.names[node]

        name = node.name
        names = set(self.names.values())
        if name in names:
            i = 1
            while f"{name}_{i}" in names:
                i += 1
            name = f"{name}_{i}"

        if isinstance(node, SubgrammarNode):
            # Important: insert name BEFORE visiting body to avoid infinite recursion
            self.names[node] = name
            lark_grammar = LarkSerializer(self).serialize(node.body)
            if node.skip_regex:
                lark_grammar += f"\n%ignore /{node.skip_regex}/"
            self.grammars[name] = LarkGrammar(name=name, lark_grammar=lark_grammar)

        elif isinstance(node, JsonNode):
            # Important: insert name BEFORE visiting body to avoid infinite recursion
            self.names[node] = name
            self.grammars[name] = JSONGrammar(name=name, json_schema=node.schema)

        else:
            raise TypeError(f"Unknown subgrammar type: {node}")

        return name


class LarkSerializer:
    def __init__(self, ll_serializer: LLSerializer):
        self.ll_serializer = ll_serializer

        self.rules: dict[str, str] = {}
        self.names: dict[RuleNode, str] = {}

    def serialize(self, node: GrammarNode) -> str:
        if isinstance(node, RuleNode) and node.name == "start":
            self.visit(node)
        else:
            self.visit(RuleNode("start", node))

        res = "%llguidance {}\n\n"
        if "start" not in self.rules:
            assert "START" in self.rules
            res += "start: START\n"

        prev_nl = True
        for name in self.names.values():
            s = self.rules[name]
            if not prev_nl and "\n" in s:
                res += "\n"
            res += s + "\n"
            prev_nl = "\n" in s
            if prev_nl:
                res += "\n"

        return res

    def visit(self, node: GrammarNode, top=False) -> str:
        if isinstance(node, BaseSubgrammarNode):
            return f"@{self.ll_serializer.visit(node)}"

        if isinstance(node, RuleNode):
            if node in self.names:
                return self.names[node]

            name = self.normalize_name(node.name, node.is_terminal)
            names = set(self.names.values())
            if name in names:
                i = 1
                while f"{name}_{i}" in names:
                    i += 1
                name = f"{name}_{i}"
            self.names[node] = name
            res = name
            attrs = []
            if node.capture is not None:
                if node.capture != node.name or node.list_append:
                    capture_name = node.capture
                    if node.list_append:
                        capture_name = f"__LIST_APPEND:{capture_name}"
                    attrs.append(f"capture={json.dumps(capture_name)}")
                else:
                    attrs.append("capture")
            if node.temperature is not None:
                attrs.append(f"temperature={node.temperature}")
            if node.max_tokens is not None:
                attrs.append(f"max_tokens={node.max_tokens}")
            if isinstance(node, GenNode):
                if node.stop_regex:
                    attrs.append(f"stop=/{node.stop_regex}/")
            if attrs:
                res += f"[{', '.join(attrs)}]"
            res += ": " + self.visit(node.value, top=True)
            self.rules[name] = res
            return name

        if node.is_null:
            return '""'

        if isinstance(node, LiteralNode):
            return json.dumps(node.value)

        if isinstance(node, RegexNode):
            return f"/{node.regex}/"

        if isinstance(node, SelectNode):
            if top:
                return "\n     | ".join(self.visit(alt) for alt in node.alternatives)
            else:
                return "(" + " | ".join(self.visit(alt) for alt in node.alternatives) + ")"

        if isinstance(node, JoinNode):
            return " ".join(self.visit(n) for n in node.nodes if not n.is_null)

        if isinstance(node, RepeatNode):
            inner = self.visit(node.node)
            if not node.node.is_atomic:
                inner = f"({inner})"
            if (node.min, node.max) == (0, None):
                return f"{inner}*"
            if (node.min, node.max) == (1, None):
                return f"{inner}+"
            if (node.min, node.max) == (0, 1):
                return f"{inner}?"
            if node.max is None:
                return f"{inner}{{{node.min},}}"
            return f"{inner}{{{node.min},{node.max}}}"

        if isinstance(node, SubstringNode):
            return f'%regex {json.dumps({"substring_chunks": node.chunks}, indent=2)}'

        if isinstance(node, RuleRefNode):
            if node.target is None:
                raise ValueError("RuleRefNode has no target")
            return self.names[node.target]

        raise TypeError(f"Unknown node type: {node}")

    def normalize_name(self, name: str, terminal: bool) -> str:
        new_name = name.replace("-", "_")
        # convert fooBar_Baz to foo_Bar_Baz
        new_name = re.sub(r"([a-z])([A-Z])", r"\1_\2", new_name)
        if terminal:
            new_name = new_name.upper()
        else:
            new_name = new_name.lower()
        return new_name
