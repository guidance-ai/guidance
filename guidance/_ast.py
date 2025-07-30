import copy
import inspect
import json
import re
import textwrap
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import cached_property
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Iterator,
    Literal,
    Optional,
    Sequence,
    TypedDict,
    TypeVar,
    Union,
    cast,
)

import pydantic
from llguidance import LLMatcher
from pydantic import Base64Bytes
from typing_extensions import assert_never

from ._parser import ByteParser, ByteParserException
from .trace import OutputAttr

if TYPE_CHECKING:
    from .models._base import Interpreter, State

# to support the embedding of guidance functions inside Python f-strings we use tags with these delimiters
tag_start = "{{G|"  # start of a call tag
tag_end = "|G}}"  # end of a call tag
_tag_pool: dict[str, Union["Function", "GrammarNode"]] = {}  # the Functions or GrammarNodes associated with the tags
_tag_pattern = re.compile(re.escape(tag_start) + r"([^\|]+)" + re.escape(tag_end))  # the pattern for matching call tags


def _parse_tags(s: str) -> Union["GrammarNode", "Function"]:
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
        return "<guidance.Match object; captures=" + str(self.captures) + "; partial=" + str(self.partial) + ">"


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
            other = _parse_tags(other)

        if isinstance(other, GrammarNode) and other.is_null:
            return self

        def __add__(model):
            return self(model) + other

        return Function(__add__, [], {})

    def __radd__(self, other):
        if not isinstance(other, (str, GrammarNode, Function)):
            return NotImplemented

        if isinstance(other, str):
            other = _parse_tags(other)

        if isinstance(other, GrammarNode) and other.is_null:
            return self

        def __radd__(model):
            return self(model + other)

        return Function(__radd__, [], {})


S = TypeVar("S", bound="State")


class ASTNode(ABC):
    @abstractmethod
    def _run(self, interpreter: "Interpreter[S]", **kwargs) -> Iterator[OutputAttr]:
        pass

    def simplify(self) -> "ASTNode":
        return self


@dataclass
class RoleStart(ASTNode):
    role: str

    def _run(self, interpreter: "Interpreter[S]", **kwargs) -> Iterator[OutputAttr]:
        return interpreter._role_start(self, **kwargs)


@dataclass
class RoleEnd(ASTNode):
    role: str

    def _run(self, interpreter: "Interpreter[S]", **kwargs) -> Iterator[OutputAttr]:
        return interpreter._role_end(self, **kwargs)


@dataclass
class ImageBlob(ASTNode):
    data: Base64Bytes

    def _run(self, interpreter: "Interpreter[S]", **kwargs) -> Iterator[OutputAttr]:
        return interpreter.image_blob(self, **kwargs)


@dataclass
class ImageUrl(ASTNode):
    url: str

    def _run(self, interpreter: "Interpreter[S]", **kwargs) -> Iterator[OutputAttr]:
        return interpreter.image_url(self, **kwargs)


@dataclass
class AudioBlob(ASTNode):
    data: Base64Bytes

    def _run(self, interpreter: "Interpreter[S]", **kwargs) -> Iterator[OutputAttr]:
        return interpreter.audio_blob(self, **kwargs)


class GenAudio(ASTNode):
    def __init__(self, kwargs: dict[str, Any]):
        self.kwargs = kwargs

    def _run(self, interpreter: "Interpreter[S]", **kwargs) -> Iterator[OutputAttr]:
        return interpreter.gen_audio(self, **kwargs)


@dataclass(frozen=True)
class GrammarNode(Tagged, ASTNode):
    @property
    def is_null(self) -> bool:
        """
        If this returns true, then this node matches empty string and empty string only.
        """
        return False

    @property
    def is_allowed_in_lark_terminal(self) -> bool:
        """
        If this returns true, then this node will be compiled down to a regular expression.
        It cannot be recursive.
        """
        return all(child.is_allowed_in_lark_terminal for child in self.children())

    @property
    def is_allowed_in_lark_rule_with_attrs(self) -> bool:
        """
        If this returns true, then this node can be used as a Lark rule with attributes.
        """
        # Typically, not being allowed in terminal implies that a node is not allowed in a rule with attributes,
        # however this is notably false for subgrammars
        return self.is_allowed_in_lark_terminal

    def simplify(self) -> "GrammarNode":
        return self

    def children(self) -> Sequence["GrammarNode"]:
        return ()

    def __add__(self, other) -> "GrammarNode":
        if not isinstance(other, (str, GrammarNode)):
            return NotImplemented

        if isinstance(other, str):
            other = _parse_tags(other)

        if self.is_null:
            return other

        if isinstance(other, Function):
            return other.__radd__(self)

        if other.is_null:
            return self

        return JoinNode((self, other))

    def __radd__(self, other) -> "GrammarNode":
        if not isinstance(other, (str, GrammarNode)):
            return NotImplemented

        if isinstance(other, str):
            other = _parse_tags(other)

        if self.is_null:
            return other

        if isinstance(other, Function):
            return other.__add__(self)

        if other.is_null:
            return self

        return JoinNode((other, self))

    def __getitem__(self, key):
        raise StatefulException("GrammarNodes can't access state!")

    def match(
        self,
        byte_string: Union[str, bytes],
        allow_partial: bool = False,
        raise_exceptions: bool = False,
        enforce_max_tokens: bool = True,
    ) -> Union[Match, None]:
        if isinstance(byte_string, str):
            byte_string = byte_string.encode()
        parser = ByteParser(self.ll_grammar(enforce_max_tokens=enforce_max_tokens))

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

    def ll_grammar(self, enforce_max_tokens: bool = True) -> str:
        lark_str = LarkSerializer(enforce_max_tokens=enforce_max_tokens).serialize(self.simplify())
        return lark_str

    def _llguidance_validate(self) -> None:
        """Validate the grammar with `llguidance` and warn about any issues."""
        is_err, messages = LLMatcher.validate_grammar_with_warnings(self.ll_grammar())
        if is_err:
            raise ValueError(messages[0])
        else:
            # this will warn about oneOf coercion, and any other unsupported features if lenient is enabled
            for message in messages:
                warnings.warn(message, UserWarning, stacklevel=2)


@dataclass(frozen=True)
class LiteralNode(GrammarNode):
    value: str

    @property
    def is_null(self) -> bool:
        return self.value == ""

    def _run(self, interpreter: "Interpreter[S]", **kwargs) -> Iterator[OutputAttr]:
        return interpreter.text(self, **kwargs)


@dataclass(frozen=True)
class SpecialToken(GrammarNode):
    text: Optional[str] = None
    id: Optional[int] = None
    range: Optional[tuple[int, int]] = None

    def __post_init__(self):
        if [self.text, self.id, self.range].count(None) != 2:
            raise ValueError("Exactly one of text, id, or range must be set")

    def format(self) -> str:
        if self.text is not None:
            return f"<{self.text}>"
        if self.id is not None:
            return f"<[{self.id}]>"
        if self.range is not None:
            return f"<[{self.range[0]}-{self.range[1]}]>"
        raise ValueError("SpecialToken must have either text, id, or range set")

    @property
    def is_allowed_in_lark_terminal(self) -> bool:
        return False

    @property
    def is_allowed_in_lark_rule_with_attrs(self) -> bool:
        return True

    def _run(self, interpreter: "Interpreter[S]", **kwargs) -> Iterator[OutputAttr]:
        # Just use grammar -- I don't think we need a special case for this
        return interpreter.grammar(self, **kwargs)


@dataclass(frozen=True)
class RegexNode(GrammarNode):
    regex: Optional[str]

    def _run(self, interpreter: "Interpreter[S]", **kwargs) -> Iterator[OutputAttr]:
        return interpreter.regex(self, **kwargs)


@dataclass(frozen=True)
class SelectNode(GrammarNode):
    alternatives: tuple[GrammarNode, ...]

    @property
    def is_null(self) -> bool:
        return all(alt.is_null for alt in self.alternatives)

    def simplify(self) -> "GrammarNode":
        if self.is_null:
            return LiteralNode("")
        alternatives = tuple(alt.simplify() for alt in self.alternatives if not alt.is_null)
        if len(alternatives) == 1:
            node = alternatives[0]
        else:
            node = SelectNode(alternatives)

        if any(alt.is_null for alt in self.alternatives):
            return RepeatNode(node, 0, 1)
        return node

    def children(self) -> Sequence["GrammarNode"]:
        return self.alternatives

    def _run(self, interpreter: "Interpreter[S]", **kwargs) -> Iterator[OutputAttr]:
        return interpreter.select(self, **kwargs)


@dataclass(frozen=True)
class JoinNode(GrammarNode):
    nodes: tuple[GrammarNode, ...]

    @property
    def is_null(self) -> bool:
        return all(node.is_null for node in self.nodes)

    def simplify(self) -> "GrammarNode":
        if self.is_null:
            return LiteralNode("")
        nodes = tuple(node.simplify() for node in self.nodes if not node.is_null)
        if len(nodes) == 1:
            return nodes[0]
        return JoinNode(nodes)

    def children(self) -> Sequence["GrammarNode"]:
        return self.nodes

    def _run(self, interpreter: "Interpreter[S]", **kwargs) -> Iterator[OutputAttr]:
        return interpreter.join(self, **kwargs)


@dataclass(frozen=True)
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

    def children(self) -> Sequence["GrammarNode"]:
        return (self.node,)

    def simplify(self) -> GrammarNode:
        return RepeatNode(self.node.simplify(), self.min, self.max)

    def _run(self, interpreter: "Interpreter[S]", **kwargs) -> Iterator[OutputAttr]:
        return interpreter.repeat(self, **kwargs)


@dataclass(frozen=True)
class SubstringNode(GrammarNode):
    chunks: tuple[str, ...]

    @property
    def is_allowed_in_lark_terminal(self) -> bool:
        # this can be used as part of bigger regexes
        return True

    def _run(self, interpreter: "Interpreter[S]", **kwargs) -> Iterator[OutputAttr]:
        return interpreter.substring(self, **kwargs)


# This creates a name for the given grammar node (value), which can be referenced
# via RuleRefNode (or directly).
# In Lark syntax this results in approx. "{name}: {value}"
# This can either Lark rule (non-terminal) or terminal definition
# (meaning name can be upper- or lowercase).
@dataclass(frozen=True)
class RuleNode(GrammarNode):
    name: str
    value: GrammarNode
    capture: Optional[str] = None
    list_append: bool = False
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    stop: Optional[Union[RegexNode, LiteralNode]] = None
    suffix: Optional[LiteralNode] = None
    stop_capture: Optional[str] = None
    lazy: bool = False

    def __post_init__(self) -> None:
        if (
            # Note: capture is very intentionally missing from this list, as it's not like the other attributes
            self.temperature is not None
            or self.max_tokens is not None
            or self.stop is not None
            or self.suffix is not None
            or self.stop_capture is not None
            or self.lazy
        ) and not self.value.is_allowed_in_lark_rule_with_attrs:
            raise ValueError("RuleNode is not terminal, so it cannot have a temperature, max_tokens, or stop condition")

    @property
    def is_allowed_in_lark_terminal(self) -> bool:
        check_self = (
            self.capture is None
            and self.temperature is None
            and self.max_tokens is None
            and self.stop is None
            and self.suffix is None
            and self.stop_capture is None
            and not self.lazy
        )
        return check_self and super().is_allowed_in_lark_terminal

    def children(self) -> tuple[GrammarNode]:
        return (self.value,)

    def _run(self, interpreter: "Interpreter[S]", **kwargs) -> Iterator[OutputAttr]:
        return interpreter.rule(self, **kwargs)


@dataclass(frozen=True, eq=False)
class RuleRefNode(GrammarNode):
    target: Optional[RuleNode] = field(default=None, init=False)

    def set_target(self, target: RuleNode) -> None:
        if self.target is not None:
            raise ValueError("RuleRefNode target already set")
        # Side-step frozen=True to set target
        object.__setattr__(self, "target", target)

    @property
    def is_allowed_in_lark_terminal(self) -> bool:
        # RuleRefNode should only ever be used to enable recursive rule definitions,
        # so it should never be terminal.
        return False

    def _run(self, interpreter: "Interpreter[S]", **kwargs) -> Iterator[OutputAttr]:
        if self.target is None:
            raise ValueError("RuleRefNode target not set")
        return interpreter.rule(self.target)


@dataclass(frozen=True)
class BaseSubgrammarNode(GrammarNode):
    @property
    def is_allowed_in_lark_terminal(self) -> bool:
        return False

    @property
    def is_allowed_in_lark_rule_with_attrs(self) -> bool:
        # Typically, not being allowed in terminal implies that a node is not allowed in a rule with attributes,
        # however this is notably false for subgrammars
        return True


@dataclass(frozen=True)
class SubgrammarNode(BaseSubgrammarNode):
    body: GrammarNode
    skip_regex: Optional[str] = None

    def _run(self, interpreter: "Interpreter[S]", **kwargs) -> Iterator[OutputAttr]:
        return interpreter.subgrammar(self, **kwargs)


class LLGJsonCompileOptions(TypedDict):
    # defaults to ","
    item_separator: Optional[str]
    # defaults to ":"
    key_separator: Optional[str]
    # defaults to None - depends on whitespace_flexible
    whitespace_pattern: Optional[str]
    # defaults to true (r"[\x20\x0A\x0D\x09]+"); if false, no whitespace is allowed
    whitespace_flexible: Optional[bool]
    # defaults to false
    coerce_one_of: Optional[bool]
    # ignore unimplemented keywords; defaults to false
    lenient: Optional[bool]


@dataclass(frozen=True, eq=False)
class JsonNode(BaseSubgrammarNode):
    schema: Optional[dict[str, Any]] = None
    llg_options: Optional[LLGJsonCompileOptions] = None

    def _run(self, interpreter: "Interpreter[S]", **kwargs) -> Iterator[OutputAttr]:
        return interpreter.json(self, **kwargs)

    @cached_property
    def _llguidance_json(self) -> dict[str, Any]:
        if self.schema is None:
            # The user did not pass a schema. Let's assume that they want an object
            # (this should match the behavior of most remote providers)
            schema = {"type": "object"}
        else:
            # shallow copy is ok
            schema = copy.copy(self.schema)

        if self.llg_options is not None:
            # Maybe TODO: let LLGJsonCompileOptions be non-total
            # and update the schema with any present options
            # (in case x-guidance was already set with some options)
            schema["x-guidance"] = self.llg_options
        return schema


@dataclass(frozen=True, eq=False)
class LarkNode(BaseSubgrammarNode):
    lark_grammar: str

    def _run(self, interpreter: "Interpreter[S]", **kwargs) -> Iterator[OutputAttr]:
        return interpreter.lark(self, **kwargs)


@dataclass
class ToolDefinition:
    callable: Callable
    name: str
    description: str
    args: type[pydantic.BaseModel]

    @classmethod
    def from_callable(cls, callable: Callable) -> "ToolDefinition":
        signature = inspect.signature(callable)
        args = {}
        for name, param in signature.parameters.items():
            if param.kind not in {
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            }:
                raise ValueError(f"Unsupported parameter kind: {param.kind.description}")
            args[name] = param.annotation if param.annotation is not inspect.Parameter.empty else Any
        return cls(
            callable=callable,
            name=callable.__name__,
            description=(callable.__doc__ or "").strip(),
            args=pydantic.create_model(
                callable.__name__,
                __config__=pydantic.ConfigDict(extra="forbid"),
                **{name: (annotation, ...) for name, annotation in args.items()},
            ),
        )


@dataclass(frozen=True)
class ToolCallNode(ASTNode):
    tools: dict[str, ToolDefinition]
    tool_choice: Literal["auto", "required"] = "auto"
    parallel_tool_calls: bool = False
    plaintext_regex: Optional[str] = None

    @classmethod
    def from_tools(
        cls,
        tools: list[Union[callable, ToolDefinition]],
        tool_choice: Literal["auto", "required"] = "auto",
        parallel_tool_calls: bool = False,
        plaintext_regex: Optional[str] = None,
    ) -> "ToolCallNode":
        tool_defs = {}
        for tool in tools:
            if isinstance(tool, ToolDefinition):
                tool_def = tool
            elif callable(tool):
                tool_def = ToolDefinition.from_callable(tool)
            else:
                raise ValueError(f"Unsupported tool type: {type(tool)}")
            if tool_def.name in tool_defs:
                raise ValueError(f"Duplicate tool name: {tool_def.name}")
            tool_defs[tool_def.name] = tool_def
        return cls(
            tools=tool_defs,
            tool_choice=tool_choice,
            parallel_tool_calls=parallel_tool_calls,
            plaintext_regex=plaintext_regex,
        )

    def __post_init__(self):
        if not self.tools:
            raise ValueError("ToolCallNode must have at least one tool")

    def _run(self, interpreter: "Interpreter[S]", **kwargs) -> Iterator[OutputAttr]:
        return interpreter.tool_call(self, **kwargs)


class LarkSerializer:
    def __init__(self, enforce_max_tokens: bool = True):
        self.enforce_max_tokens = enforce_max_tokens
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
        if isinstance(node, RuleNode):
            if node in self.names:
                return self.names[node]

            name = self.normalize_name(node.name, node.is_allowed_in_lark_terminal)
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
                capture_name = node.capture
                if node.list_append:
                    capture_name = f"__LIST_APPEND:{capture_name}"
                if capture_name != name:
                    attrs.append(f"capture={json.dumps(capture_name)}")
                else:
                    attrs.append("capture")
            if node.temperature is not None:
                attrs.append(f"temperature={node.temperature}")
            if self.enforce_max_tokens and node.max_tokens is not None:
                attrs.append(f"max_tokens={node.max_tokens}")
            if node.stop:
                attrs.append(f"stop={self.visit(node.stop)}")
            if node.suffix:
                attrs.append(f"suffix={self.visit(node.suffix)}")
            if node.stop_capture:
                attrs.append(f"stop_capture={json.dumps(node.stop_capture)}")
            if node.lazy:
                attrs.append("lazy")
            if attrs:
                res += f"[{', '.join(attrs)}]"

            res += ": "
            target = node.value
            if isinstance(target, JsonNode):
                res += "%json " + json.dumps(target._llguidance_json, indent=2)
            elif isinstance(target, LarkNode):
                # TODO: we can't decide whether or not to enforce max tokens here easily.
                # We could in principle parse the grammar and/or use a regex?
                res += f"%lark {{\n{textwrap.indent(target.lark_grammar, '  ').strip()}\n}}"
            elif isinstance(target, SubgrammarNode):
                lark_grammar = LarkSerializer(enforce_max_tokens=self.enforce_max_tokens).serialize(target.body)
                if target.skip_regex:
                    lark_grammar += f"\n%ignore /{target.skip_regex}/"
                res += f"%lark {{\n{textwrap.indent(lark_grammar, '  ').strip()}\n}}"
            elif isinstance(target, GrammarNode):
                if (
                    not isinstance(target, RuleNode)
                    and target.is_allowed_in_lark_terminal
                    and not node.is_allowed_in_lark_terminal
                ):
                    """
                    If the RHS could be written as a terminal, but the presence of attributes on the LHS
                    prevents it, we wrap the RHS in a new rule like so:
                    ```
                    rule[attr]: TERMINAL | TERMINAL | TERMINAL
                    ```
                    gets rewritten as:
                    ```
                    rule[attr]: RULE
                    RULE: TERMINAL | TERMINAL | TERMINAL
                    ```
                    In particular, this lets us ensure that large alternations are handled as single lexemes
                    rather than a choice between multiple lexemes. Keeping the number of individual lexemes
                    to a minimum is important for performance.
                    Indeed, llguidance imposes a limit to maintain performance: see issue #1320
                    """
                    target = RuleNode(
                        name=node.name,
                        value=target,
                    )
                res += self.visit(target.simplify(), top=True)
            else:
                if TYPE_CHECKING:
                    assert_never(target)
                raise TypeError(f"Unknown rule value type: {target}")
            self.rules[name] = res
            return name
        if node.is_null:
            return '""'

        if isinstance(node, LiteralNode):
            return json.dumps(node.value)

        if isinstance(node, SpecialToken):
            return node.format()

        if isinstance(node, RegexNode):
            rx = node.regex
            if rx is None:
                rx = "(?s:.*)"
            return self.regex(rx)

        if isinstance(node, SelectNode):
            if top:
                return "\n     | ".join(self.visit(alt) for alt in node.alternatives)
            else:
                return "(" + " | ".join(self.visit(alt) for alt in node.alternatives) + ")"

        if isinstance(node, JoinNode):
            return " ".join(self.visit(n) for n in node.nodes if not n.is_null)

        if isinstance(node, RepeatNode):
            inner = self.visit(node.node)
            if isinstance(node.node, (JoinNode, RepeatNode)):
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
            return f"%regex {json.dumps({'substring_chunks': node.chunks}, indent=2)}"

        if isinstance(node, RuleRefNode):
            if node.target is None:
                raise ValueError("RuleRefNode has no target")
            return self.visit(node.target)

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

    def regex(self, pattern: str) -> str:
        escaped = re.sub(r"(?<!\\)/", r"\/", pattern).replace("\n", "\\n")
        return f"/{escaped}/"
