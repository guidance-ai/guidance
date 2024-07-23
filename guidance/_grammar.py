import re
import types

from typing import Any, TYPE_CHECKING, TypeVar, Union, cast, Optional

from . import _parser

_T = TypeVar("_T")

# to support the embedding of guidance functions inside Python f-strings we use tags with these delimiters
tag_start = "{{G|"  # start of a call tag
tag_end = "|G}}"  # end of a call tag
_call_pool: dict[str, "Function"] = {}  # the functions associated with the call tags
_tag_pattern = re.compile(
    re.escape(tag_start) + r"([^\|]+)" + re.escape(tag_end)
)  # the pattern for matching call tags


class StatefulException(Exception):
    """This is raised when we try and use the state of a grammar object like it was a live model.

    Note that eventually it would be nice to support stateful parser/grammar constructs directly, but
    such "parser combinators" cannot be run effciently in Python. So we use a traditional parser and
    grammar separation (hence the need for this exception)."""

    pass


class Function:
    """This is the abstract class representing all guidance functions.

    There are two main subclasses: GrammarFunction and RawFunction. GrammarFunctions
    represent guidance grammars that can be serialized and sent across the wire, while
    RawFunctions represent unconstrained native Python functions.
    """

    def __init__(self, name, value=None) -> None:
        self.name = name
        self.value = value

    def __str__(self):
        """Creates a string tag that can be used to retrieve this object."""

        # save the call in our call pool, ready to be run when it is attached to an LM object
        str_id = str(id(self))
        if str_id not in _call_pool:
            _call_pool[str_id] = self

        # return a string representation of this call so it can be combined with other strings/calls
        return tag_start + str_id + tag_end


class RawFunction(Function):
    __slots__ = ("f", "args", "kwargs")

    def __init__(self, f, args, kwargs):
        self.f = f
        self.args = args
        self.kwargs = kwargs

    def __call__(self, model):
        return self.f(model, *self.args, **self.kwargs)

    def __add__(self, other):

        # if we are joining with a string we use the string representation for ourselves
        if isinstance(other, str):
            return str(self) + other

        def __add__(model):
            model = self(model)
            if model is None:
                raise Exception(
                    f"The guidance function `{self.f.__name__}` did not return a model object! You need to return an updated model object at the end of your guidance function."
                )
            if isinstance(other, GrammarFunction):
                return model + other
            else:
                return other(model)

        return RawFunction(__add__, [], {})

    def __radd__(self, other):

        # if we are joining with a string we use the string representation for ourselves
        if isinstance(other, str):
            return other + str(self)

        def __radd__(model):
            if isinstance(other, GrammarFunction):
                model += other
            else:
                model = other(model)
            return self(model)

        return RawFunction(__radd__, [], {})


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


class GrammarFunction(Function):
    num_used_names = 0

    def __add__(self, value):

        # see if we have a string with calls or a simple string
        if isinstance(value, str) or isinstance(value, bytes):
            if isinstance(value, str) and re.search(_tag_pattern, value):
                return str(self) + value
            else:
                value = string(value)

        # see if we can keep building a stateless grammar
        if isinstance(value, GrammarFunction):
            return Join([self, value])

        # otherwise we let the stateful object handle things
        else:
            return value.__radd__(self)

    def __radd__(self, value):

        # see if we have a string with calls or a simple string
        if isinstance(value, str) or isinstance(value, bytes):
            if isinstance(value, str) and re.search(_tag_pattern, value):
                return value + str(self)
            else:
                value = string(value)

        # see if we can keep building a stateless grammar
        if isinstance(value, GrammarFunction):
            return Join([value, self])

        # otherwise we let the stateful object handle things
        else:
            return value.__add__(self)

    def __getitem__(self, value):
        raise StatefulException("GrammarFunctions can't access state!")

    def match(
        self,
        byte_string: Union[str, bytes],
        allow_partial: bool = False,
        raise_exceptions: bool = False,
    ) -> Union[Match, None]:
        if isinstance(byte_string, str):
            byte_string = byte_string.encode()
        parser = _parser.ByteParser(self)

        try:
            parser.consume_bytes(byte_string)
        except _parser.ByteParserException:
            if raise_exceptions:
                raise
            else:
                return None

        if not allow_partial and not parser.matched():
            return None

        if parser.matched():
            parser.force_done()

        return Match(*parser.get_captures(), partial=not parser.matched())  # type: ignore[misc]

    @staticmethod
    def _new_name():
        num_used = GrammarFunction.num_used_names

        a_ord = ord("a")

        # name the name in base 26 letter notation
        name = chr(a_ord + num_used % 26)
        if num_used >= 26:
            name = chr(a_ord + (num_used % 676) // 26) + name
            if num_used >= 676:
                name = chr(a_ord + (num_used % 17576) // 676) + name
                if num_used >= 17576:
                    name = chr(a_ord + (num_used % 456976) // 17576) + name

        GrammarFunction.num_used_names += 1

        return name

    def gbnf_string(self):
        used_names = set()
        names = {}
        lines = []
        root_name = self._rec_gbnf_string(lines, used_names, names)
        lines.append("root ::= " + root_name)
        return "\n".join(lines)

    def ll_serialize(self):
        return {"grammars": LLSerializer().run(self)}


class Terminal(GrammarFunction):
    def match_byte(self, byte):
        pass  # abstract

    @property
    def max_tokens(self):
        return 1000000000000


class Byte(Terminal):
    __slots__ = ("byte", "capture_name", "temperature")

    def __init__(self, byte):
        assert isinstance(byte, bytes)
        assert len(byte) == 1
        self.byte = byte
        self.capture_name = None
        self.temperature = -1

    @property
    def name(self):
        return str(self.byte)

    def __hash__(self):
        return self.byte[0]

    def __eq__(self, other):
        return isinstance(other, Byte) and self.byte[0] == other.byte[0]

    def __repr__(self) -> str:
        return str(self.byte)

    def __len__(self):
        return 1

    def match_byte(self, byte):
        return byte == self.byte

    @property
    def nullable(self):
        return False


class ByteRange(Terminal):
    __slots__ = ("byte_range", "capture_name", "temperature")

    def __init__(self, byte_range):
        assert isinstance(byte_range, bytes)
        assert len(byte_range) == 2
        self.byte_range = byte_range
        self.capture_name = None
        self.temperature = -1  # -1 means not set

    def match_byte(self, byte):
        return self.byte_range[0] <= byte[0] <= self.byte_range[1]

    @property
    def name(self):
        return str(self.byte_range)

    @name.setter
    def name(self, value):
        pass  # we ignore name changes

    @property
    def nullable(self):
        return False

    def __hash__(self):
        return self.byte_range[0] + 256 * self.byte_range[1]

    def __eq__(self, other):
        return (
            isinstance(other, ByteRange)
            and self.byte_range[0] == other.byte_range[0]
            and self.byte_range[1] == other.byte_range[1]
        )

    def __repr__(self) -> str:
        return str(self.byte_range)

    def __len__(self):
        return 1


class Null(Terminal):
    __slots__ = ("name", "capture_name")

    nullable = True

    def __init__(self):
        self.name = "ε"
        self.capture_name = None

    def __add__(self, other):
        # see if we have a string with calls or a simple string
        if isinstance(other, bytes):
            return string(other)
        elif isinstance(other, str):
            return str_to_grammar(other)

        # otherwise we return unchanged
        else:
            return other

    def __radd__(self, other):
        return self.__add__(
            other
        )  # left vs right makes no difference since we are null


class ModelVariable(GrammarFunction):
    """This represents a variable that will be read from the model object when this grammar is executed.

    Note that the name is the name of the attribute on the model object this node
    will get replaced with.
    """

    __slots__ = ("name", "capture_name")

    def __init__(self, name):
        self.name = name
        self.capture_name = None
        self.nullable = False


def replace_grammar_node(grammar, target, replacement):
    # Use a stack to keep track of the nodes to be visited
    stack = [grammar]
    visited_set = set()  # use set for O(1) lookups

    while stack:
        current = stack.pop()

        # Check if we have already visited this node
        if current in visited_set:
            continue
        visited_set.add(current)

        # We are done with this node if it's a terminal
        if isinstance(current, (Terminal, ModelVariable, Placeholder)):
            continue

        # Iterate through the node's values and replace target with replacement
        for i, value in enumerate(current.values):
            if value == target:
                current.values[i] = replacement
            else:
                stack.append(value)


# def replace_grammar_node(grammar, target, replacement, visited_set={}):

#     # see if we have already visited this node
#     if grammar in visited_set:
#         return
#     else:
#         visited_set[grammar] = True

#     # we are done if this is a terminal
#     if isinstance(grammar, (Terminal, ModelVariable)):
#         return

#     # replace all matching sub-nodes
#     for i,value in enumerate(grammar.values):
#         if value == target:
#             grammar.values[i] = replacement
#         else:
#             replace_grammar_node(value, target, replacement, visited_set)


def replace_model_variables(grammar, model, allowed_vars=None):
    """Replace all the ModelVariable nodes with their values in an iterative manner."""
    visited_set = set()
    stack = [
        (grammar, None, None)
    ]  # Stack stores tuples of (node, parent_node, child_index)
    replacements = []

    while stack:
        current, parent, child_index = stack.pop()

        # This node is being visited for the first time
        if current not in visited_set:
            visited_set.add(current)

            # If it's a terminal node, skip it
            if isinstance(current, Terminal):
                continue

            # Process non-terminal nodes in reverse order to maintain the depth-first order
            for i in reversed(range(len(current.values))):
                value = current.values[i]
                if isinstance(value, ModelVariable):
                    if allowed_vars is not None and value.name not in allowed_vars:
                        raise Exception(f"Invalid model variable name: {value.name}")
                    # Replace the ModelVariable with its value from 'model' (or the tokenizer if model does not have it)
                    # note we skip over attrs we don't have since we may be run twice, once on the model and once for the engine
                    if hasattr(model, value.name):
                        obj = model
                    elif hasattr(model, "tokenizer") and hasattr(
                        model.tokenizer, value.name
                    ):
                        obj = model.tokenizer
                    else:
                        obj = None
                    if obj is not None:
                        replacement_value = _wrap_as_grammar(getattr(obj, value.name))
                        replacements.append(
                            (current, i, value)
                        )  # Record the replacement
                        current.values[i] = replacement_value  # Perform the replacement
                else:
                    # If not ModelVariable, push onto the stack to process later
                    stack.append((value, current, i))

    return replacements


# def replace_model_variables(grammar, model, visited_set={}):
#     '''Replace all the ModelVariable nodes with their values.'''

#     # see if we have already visited this node
#     if grammar in visited_set:
#         return []
#     else:
#         visited_set[grammar] = True

#     # we are done if this is a terminal
#     if isinstance(grammar, Terminal):
#         return []

#     # replace all matching sub-nodes
#     replacements = []
#     for i,value in enumerate(grammar.values):
#         if isinstance(value, ModelVariable):
#             g = _wrap_as_grammar(getattr(model, value.name))
#             if value.commit_point:
#                 g = commit_point(g, hidden=value.hidden)
#             replacements.append((grammar, i, value))
#             grammar.values[i] = g
#         else:
#             replacements.extend(replace_model_variables(value, model, visited_set))
#     return replacements


def unreplace_model_variables(replacements):
    """This restores a grammar back to its original state, ready for another execution."""
    for grammar, i, orig_value in replacements:
        grammar.values[i] = orig_value


def _wrap_as_grammar(value):
    """This takes whatever value was given and tries to turn in into a guidance grammar."""

    # if it is already a valid grammar we have no need to wrap it
    if isinstance(value, GrammarFunction):
        return value

    # if it is already a valid grammar we have no need to wrap it
    if value is None:
        return Null()

    # we have a constant value
    if isinstance(value, (str, bytes)):
        return string(value)

    raise Exception("Can't wrap as a grammar!")


def commit_point(value, hidden=False):
    """Force the grammar to commit to a parse that includes this node once it can.

    Not that commit point nodes can be optionally hidden (in fact they are the only
    nodes that can be hidden since they are by definition not impacted by multiple possible
    inconsistent parses.)"""
    # TODO: assert that value is not empty since we don't yet support that
    if isinstance(value, str):
        value = string(value)
    if hidden:
        raise Exception("Hidden commit points are not supported!")
    # check if it serializes
    _ignore = LLSerializer().regex(value)
    return GenCommitPoint(value)


class Placeholder(GrammarFunction):
    __slots__ = tuple("nullable")

    def __init__(self):
        self.nullable = False


class Join(GrammarFunction):
    __slots__ = (
        "nullable",
        "values",
        "name",
        "capture_name",
        "max_tokens",
    )

    def __init__(
        self, values, name: Union[str, None] = None, max_tokens=100000000
    ) -> None:
        values = [
            string(v) if isinstance(v, (str, bytes)) else v for v in values
        ]  # wrap raw strings
        self.nullable = all(getattr(v, "nullable", False) for v in values)
        self.values = [v for v in values if not isinstance(v, Null)]
        self.name = name if name is not None else GrammarFunction._new_name()
        self.capture_name = None
        self.max_tokens = max_tokens

    def __repr__(self, indent="", done=None):
        if done is None:
            done = set()
        s = self.name.ljust(20) + " <- " + " ".join([v.name for v in self.values])
        s += (
            "        "
            + (f"capture_name={self.capture_name} " if self.capture_name else "")
            + (f"max_tokens={self.max_tokens}" if self.max_tokens < 100000 else "")
            + "\n"
        )
        done.add(self)
        for v in self.values:
            if v not in done and (isinstance(v, Join) or isinstance(v, Select)):
                s += v.__repr__(indent, done)
        return s


def quote_regex(value: str) -> str:
    assert isinstance(value, str)
    return re.sub(r"([\\+*?^$(){}\[\]\.|])", r"\\\1", value)


class Gen(Terminal):
    __slots__ = (
        "nullable",
        "body_regex",
        "stop_regex",
        "save_stop_text",
        "name",
        "capture_name",
        "_max_tokens",
    )

    def __init__(
        self,
        body_regex: str,
        stop_regex: str,
        name: Union[str, None] = None,
        save_stop_text: Optional[str] = None,
        max_tokens=100000000,
    ) -> None:
        self.nullable = False
        self.body_regex = body_regex
        self.stop_regex = stop_regex
        self.name = name if name is not None else GrammarFunction._new_name()
        self.capture_name = None
        self.save_stop_text = save_stop_text
        self._max_tokens = max_tokens
        self.temperature = -1

    @property
    def max_tokens(self):
        return self._max_tokens

    def __repr__(self, indent="", done=None, lbl="Gen"):
        if done is None:
            done = set()
        s = (
            self.name.ljust(20)
            + " <- "
            + lbl
            + " "
            + repr(self.body_regex)
            + " + "
            + repr(self.stop_regex)
        )
        s += (
            "        "
            + (f"capture_name={self.capture_name} " if self.capture_name else "")
            + (f"max_tokens={self.max_tokens}" if self.max_tokens < 100000 else "")
            + "\n"
        )
        done.add(self)
        return s


class Lexeme(Gen):
    __slots__ = ("contextual",)

    def __init__(
        self,
        body_regex: str,
        contextual: bool = False,
        name: Union[str, None] = None,
        max_tokens=100000000,
    ) -> None:
        super().__init__(body_regex, "", name=name, max_tokens=max_tokens)
        self.contextual = contextual

    def __repr__(self, indent="", done=None):
        return super().__repr__(indent, done, "Lex")


class GenCommitPoint(Gen):
    __slots__ = ("grammar",)

    def __init__(
        self,
        grammar: GrammarFunction,
        name: Union[str, None] = None,
        max_tokens=100000000,
    ) -> None:
        super().__init__("", "", name=name, max_tokens=max_tokens)
        self.grammar = grammar

    def __repr__(self, indent="", done=None):
        # TODO add grammar repr
        return super().__repr__(indent, done, "CommitPoint")


class Subgrammar(Gen):
    __slots__ = (
        "body",
        "skip_regex",
        "no_initial_skip",
    )

    def __init__(
        self,
        body: GrammarFunction,
        skip_regex: Optional[str] = None,
        no_initial_skip: bool = False,
        name: Union[str, None] = None,
        max_tokens=100000000,
    ) -> None:
        super().__init__(
            body_regex="",
            stop_regex="",
            name=name,
            max_tokens=max_tokens,
        )
        self.body = body
        self.skip_regex = skip_regex
        self.no_initial_skip = no_initial_skip

    def __repr__(self) -> str: # type: ignore[override]
        return self.name.ljust(20) + " <- " + self.body.name


class Select(GrammarFunction):
    __slots__ = (
        "nullable",
        "_values",
        "name",
        "capture_name",
        "max_tokens",
        "recursive",
    )

    def __init__(
        self, values, capture_name=None, name=None, max_tokens=10000000, recursive=False
    ) -> None:
        self.values = values
        self.name = name if name is not None else GrammarFunction._new_name()
        self.capture_name = capture_name
        self.max_tokens = max_tokens
        self.recursive = recursive

    @property
    def values(self):
        return self._values

    @values.setter
    def values(self, vals):
        self._values = [string(v) if isinstance(v, (str, bytes)) else v for v in vals]
        self.nullable = any(getattr(v, "nullable", False) for v in self._values)

    def __repr__(self, indent="", done=None):
        if done is None:
            done = set()
        s = self.name.ljust(20) + " <- " + " | ".join([v.name for v in self.values])
        s += (
            "        "
            + (f"max_tokens={self.max_tokens}" if self.max_tokens < 100000 else "")
            + "\n"
        )
        done.add(self)
        for v in self.values:
            if v not in done and (isinstance(v, Join) or isinstance(v, Select)):
                s += v.__repr__(indent, done)
        return s


def string(value: Union[str, bytes]) -> Union[Null, Join]:
    if isinstance(value, str):
        b = bytes(value, encoding="utf8")
    elif isinstance(value, bytes):
        b = value
    else:
        raise Exception("Must pass bytes or str to the string() function!")
    if len(value) == 0:
        return Null()
    else:
        return Join([Byte(b[i : i + 1]) for i in range(len(b))], name=str(b))


def select(
    options: list[_T], name=None, list_append=False, recurse=False, skip_checks=False
) -> Union[Select, _T]:
    """Choose between a set of options.

    This function constrains the next generation from the LLM to be one of the
    given `options`.
    If the list only has a single element, then that value can be returned
    immediately, without calling the LLM.

        >>> lm += select(["Temeraire", "Redoutable", "Bucentaure"], name="my_selection")
        >>> print(lm["my_selection"])
        Temeraire

    Parameters
    ----------
    name : str or None
        If this is not None then the the results of the generation will be saved as a variable on
        the Model object (so you can access the result as `lm["var_name"]`).

    options : list
        The set of available choices for the next generation

    list_append : bool
        If this is True then the results saved to `lm[name]` will not be written directly but rather appended
        to a list (if no list with the current name is present one will be created). This is useful for
        building lists inside python loops.

    recurse : bool
        Indicate whether multiple choices should be made. This is useful for tasks such as
        building up integers digit by digit: `select(options=list(range(10)), recurse=True)`

    skip_checks: bool
        Whether or not to perform sanity checks on the supplied options, to ensure
        that more obscure errors do not appear later.
    """
    # TODO: allow for returning the probabilites of the selected item
    # TODO: also the full probabilites distribution over all items. We can implement this using the prob of the selected item by repeating the call, removing the selected item each time
    if not skip_checks:
        for i, value in enumerate(options):
            assert not isinstance(
                value, RawFunction
            ), "You cannot select between stateful functions in the current guidance implementation!"
            assert not isinstance(
                value, types.FunctionType
            ), "Did you pass a function without calling it to select? You need to pass the results of a called guidance function to select."
            if isinstance(value, int) or isinstance(value, float):
                options[i] = str(value)  # type: ignore[assignment]

    # set up list append var saving if requested
    if list_append:
        name = "__LIST_APPEND:" + name

    if recurse:
        node = Select([], capture_name=name, recursive=True)
        if "" in options:
            # if we have an empty option, 'node + v' also covers the case of 'v' itself
            # thus, we don't have to add 'options' (except for the empty string)
            node.values = [node + v for v in options if v != ""] + [""]
        else:
            node.values = [node + v for v in options if v != ""] + options
        return node
    else:
        if len(options) == 1 and name is None:
            return options[0]
        else:
            return Select(options, capture_name=name, recursive=False)


def byte_range(low, high) -> ByteRange:
    return ByteRange(low + high)


# def ignore_placeholders(value):
#     if not isinstance(value, Join): # don't double wrap
#         value = Join([value]) # this ensures we capture what we want, and not something surprisingly self_recursive
#     value.ignore_placeholders = True
#     return value


def capture(value, name):
    # if log_probs:
    #     name += ":__LOG_PROBS"
    if not (isinstance(value, Join) and len(value.values) == 1):  # don't double wrap
        value = Join(
            [value]
        )  # this ensures we capture what we want, and not something surprisingly self_recursive
    value.capture_name = name
    return value


def token_limit(value, max_tokens: int):
    _rec_token_limit(value, max_tokens)
    return value


def _rec_token_limit(grammar, max_tokens: int):
    if grammar.max_tokens > max_tokens and not isinstance(grammar, Terminal):
        if getattr(
            grammar, "recursive", False
        ):  # only restrict recursive selects, otherwise we would block all ways to complete the grammar
            grammar.max_tokens = max_tokens
            for value in getattr(
                grammar, "values", []
            ):  # restrict recursive selects recursive nodes
                if not isinstance(value, Terminal):
                    value.max_tokens = max_tokens
        if hasattr(grammar, "values"):
            for g in grammar.values:
                _rec_token_limit(g, max_tokens)


def with_temperature(value, temperature):
    """This sets the sampling temperature to be used for the given portion of the grammar.

    Note that if the grammar passed to us already has some portions with a temperature
    setting in place, those settings will not be overridden.
    """
    _re_with_temperature(value, temperature, {})
    return value


def _re_with_temperature(grammar, temperature, visited_set):

    # don't go down the same path twice
    if grammar in visited_set:
        return
    visited_set[grammar] = True

    # if getattr(grammar, "temperature", 100000000) > temperature:
    if (
        isinstance(grammar, Terminal) and not isinstance(grammar, Null) and grammar.temperature < 0
    ):  # only need to set temp for terminals
        grammar.temperature = temperature
    elif getattr(grammar, "temperature", 100000000) > temperature and hasattr(
        grammar, "values"
    ):
        for g in grammar.values:
            _re_with_temperature(g, temperature, visited_set)


# def model_variable(name):
#     return ModelVariable(name)


def active_role_end() -> ModelVariable:
    return ModelVariable("active_role_end")


def eos_token() -> ModelVariable:
    return ModelVariable("eos_token")


def bos_token() -> ModelVariable:
    return ModelVariable("bos_token")


_null_grammar = string("")


# def char_range(low, high):
#     low_bytes = bytes(low, encoding="utf8")
#     high_bytes = bytes(high, encoding="utf8")
#     if len(low_bytes) > 1 or len(high_bytes) > 1:
#         raise Exception("We don't yet support multi-byte character ranges!")
#     return ByteRange(low_bytes + high_bytes)
def str_to_grammar(value: str):
    is_id = False
    parts = re.split(_tag_pattern, value)

    # we have no embedded objects
    if len(parts) == 1:
        return string(value)

    # if we have embedded objects we have to convert the string to a grammar tree
    else:
        partial_grammar: Any = _null_grammar
        # lm.suffix = ""
        for i, part in enumerate(parts):
            # if i < len(parts) - 1:
            #     lm.suffix = parts[i+1]
            if is_id:
                call = _call_pool[part]
                if isinstance(call, GrammarFunction):
                    partial_grammar += _call_pool[part]
                else:
                    partial_grammar = RawFunction(
                        lambda lm, g, call: call(lm + g),
                        partial_grammar,
                        _call_pool[part],
                    )
                    # lm += partial_grammar
                    # lm = _call_pool[part](lm)
                    # partial_grammar = _null_grammar
            elif part != "":
                partial_grammar += string(part)
            is_id = not is_id
    return partial_grammar


def _is_string_literal(node: GrammarFunction):
    if isinstance(node, Byte):
        return True
    if isinstance(node, Join):
        return all(_is_string_literal(v) for v in node.values)
    return False


class LLSerializer:
    def __init__(self) -> None:
        self.nodes: list[dict] = []
        self.curr_grammar = {
            "nodes": self.nodes,
            "rx_nodes": [],
        }
        self.grammars = [self.curr_grammar]
        self.node_id_cache: dict[GrammarFunction, int] = {}
        self.todo: list[GrammarFunction] = []
        self.grammar_id_cache: dict[Subgrammar, int] = {}
        self.grammar_todo: list[Subgrammar] = []

        self.regex_id_cache: dict[GrammarFunction, int] = {}

    def _add_regex_json(self, json):
        id = len(self.curr_grammar["rx_nodes"])
        self.curr_grammar["rx_nodes"].append(json)
        return id

    def _add_regex(self, key: str, val):
        return self._add_regex_json({key: val})

    def _regex_or(self, nodes: list[GrammarFunction]):
        if len(nodes) == 1:
            return self.regex_id_cache[nodes[0]]
        else:
            return self._add_regex("Or", [self.regex_id_cache[v] for v in nodes])

    def regex(self, node: GrammarFunction):
        """
        Serialize node as regex. Throws if impossible.
        """

        node0 = node
        todo = [node]
        pending: set[GrammarFunction] = set()

        def node_finished(node: GrammarFunction):
            return node not in pending and node in self.regex_id_cache

        def all_finished(nodes):
            return all(node_finished(v) for v in nodes)

        def add_todo(n: GrammarFunction):
            if n in pending:
                raise ValueError(
                    "GrammarFunction is recursive - cannot serialize as regex: "
                    + n.__repr__()
                )
            todo.append(n)

        def add_todos(nodes):
            for n in nodes:
                add_todo(n)

        def check_unserializable_attrs(node: GrammarFunction):
            if not isinstance(node, Terminal):
                for v in getattr(node, "values", []):
                    # Only check one level deeper as we'll soon be processing the children
                    if isinstance(v, Terminal):
                        check_unserializable_attrs(v)

            if getattr(node, "capture_name", None) is not None:
                raise ValueError(
                    f"Regex serialization does not support captures. Node: {node.__repr__()}"
                )
            if getattr(node, "temperature", -1) >= 0:
                raise ValueError(
                    f"Regex serialization does not support temperature. Node: {node.__repr__()}"
                )

        while todo:
            node = todo.pop()
            check_unserializable_attrs(node)

            if node in self.regex_id_cache:
                continue
            if isinstance(node, Select) and node.values:
                with_node = []
                without_node = []
                for v in node.values:
                    if (
                        isinstance(v, Join)
                        and len(v.values) == 2
                        and v.values[0] is node
                    ):
                        with_node.append(v.values[1])
                    else:
                        without_node.append(v)
                if not all_finished(with_node) or not all_finished(without_node):
                    add_todo(node)
                    pending.add(node)
                    add_todos(with_node)
                    add_todos(without_node)
                    continue
                #print(with_node, without_node)
                if len(with_node) == 0:
                    # non-recursive
                    res = self._regex_or(without_node)
                elif len(without_node) == 1 and isinstance(without_node[0], Null):
                    # zero_or_more()
                    inner = self._regex_or(with_node)
                    res = self._add_regex("Repeat", [inner, 0, None])
                elif with_node == without_node:
                    # one_or_more()
                    inner = self._regex_or(with_node)
                    res = self._add_regex("Repeat", [inner, 1, None])
                else:
                    raise ValueError(
                        "Cannot detect structure of recursive Select as regex: "
                        + node.__repr__()
                    )
            elif isinstance(node, Join):
                if all(isinstance(v, Byte) for v in node.values):
                    literal = [cast(Byte, v).byte[0] for v in node.values]
                    try:
                        literal_ = bytes(literal).decode("utf-8", errors="strict")
                        res = self._add_regex("Literal", literal_)
                    except UnicodeDecodeError:
                        res = self._add_regex("ByteLiteral", literal)
                else:
                    if not all_finished(node.values):
                        add_todo(node)
                        pending.add(node)
                        add_todos(node.values)
                        continue
                    res = self._add_regex(
                        "Concat", [self.regex_id_cache[v] for v in node.values]
                    )
            elif isinstance(node, Byte):
                res = self._add_regex("Byte", node.byte[0])
            elif isinstance(node, ByteRange):
                byteset = [0, 0, 0, 0, 0, 0, 0, 0]
                for idx in range(256):
                    if node.match_byte(bytes([idx])):
                        byteset[idx // 32] |= 1 << (idx % 32)
                res = self._add_regex("ByteSet", byteset)
            elif isinstance(node, Null):
                res = self._add_regex_json("EmptyString")
            elif isinstance(node, Lexeme):
                res = self._add_regex("Regex", node.body_regex)
            else:
                raise ValueError("Cannot serialize as regex: " + node.__repr__())
            if node in pending:
                pending.remove(node)
            self.regex_id_cache[node] = res

        assert not pending
        return self.regex_id_cache[node0]

    def grammar(self, grammar: Subgrammar):
        if grammar in self.grammar_id_cache:
            return self.grammar_id_cache[grammar]
        id = len(self.grammars)
        self.grammar_id_cache[grammar] = id
        self.grammars.append(
            {
                "greedy_skip_rx": grammar.skip_regex,
                "nodes": [],
                "rx_nodes": [],
            }
        )
        self.grammar_todo.append(grammar)
        return id

    def node(self, node: GrammarFunction):
        if node in self.node_id_cache:
            return self.node_id_cache[node]
        id = len(self.nodes)
        self.node_id_cache[node] = id
        self.nodes.append({})
        self.todo.append(node)
        return id

    def process(self, node: GrammarFunction):
        obj: dict[str, Any] = {}
        if isinstance(node, Select):
            obj = {
                "Select": {
                    "among": [self.node(v) for v in node.values],
                }
            }
        elif isinstance(node, Join):
            if all(isinstance(v, Byte) for v in node.values):
                literal = b"".join(cast(Byte, v).byte for v in node.values)
                obj = {
                    "String": {
                        "literal": literal.decode("utf-8", errors="strict"),
                    }
                }
            else:
                obj = {
                    "Join": {
                        "sequence": [self.node(v) for v in node.values],
                    }
                }
        elif isinstance(node, Lexeme):
            obj = {
                "Lexeme": {
                    "rx": node.body_regex,
                    "contextual": node.contextual,
                }
            }
        elif isinstance(node, Subgrammar):
            obj = {
                "GenGrammar": {
                    "grammar": self.grammar(node),
                    "stop_rx": node.stop_regex,
                    "no_initial_skip": node.no_initial_skip,
                    "temperature": node.temperature if node.temperature >= 0 else None,
                }
            }
        elif isinstance(node, GenCommitPoint):
            obj = {
                "Gen": {
                    "body_rx": self.regex(node.grammar),
                    "stop_rx": "",
                    "lazy": False, # TODO this should be True
                    "temperature": node.temperature if node.temperature >= 0 else None,
                }
            }
        elif isinstance(node, Gen):
            obj = {
                "Gen": {
                    "body_rx": node.body_regex,
                    "stop_rx": node.stop_regex,
                    "lazy": node.stop_regex != "",
                    "stop_capture_name": node.save_stop_text,
                    "temperature": node.temperature if node.temperature >= 0 else None,
                }
            }
        elif isinstance(node, ByteRange):
            # TODO: maybe raise a warning in this case, as user should probably be using a larger
            # GenCommitPoint?
            obj = {
                "Gen": {
                    "body_rx": self.regex(node),
                    "stop_rx": "",
                    "lazy": True,
                    "temperature": node.temperature if node.temperature >= 0 else None,
                }
            }
        elif isinstance(node, Byte):
            obj = {
                "String": {
                    "literal": node.byte.decode("utf-8", errors="strict"),
                }
            }
        elif isinstance(node, Null):
            obj = {
                "String": {
                    "literal": "",
                }
            }
        else:
            raise Exception("Unknown node type:", type(node))
        tp = next(iter(obj))
        inner: dict = obj[tp]
        if (capture_name:=getattr(node, "capture_name")):
            inner["capture_name"] = capture_name
        # Names on nodes are mostly useless
        # if getattr(node, "name", None):
        #     inner["name"] = node.name
        if (max_tokens:=getattr(node, "max_tokens")) and max_tokens < 1000000:
            inner["max_tokens"] = max_tokens
        self.nodes[self.node(node)] = obj

    def run_grammar(self, node: GrammarFunction):
        assert self.todo == []
        id = self.node(node)
        assert id == 0
        while self.todo:
            node = self.todo.pop()
            self.process(node)

    def run(self, node: GrammarFunction):
        # avoid top-level node being a String
        if _is_string_literal(node):
            node = Select([node])
        self.run_grammar(node)
        while self.grammar_todo:
            grammar = self.grammar_todo.pop()
            self.curr_grammar = self.grammars[self.grammar(grammar)]
            self.nodes = cast(list[dict], self.curr_grammar["nodes"])
            self.node_id_cache = {}
            self.regex_id_cache = {}
            self.run_grammar(grammar.body)
        return self.grammars
