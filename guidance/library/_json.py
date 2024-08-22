from dataclasses import dataclass
from enum import Enum
from functools import cached_property
from json import dumps as json_dumps
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Type,
    TYPE_CHECKING,
    Union,
)

try:
    import jsonschema
    import pydantic
except ImportError:
    if TYPE_CHECKING:
        raise

from ..llg import GrammarWithLexer, TopLevelGrammar, NodeJSON, NodeLexeme

from .._guidance import guidance

from .._grammar import (
    GrammarFunction,
    with_temperature,
    ReferencingGrammarFunction,
)
from ._pydantic import pydantic_to_json_schema
from ._subgrammar import lexeme, subgrammar


GrammarNode = Tuple[int, GrammarWithLexer]
Grammar = Union[str, GrammarNode]


class GrammarBuilder:

    def __init__(self) -> None:
        self.top_grammar: TopLevelGrammar = {
            "grammars": [],
            "max_tokens": None,
        }
        self._placeholder: NodeJSON = {"String": {"literal": "placeholder"}}
        self._strings: Dict[str, GrammarNode] = {}

    def string(self, s: str) -> GrammarNode:
        if s in self._strings:
            return self._strings[s]
        r = (len(self.nodes), self.curr_grammar)
        self.nodes.append({"String": {"literal": s}})
        self._strings[s] = r
        return r

    def lexeme(
        self,
        rx: str,
        contextual: Optional[bool] = None,
        max_tokens: Optional[int] = None,
        json_string: Optional[bool] = None,
        json_raw: Optional[bool] = None,
        json_allowed_escapes: Optional[str] = None,
    ) -> GrammarNode:
        r = (len(self.nodes), self.curr_grammar)
        props: NodeLexeme = {
            "rx": rx,
            "contextual": contextual,
            "max_tokens": max_tokens,
            "temperature": None,
            "json_string": json_string,
            "json_raw": json_raw,
            "json_allowed_escapes": json_allowed_escapes,
        }
        self.nodes.append({"Lexeme": props})
        return r

    def _child_nodes(self, options: Sequence[Grammar]) -> List[int]:
        r: List[int] = []
        for e in options:
            if isinstance(e, str):
                r.append(self.string(e)[0])
            else:
                assert e[1] is self.curr_grammar
                r.append(e[0])
        return r

    def select(self, options: Sequence[Grammar]) -> GrammarNode:
        ch = self._child_nodes(options)
        r = (len(self.nodes), self.curr_grammar)
        self.nodes.append({"Select": {"among": ch}})
        return r

    def join(self, values: Sequence[Grammar]) -> GrammarNode:
        ch = self._child_nodes(values)
        r = (len(self.nodes), self.curr_grammar)
        self.nodes.append({"Join": {"sequence": ch}})
        return r

    def optional(self, value: Grammar) -> GrammarNode:
        return self.select([value, ""])

    def zero_or_more(self, elt: GrammarNode) -> GrammarNode:
        p = self.placeholder()
        self.set_placeholder(p, self.select(["", self.join([p, elt])]))
        return p

    def placeholder(self) -> GrammarNode:
        r = (len(self.nodes), self.curr_grammar)
        self.nodes.append(self._placeholder)
        return r

    def is_placeholder(self, node: GrammarNode) -> bool:
        return self.nodes[node[0]] is self._placeholder

    def set_placeholder(
        self, placeholder: GrammarNode, node: GrammarNode, direct: bool = False
    ) -> None:
        self._child_nodes([placeholder[0:2], node])  # validate
        assert self.nodes[placeholder[0]] is self._placeholder, "placeholder already set"
        if direct:
            self.nodes[placeholder[0]] = self.nodes[node[0]]
        else:
            body: NodeJSON = {"Join": {"sequence": [node[0]]}}
            self.nodes[placeholder[0]] = body

    def set_start_node(self, node: GrammarNode) -> None:
        self.set_placeholder((0, self.curr_grammar), node)

    def add_grammar(
        self,
        skip: Optional[str] = None,
        contextual: bool = True,
        allow_initial_skip: bool = False,
        no_forcing: bool = False,
        allow_invalid_utf8: bool = False,
    ) -> "GrammarBuilder":
        # TODO blow caches
        self.curr_grammar: GrammarWithLexer = {
            "nodes": [],
            "greedy_skip_rx": skip,
            "contextual": contextual,
            "rx_nodes": [],
            "allow_initial_skip": allow_initial_skip,
            "no_forcing": no_forcing,
            "allow_invalid_utf8": allow_invalid_utf8,
        }
        self.nodes: List[NodeJSON] = self.curr_grammar["nodes"]
        self.top_grammar["grammars"].append(self.curr_grammar)
        top, _ = self.placeholder()
        assert top == 0
        return self

    def finalize(self) -> TopLevelGrammar:
        for g in self.top_grammar["grammars"]:
            for n in g["nodes"]:
                assert n is not self._placeholder, "unresolved placeholder"
        return self.top_grammar


def _to_compact_json(target: Any) -> str:
    # See 'Compact Encoding':
    # https://docs.python.org/3/library/json.html
    # Since this is ultimately about the generated
    # output, we don't need to worry about pretty printing
    # and whitespace
    return json_dumps(target, separators=(",", ":"))


@dataclass
class CompileOptions:
    compact: bool = False
    validate: bool = True


def _to_compact_json(target: Any) -> str:
    # See 'Compact Encoding':
    # https://docs.python.org/3/library/json.html
    # Since this is ultimately about the generated
    # output, we don't need to worry about pretty printing
    # and whitespace
    return json_dumps(target, separators=(",", ":"))


class Keyword(str, Enum):
    ANYOF = "anyOf"
    ONEOF = "oneOf"
    ALLOF = "allOf"
    REF = "$ref"
    CONST = "const"
    ENUM = "enum"
    TYPE = "type"
    PATTERN = "pattern"
    MIN_LENGTH = "minLength"
    MAX_LENGTH = "maxLength"


KEYS = {member.value for member in Keyword}


def looks_like_schema(map: Mapping[str, Any]) -> bool:
    return (
        Keyword.TYPE in map
        or Keyword.ANYOF in map
        or Keyword.ALLOF in map
        or Keyword.ONEOF in map
        or Keyword.ENUM in map
        or Keyword.CONST in map
        or Keyword.REF in map
    )


DEFS_KEYS = {"$defs", "definitions"}

IGNORED_KEYS = {
    "$schema",
    "$id",
    "$comment",
    "title",
    "description",
    "default",
    "examples",
    "discriminator",  # we hope it's part of the grammar anyways
    "id",  # doesn't do anything?
    "required",  # TODO: implement and remove from ignored list
}

TYPE_SPECIFIC_KEYS = {
    "array": {"items", "prefixItems", "minItems", "maxItems"},
    "object": {"properties", "additionalProperties"},
}


CHAR_REGEX = r"(\\([\"\\\/bfnrt]|u[a-fA-F0-9]{4})|[^\"\\\x00-\x1F\x7F])"


def validate_json_node_keys(node: Mapping[str, Any]) -> None:
    keys = set(node.keys())
    valid_keys = KEYS | IGNORED_KEYS | DEFS_KEYS
    if Keyword.TYPE in node:
        valid_keys |= TYPE_SPECIFIC_KEYS.get(node[Keyword.TYPE], set())
    invalid_keys = keys - valid_keys
    if invalid_keys:
        # don't include schema in exception message as it may be large
        raise ValueError(f"JSON schema had keys that could not be processed: {invalid_keys}")


class Compiler:

    def __init__(self, options: CompileOptions) -> None:
        self.builder = GrammarBuilder()
        self.options = options

    def run(
        self,
        schema: Mapping[str, Any],
    ) -> None:
        # Raises jsonschema.exceptions.SchemaError or ValueError
        # if schema is not valid
        # For large schemas this takes much longer than grammar generation
        if self.options.validate:
            import jsonschema

            jsonschema.Draft202012Validator.check_schema(schema)

        self._gen_json_any_cache: Optional[GrammarNode] = None

        self.builder.add_grammar(
            skip=(None if self.options.compact else r"[\x20\x0A\x0D\x09]+"),
            allow_initial_skip=False,
        )

        self.definitions: Mapping[str, Tuple[Any, GrammarNode]] = {}
        for dk in DEFS_KEYS:
            if dk in schema:
                assert len(self.definitions) == 0, "Found duplicate definitions"
                self._build_definitions(schema[dk])

        root = self._gen_json(schema)
        self.builder.set_start_node(root)

        for schema, target_definition in self.definitions.values():
            self.builder.set_placeholder(target_definition, self._gen_json(schema))

    def _build_definitions(self, raw_definitions: Mapping[str, Any]) -> None:
        self.definitions = {
            ref: (schema, self.builder.placeholder()) for ref, schema in raw_definitions.items()
        }

    def _get_definition(
        self,
        reference: str,
    ) -> GrammarNode:
        target_definition = None
        for dk in DEFS_KEYS:
            ref_start = f"#/{dk}/"
            if reference.startswith(ref_start):
                target_name = reference[len(ref_start) :]
                _, target_definition = self.definitions[target_name]
        assert target_definition is not None
        return target_definition

    def _gen_json(self, json_schema: Mapping[str, Any]) -> GrammarNode:
        validate_json_node_keys(json_schema)

        if Keyword.ANYOF in json_schema:
            return self._process_anyOf(json_schema[Keyword.ANYOF])

        # TODO approximate oneOf as anyOf
        if Keyword.ONEOF in json_schema:
            return self._process_anyOf(json_schema[Keyword.ONEOF])

        if Keyword.ALLOF in json_schema:
            allof_list = json_schema[Keyword.ALLOF]
            if len(allof_list) != 1:
                raise ValueError("Only support allOf with exactly one item")
            return self._gen_json(allof_list[0])

        if Keyword.REF in json_schema:
            return self._get_definition(json_schema[Keyword.REF])

        if Keyword.CONST in json_schema:
            # TODO: can we support a whitespace-flexible version of this?
            return self.builder.string(_to_compact_json(json_schema[Keyword.CONST]))

        if Keyword.ENUM in json_schema:
            return self._process_enum(options=json_schema[Keyword.ENUM])

        if Keyword.TYPE in json_schema:
            target_type = json_schema[Keyword.TYPE]
            if target_type == "null":
                return self.builder.string("null")
            if target_type == "boolean":
                return self.builder.select(["true", "false"])
            if target_type == "integer":
                return self.json_int
            if target_type == "number":
                return self.json_number
            if target_type == "string":
                return self._gen_json_string(
                    min_length=json_schema.get(Keyword.MIN_LENGTH, 0),
                    max_length=json_schema.get(Keyword.MAX_LENGTH, None),
                    regex=json_schema.get(Keyword.PATTERN, None),
                )
            if target_type == "array":
                return self._gen_json_array(
                    prefix_items_schema=json_schema.get("prefixItems", []),
                    item_schema=json_schema.get("items", True),
                    min_items=json_schema.get("minItems", 0),
                    max_items=json_schema.get("maxItems"),
                )
            if target_type == "object":
                return self._gen_json_object(
                    properties=json_schema.get("properties", {}),
                    additional_properties=json_schema.get("additionalProperties", True),
                )
            raise ValueError(f"Unsupported type in schema: {target_type}")

        return self._gen_json_any()

    def _process_anyOf(
        self,
        anyof_list: Sequence[Mapping[str, Any]],
    ) -> GrammarNode:
        options = [self._gen_json(json_schema=item) for item in anyof_list]
        return self.builder.select(options)

    def _process_enum(self, options: Sequence[Mapping[str, Any]]) -> GrammarNode:
        # TODO: can we support a whitespace-flexible version of this?
        return self.builder.select([_to_compact_json(opt) for opt in options])

    def _gen_json_object(
        self,
        properties: Mapping[str, Any],
        additional_properties: Union[bool, Mapping[str, Any]],
    ) -> GrammarNode:
        if additional_properties is True:
            # True means that anything goes
            additional_properties = {}

        lm: List[Grammar] = ["{"]
        if properties:
            lm += self._process_properties(properties=properties)
            if additional_properties is not False:
                lm.append(",")
        if additional_properties is not False:
            lm.append(self._process_additional_properties(additional_properties))
        lm.append("}")
        return self.join(*lm)

    def _process_properties(
        self,
        properties: Mapping[str, Any],
    ) -> List[Grammar]:
        res: List[Grammar] = []
        properties_added = 0
        for name, property_schema in properties.items():
            res.append('"' + name + '"')
            res.append(":")
            res.append(self._gen_json(property_schema))
            properties_added += 1
            if properties_added < len(properties):
                res.append(",")
        return res

    def join(self, *args: Grammar) -> GrammarNode:
        return self.builder.join(args)

    def _process_additional_properties(
        self,
        additional_properties: Mapping[str, Any],
    ) -> GrammarNode:
        item = self.join(self.json_simple_string, ":", self._gen_json(additional_properties))
        return self.builder.optional(self._sequence(item))

    def _sequence(self, item: GrammarNode) -> GrammarNode:
        return self.builder.join([self.builder.zero_or_more(self.join(item, ",")), item])

    @cached_property
    def json_int(self) -> GrammarNode:
        return self.builder.lexeme(r"-?(?:0|[1-9][0-9]*)")

    @cached_property
    def json_number(self) -> GrammarNode:
        # TODO: guidance had a select() including json_int() here
        return self.builder.lexeme(r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)")

    @cached_property
    def json_simple_string(self) -> GrammarNode:
        return self.builder.lexeme('"' + CHAR_REGEX + "*" + '"')

    def _gen_json_string(
        self,
        min_length: int = 0,
        max_length: Union[int, None] = None,
        regex: Optional[str] = None,
    ) -> GrammarNode:
        if min_length == 0 and max_length is None and regex is None:
            return self.json_simple_string

        if regex is not None:
            if min_length > 0 or max_length is not None:
                msg = (
                    "If a pattern is specified for a JSON "
                    "string, minLength and maxLength must be "
                    "left unspecified."
                )
                raise ValueError(msg)
            # the regex has implicit ^...$ anyways
            regex = regex.lstrip("^").rstrip("$")
            return self.builder.lexeme(regex, json_string=True)

        max_length_str = f"{max_length}" if max_length is not None else ""
        return self.builder.lexeme('"' + CHAR_REGEX + f"{{{min_length},{max_length_str}}}" + '"')

    def _gen_json_array(
        self,
        prefix_items_schema: Sequence[Mapping[str, Any]],
        item_schema: Union[bool, Mapping[str, Any]],
        min_items: int,
        max_items: Optional[int],
    ) -> GrammarNode:
        if item_schema is True:
            # True means that anything goes
            item_schema = {}

        if len(prefix_items_schema) < min_items and item_schema is False:
            raise ValueError(
                f"PrefixItems has too few elements ({len(prefix_items_schema)}) to"
                f" satisfy minItems ({min_items}) but no extra items were allowed"
            )

        if max_items is not None and max_items < min_items:
            raise ValueError(f"maxItems ({max_items}) can't be less than minItems ({min_items})")

        required_items: List[GrammarNode] = []
        optional_items: List[GrammarNode] = []

        # If max_items is None, we can add an infinite tail of items later
        n_to_add = max(len(prefix_items_schema), min_items) if max_items is None else max_items

        item_schema_compiled = None if item_schema is False else self._gen_json(item_schema)
        for i in range(n_to_add):
            if i < len(prefix_items_schema):
                item = self._gen_json(prefix_items_schema[i])
            elif item_schema is not False:
                assert item_schema_compiled
                item = item_schema_compiled
            else:
                assert i >= min_items
                break

            if i < min_items:
                required_items.append(item)
            else:
                optional_items.append(item)

        if max_items is None and item_schema is not False:
            # Add an infinite tail of items
            assert item_schema_compiled
            optional_items.append(self._sequence(item_schema_compiled))

        lm: List["Grammar"] = ["["]

        if required_items:
            first, *rest = required_items
            lm.append(first)
            for item in rest:
                lm.append(",")
                lm.append(item)

        if optional_items:
            # This is a bit subtle and would not be required if not for prefixItems -- the previous
            # must be present before the next one may be added, meaning we have nested optionals:
            # (first optional(,second optional(,third (optional(,...)))))
            first, *rest = optional_items
            tail: Grammar = ""
            for item in reversed(rest):
                tail = self.builder.optional(self.join(",", item, tail))
            tail = self.join(first, tail)

            if required_items:
                lm.append(self.builder.optional(self.join(",", tail)))
            else:
                lm.append(self.builder.optional(tail))

        lm.append("]")

        return self.join(*lm)

    def _gen_json_any(self) -> GrammarNode:
        if self._gen_json_any_cache is None:
            self._gen_json_any_cache = self.builder.placeholder()
            self.builder.set_placeholder(
                self._gen_json_any_cache,
                self.builder.select(
                    [
                        self._gen_json(json_schema={"type": "null"}),
                        self._gen_json(json_schema={"type": "boolean"}),
                        self._gen_json(json_schema={"type": "integer"}),
                        self._gen_json(json_schema={"type": "number"}),
                        self._gen_json(json_schema={"type": "string"}),
                        # Recursive cases
                        self._gen_json(
                            json_schema={
                                "type": "array",
                                "items": True,
                            },
                        ),
                        self._gen_json(
                            json_schema={
                                "type": "object",
                                "additionalProperties": True,
                            }
                        ),
                    ]
                ),
            )
        return self._gen_json_any_cache


@guidance(stateless=True)
def json(
    lm,
    name: Optional[str] = None,
    *,
    schema: Union[
        None,
        Mapping[str, Any],
        Type["pydantic.BaseModel"],
        "pydantic.TypeAdapter",
    ] = None,
    compact: bool = False,
    temperature: float = 0.0,
    max_tokens: int = 100000000,
):
    """Generate valid JSON according to the supplied JSON schema or `pydantic` model.

    Not all parts of `JSON schema <https://json-schema.org/>`_ are supported. Indeed some parts
    (such as bounds on numbers) cannot really be supported in the context of LLM generation.

    Using a JSON schema:

        >>> schema = ''{ "type": "object", "properties": { "a" : {"type": "integer"} } }'
        >>> schema_obj = json.loads(schema)
        >>> lm += json(name="generated_object", schema=schema_obj)
        >>> print(json.loads(lm["generated_object"]))
        { 'a' : 2 }

    Using a ``pydantic.BaseModel``:

        >>> class Schema(BaseModel):
        ...     b: bool
        >>> lm += json(name="generated_object", schema=Schema)
        >>> print(json.loads(lm["generated_object"]))
        { 'b' : False }

    Using a ``pydantic.TypeAdapter``:

        >>> schema = TypeAdapter(list[int])
        >>> lm += json(name="generated_object", schema=schema)
        >>> print(json.loads(lm["generated_object"]))
        [1, 2, 3]

    Parameters
    ----------

    name : str or None
        If this is not None then the the results of the generation will be saved as a variable on
        the Model object (so you can access the result as ``lm["var_name"]``).

    schema : Union[None, Mapping[str, Any], Type[pydantic.BaseModel], pydantic.TypeAdapter]
        One of:
            - None, in which case any valid JSON will be generated
            - A JSON schema object. This is a JSON schema string which has been passed to ``json.loads()``
            - A subclass of ``pydantic.BaseModel``
            - An instance of ``pydantic.TypeAdapter``

    compact : bool
        If True, the generated JSON will be forced to be compact (no whitespace).
        If False, output will be whitespace-flexible (i.e. decided by the model).
    """
    if isinstance(schema, Mapping):
        # Raises jsonschema.exceptions.SchemaError or ValueError
        # if schema is not valid
        jsonschema.validators.Draft202012Validator.check_schema(schema)
    elif schema is None:
        schema = {}
    else:
        schema = pydantic_to_json_schema(schema)

    options = CompileOptions(compact=compact)
    c = Compiler(options)
    c.run(schema)
    json_top_level_grammar = c.builder.finalize()
    assert len(json_top_level_grammar["grammars"]) == 1

    json_grammar = ReferencingGrammarFunction(name=name)
    json_grammar.grammars = json_top_level_grammar["grammars"][0]["nodes"]

    return lm + with_temperature(
        json_grammar,
        temperature=temperature,
    )
