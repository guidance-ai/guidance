from json import dumps as json_dumps, loads as json_loads
from enum import Enum
import math
from typing import (
    Any,
    Callable,
    Mapping,
    Optional,
    Sequence,
    Union,
    Type,
    TYPE_CHECKING,
    cast,
)
import warnings
import referencing
from collections import defaultdict
import urllib.parse
import functools

from .._guidance import guidance
from ..library import optional, sequence
from ..library._regex_utils import rx_int_range, rx_float_range

from .._grammar import GrammarFunction, select, with_temperature, Not, And, quote_regex
from ._pydantic import pydantic_to_json_schema
from ._subgrammar import as_regular_grammar, lexeme, subgrammar

try:
    import jsonschema
    import pydantic
except ImportError:
    if TYPE_CHECKING:
        raise

JSONSchema = Union[bool, dict[str, Any]]

DRAFT202012_RESERVED_KEYWORDS = {
    # Anchors and References
    '$anchor',
    '$dynamicAnchor',
    '$dynamicRef',
    '$id',
    '$recursiveAnchor',
    '$recursiveRef',
    '$ref',
    '$schema',
    '$vocabulary',

    # Schema Structure and Combining Schemas
    '$defs',
    'allOf',
    'anyOf',
    'definitions',
    'dependencies',
    'dependentRequired',
    'dependentSchemas',
    'else',
    'if',
    'not',
    'oneOf',
    'then',

    # Validation Keywords for Any Instance Type
    'const',
    'enum',
    'type',

    # Validation Keywords for Numeric Instances
    'exclusiveMaximum',
    'exclusiveMinimum',
    'maximum',
    'minimum',
    'multipleOf',

    # Validation Keywords for Strings
    'format',
    'maxLength',
    'minLength',
    'pattern',

    # Validation Keywords for Arrays
    'contains',
    'items',
    'maxContains',
    'maxItems',
    'minContains',
    'minItems',
    'prefixItems',
    'uniqueItems',

    # Validation Keywords for Objects
    'additionalProperties',
    'maxProperties',
    'minProperties',
    'patternProperties',
    'properties',
    'propertyNames',
    'required',
    'unevaluatedItems',
    'unevaluatedProperties',

    # Metadata Keywords
    '$comment',
    'default',
    'deprecated',
    'description',
    'examples',
    'readOnly',
    'title',
    'writeOnly',

    # Content Validation
    'contentEncoding',
    'contentMediaType',
    'contentSchema',
}

class JSONType(str, Enum):
    NULL = "null"
    BOOLEAN = "boolean"
    INTEGER = "integer"
    NUMBER = "number"
    STRING = "string"
    ARRAY = "array"
    OBJECT = "object"

class Keyword(str, Enum):
    ANYOF = "anyOf"
    ALLOF = "allOf" # Note: Partial support. Only supports exactly one item.
    ONEOF = "oneOf" # Note: Partial support. This is converted to anyOf.
    ID = "$id"
    REF = "$ref"
    CONST = "const"
    ENUM = "enum"
    TYPE = "type"

class NumberKeywords(str, Enum):
    MINIMUM = "minimum"
    MAXIMUM = "maximum"
    EXCLUSIVE_MINIMUM = "exclusiveMinimum"
    EXCLUSIVE_MAXIMUM = "exclusiveMaximum"

class StringKeywords(str, Enum):
    PATTERN = "pattern"
    FORMAT = "format"
    MIN_LENGTH = "minLength"
    MAX_LENGTH = "maxLength"

class ArrayKeywords(str, Enum):
    PREFIX_ITEMS = "prefixItems"
    ITEMS = "items"
    MIN_ITEMS = "minItems"
    MAX_ITEMS = "maxItems"

class ObjectKeywords(str, Enum):
    PROPERTIES = "properties"
    ADDITIONAL_PROPERTIES = "additionalProperties"
    REQUIRED = "required"

TYPE_SPECIFIC_KEYWORDS = {
    JSONType.INTEGER: NumberKeywords,
    JSONType.NUMBER: NumberKeywords,
    JSONType.STRING: StringKeywords,
    JSONType.ARRAY: ArrayKeywords,
    JSONType.OBJECT: ObjectKeywords,
}

IGNORED_KEYS = {
    "$anchor",
    "$defs",
    "$schema",
    "id",
    "$comment",
    "title",
    "default",
    "definitions",
    "description",
    "examples",
}

# discriminator is part of OpenAPI 3.1, not JSON Schema itself
# https://json-schema.org/blog/posts/validating-openapi-and-json-schema
# TODO: While ignoring this key shouldn't lead to invalid outputs, forcing
# the model to choose the value of the marked field before other fields
# are generated (statefully or statelessly) would reduce grammar ambiguity
# and possibly improve quality.
IGNORED_KEYS.add("discriminator")

WHITESPACE = {b" ", b"\t", b"\n", b"\r"}
VALID_KEYS = set(Keyword) | set(NumberKeywords) | set(StringKeywords) | set(ArrayKeywords) | set(ObjectKeywords) | IGNORED_KEYS

FORMAT_PATTERNS: dict[str, Optional[str]] = {
    # https://json-schema.org/understanding-json-schema/reference/string#built-in-formats
    # Dates and times
    "date-time": (
        r'(?P<date>[0-9]{4}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12][0-9]|3[01]))'
        r'[tT]'
        r'(?P<time>'
            r'(?:[01][0-9]|2[0-3]):[0-5][0-9]:(?:[0-5][0-9]|60)'
            r'(?P<time_fraction>\.[0-9]+)?'
            r'(?P<time_zone>[zZ]|[+-](?:[01][0-9]|2[0-3]):[0-5][0-9])'
        r')'
    ),
    "time": (
        r'(?:[01][0-9]|2[0-3]):[0-5][0-9]:(?:[0-5][0-9]|60)'
        r'(?P<time_fraction>\.[0-9]+)?'
        r'(?P<time_zone>[zZ]|[+-](?:[01][0-9]|2[0-3]):[0-5][0-9])'
    ),
    "date": r'[0-9]{4}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12][0-9]|3[01])',
    "duration": (
        r'P'                                     # Start with 'P'
        r'(?:'                                   # Non-capturing group for main alternatives
            r'(?P<dur_date>'                     # Named group for date duration
                r'(?:'                           # Non-capturing group for date components
                    r'(?P<dur_year>'             # Named group for years
                        r'[0-9]+Y'                  # One or more digits followed by 'Y'
                        r'(?:'                   # Optional month
                            r'[0-9]+M'              # One or more digits followed by 'M'
                            r'(?:[0-9]+D)?'         # Optional days
                        r')?'
                    r')'
                    r'|'                         # OR
                    r'(?P<dur_month>'            # Named group for months
                        r'[0-9]+M'                  # One or more digits followed by 'M'
                        r'(?:[0-9]+D)?'             # Optional days
                    r')'
                    r'|'                         # OR
                    r'(?P<dur_day>'              # Named group for days
                        r'[0-9]+D'                  # One or more digits followed by 'D'
                    r')'
                r')'
                r'(?:'                           # Optional time
                    r'T'                         # Time starts with 'T'
                    r'(?:'                       # Non-capturing group for time components
                        r'(?P<dur_hour>'         # Named group for hours
                            r'[0-9]+H'              # One or more digits followed by 'H'
                            r'(?:'               # Optional minutes
                                r'[0-9]+M'          # One or more digits followed by 'M'
                                r'(?:[0-9]+S)?'     # Optional seconds
                            r')?'
                        r')'
                        r'|'                     # OR
                        r'(?P<dur_minute>'       # Named group for minutes
                            r'[0-9]+M'              # One or more digits followed by 'M'
                            r'(?:[0-9]+S)?'         # Optional seconds
                        r')'
                        r'|'                     # OR
                        r'(?P<dur_second>'       # Named group for seconds
                            r'[0-9]+S'              # One or more digits followed by 'S'
                        r')'
                    r')'
                r')?'
            r')'
            r'|'                                 # OR
            r'(?P<dur_time>'                     # Named group for time-only duration
                r'T'                             # Time starts with 'T'
                r'(?:'                           # Non-capturing group for time components
                    r'(?P<dur_hour2>'             # Named group for hours
                        r'[0-9]+H'                  # One or more digits followed by 'H'
                        r'(?:'                   # Optional minutes
                            r'[0-9]+M'              # One or more digits followed by 'M'
                            r'(?:[0-9]+S)?'         # Optional seconds
                        r')?'
                    r')'
                    r'|'                         # OR
                    r'(?P<dur_minute2>'           # Named group for minutes
                        r'[0-9]+M'                  # One or more digits followed by 'M'
                        r'(?:[0-9]+S)?'             # Optional seconds
                    r')'
                    r'|'                         # OR
                    r'(?P<dur_second2>'           # Named group for seconds
                        r'[0-9]+S'                  # One or more digits followed by 'S'
                    r')'
                r')'
            r')'
            r'|'                                 # OR
            r'(?P<dur_week>'                     # Named group for weeks
                r'[0-9]+W'                          # One or more digits followed by 'W'
            r')'
        r')'
    ),
    # Email addresses
    "email": (
        r'(?P<local_part>'
            r'(?P<dot_string>'
                r'[^\s@\.]+'
                r'(\.[^\s@\.]+)*'
            r')'
            # TODO: Add support for quoted strings
        r')'
        r'@'
        r'('
            r'(?P<domain>'
                r'(?P<sub_domain>'
                    r'[a-zA-Z0-9]'
                    r'([a-zA-Z0-9-]*[a-zA-Z0-9])?'
                r')'
                r'(\.(?P<sub_domain2>'
                    r'[a-zA-Z0-9]'
                    r'([a-zA-Z0-9-]*[a-zA-Z0-9])?'
                r'))*'
            r')'
            r'|' # OR
            r'\[(?P<ipv4>((([0-9])|(([1-9])[0-9]|(25[0-5]|(2[0-4]|(1)[0-9])[0-9])))\.){3}(([0-9])|(([1-9])[0-9]|(25[0-5]|(2[0-4]|(1)[0-9])[0-9]))))\]'
        r')'
    ),
    "idn-email": None,
    # Hostnames
    "hostname": r"[a-zA-Z0-9]([a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(\.[a-zA-Z0-9]([a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)*",
    "idn-hostname": None,
    "ipv4": r'((([0-9])|(([1-9])[0-9]|(25[0-5]|(2[0-4]|(1)[0-9])[0-9])))\.){3}(([0-9])|(([1-9])[0-9]|(25[0-5]|(2[0-4]|(1)[0-9])[0-9])))',
    "ipv6": (
        # Full IPv6 address without "::"
        r'(?:'
            r'(?P<full>(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4})'
        r')'
        r'|'  # OR
        # Leading "::" (shortens leading zeros)
        r'(?:'
            r'::(?:[0-9a-fA-F]{1,4}:){0,5}(?P<ls32>[0-9a-fA-F]{1,4}:[0-9a-fA-F]{1,4})'
        r')'
        r'|'  # OR
        # "::" within the address, and variants reducing the length of the address
        r'(?:'
            r'(?P<h16_1>[0-9a-fA-F]{1,4})?::(?:[0-9a-fA-F]{1,4}:){0,4}(?P<ls32_1>[0-9a-fA-F]{1,4}:[0-9a-fA-F]{1,4})'
        r')'
        r'|'  # OR
        r'(?:'
            r'((?:[0-9a-fA-F]{1,4}:){0,1}[0-9a-fA-F]{1,4})?::(?:[0-9a-fA-F]{1,4}:){0,3}(?P<ls32_2>[0-9a-fA-F]{1,4}:[0-9a-fA-F]{1,4})'
        r')'
        r'|'  # OR
        r'(?:'
            r'((?:[0-9a-fA-F]{1,4}:){0,2}[0-9a-fA-F]{1,4})?::(?:[0-9a-fA-F]{1,4}:){0,2}(?P<ls32_3>[0-9a-fA-F]{1,4}:[0-9a-fA-F]{1,4})'
        r')'
        r'|'  # OR
        r'(?:'
            r'((?:[0-9a-fA-F]{1,4}:){0,3}[0-9a-fA-F]{1,4})?::[0-9a-fA-F]{1,4}:(?P<ls32_4>[0-9a-fA-F]{1,4}:[0-9a-fA-F]{1,4})'
        r')'
        r'|'  # OR
        r'(?:'
            r'((?:[0-9a-fA-F]{1,4}:){0,4}[0-9a-fA-F]{1,4})?::(?P<ls32_5>[0-9a-fA-F]{1,4}:[0-9a-fA-F]{1,4})'
        r')'
        r'|'  # OR
        r'(?:'
            r'((?:[0-9a-fA-F]{1,4}:){0,5}[0-9a-fA-F]{1,4})?::(?P<h16_2>[0-9a-fA-F]{1,4})'
        r')'
        r'|'  # OR
        r'(?:'
            r'((?:[0-9a-fA-F]{1,4}:){0,6}[0-9a-fA-F]{1,4})?::'
        r')'
    ),
    # Resource identifiers
    "uuid": (
        r'(?P<time_low>[0-9a-fA-F]{8})'      # 4 hex octets for time-low
        r'-'                                 # Literal hyphen
        r'(?P<time_mid>[0-9a-fA-F]{4})'      # 2 hex octets for time-mid
        r'-'                                 # Literal hyphen
        r'(?P<time_high_and_version>[0-9a-fA-F]{4})'  # 2 hex octets for time-high-and-version
        r'-'                                 # Literal hyphen
        r'(?P<clock_seq_and_reserved>[0-9a-fA-F]{2})' # 1 hex octet for clock-seq-and-reserved
        r'(?P<clock_seq_low>[0-9a-fA-F]{2})' # 1 hex octet for clock-seq-low
        r'-'                                 # Literal hyphen
        r'(?P<node>[0-9a-fA-F]{12})'         # 6 hex octets for node
    ),
    "uri": None,
    "uri-reference": None,
    "iri": None,
    "iri-reference": None,
    # URI template
    "uri-template": None,
    # JSON pointers
    "json-pointer": None,
    "relative-json-pointer": None,
    # Regular expressions
    "regex": None, # Might need a full CFG?,
    # Unknown
    "unknown": r"(?s:.*)",
}


def urijoin(base: str, uri: str) -> str:
    # Special case for fragment-only URIs
    if uri.startswith("#"):
        return f"{base}{uri}"
    return urllib.parse.urljoin(base, uri)


def _get_format_pattern(format: str) -> str:
    try:
        pattern = FORMAT_PATTERNS[format]
    except KeyError:
        raise ValueError(f"Format {format!r} is not supported")
    if pattern is None:
        raise NotImplementedError(f"Format {format!r} is not yet supported")
    return pattern


def validate_json_node_keys(node: Mapping[str, Any]):
    keys = set(node.keys())
    # Any key that is a valid JSON schema keyword but not one that we have explicit support for is "invalid"
    invalid_keys = (keys - VALID_KEYS).intersection(DRAFT202012_RESERVED_KEYWORDS)
    if invalid_keys:
        raise ValueError(
            f"JSON schema had keys that could not be processed: {invalid_keys}" f"\nSchema: {node}"
        )


def get_sibling_keys(node: Mapping[str, Any], key: str) -> set[str]:
    # Get the set of functional (non-ignored) keys that are siblings of the given key
    return set(node.keys()) & VALID_KEYS - set(IGNORED_KEYS) - {key}


def check_number_bounds(minimum: Union[float, int, None], maximum: Union[float, int, None], exclusiveMinimum: bool, exclusiveMaximum: bool):
    if minimum is not None and maximum is not None:
        if minimum > maximum:
            raise UnsatisfiableSchemaError(f"minimum ({minimum}) is greater than maximum ({maximum})")
        if minimum == maximum and (exclusiveMinimum or exclusiveMaximum):
            minimum_repr = f"exclusiveMinimum ({minimum})" if exclusiveMinimum else f"minimum ({minimum})"
            maximum_repr = f"exclusiveMaximum ({maximum})" if exclusiveMaximum else f"maximum ({maximum})"
            raise UnsatisfiableSchemaError(f"{minimum_repr} is equal to {maximum_repr}")


class UnsatisfiableSchemaError(ValueError):
    pass

class GenJson:
    item_separator = ", "
    key_separator = ": "
    def __init__(self, schema: JSONSchema, separators: Optional[tuple[str, str]] = None) -> None:
        self.schema = schema
        if separators is not None:
            self.item_separator, self.key_separator = separators

        registry: referencing.Registry[JSONSchema] = referencing.Registry()
        resource: referencing.Resource[JSONSchema] = referencing.jsonschema.DRAFT202012.create_resource(schema)
        self._base_uri = resource.id() or ""
        registry = registry.with_resource(
            uri=self._base_uri,
            resource=resource
        )
        self._resolver = registry.resolver()
        self._defs: dict[str, Callable[[], GrammarFunction]] = {}


    @guidance(stateless=True)
    def ref(
        self,
        lm,
        *,
        reference: str,
        base_uri: str,
    ):
        """
        Resolve a reference to another schema and return the grammar for that schema.

        Note: we define a zero-argument closure that will return the grammar for the reference and
        add it to the _defs cache. This allows us to avoid re-resolving the reference every time
        and to handle recursive references correctly.
        """
        abspath = urijoin(base_uri, reference)

        if abspath not in self._defs:
            resolved = self._resolver.lookup(abspath)
            base_uri_of_resolved = resolved.resolver._base_uri

            @guidance(stateless=True, dedent=False, cache=True)
            def closure(lm):
                grammar = self.json(json_schema=resolved.contents, base_uri=base_uri_of_resolved)
                return lm + grammar

            self._defs[abspath] = closure
        return lm + self._defs[abspath]()


    @guidance(stateless=True)
    def root(self, lm):
        return lm + self.json(json_schema=self.schema, base_uri=self._base_uri)


    @classmethod
    @guidance(stateless=True)
    def integer(cls, lm, minimum: Union[float, int, None] = None, maximum: Union[float, int, None] = None, exclusiveMinimum: bool = False, exclusiveMaximum: bool = False):
        check_number_bounds(minimum, maximum, exclusiveMinimum, exclusiveMaximum)

        if minimum is not None:
            if exclusiveMinimum:
                if minimum != int(minimum):
                    minimum = math.ceil(minimum)
                else:
                    minimum += 1
            else:
                minimum = math.ceil(minimum)
            minimum = int(minimum)
        if maximum is not None:
            if exclusiveMaximum:
                if maximum != int(maximum):
                    maximum = math.floor(maximum)
                else:
                    maximum -= 1
            else:
                maximum = math.floor(maximum)
            maximum = int(maximum)

        return lm + lexeme(rx_int_range(minimum, maximum), contextual=True)


    @classmethod
    @guidance(stateless=True)
    def number(cls, lm, minimum: Optional[float] = None, maximum: Optional[float] = None, exclusiveMinimum: bool = False, exclusiveMaximum: bool = False):
        check_number_bounds(minimum, maximum, exclusiveMinimum, exclusiveMaximum)

        return lm + lexeme(
            rx_float_range(
                minimum, maximum,
                left_inclusive = not exclusiveMinimum,
                right_inclusive = not exclusiveMaximum
            ),
            contextual=True
        )


    @classmethod
    @guidance(stateless=True)
    def string(
        cls,
        lm,
        *,
        min_length: int = 0,
        max_length: Union[int, None] = None,
        regex: Union[str, None] = None,
        format: Union[str, None] = None,
    ):
        if min_length is not None and max_length is not None and min_length > max_length:
            raise UnsatisfiableSchemaError(f"String minLength ({min_length}) is greater than maxLength ({max_length})")

        if (regex is not None or format is not None) and (min_length > 0 or max_length is not None):
            raise ValueError(
                "If a pattern or format is specified for a JSON string, minLength and maxLength must be left unspecified."
            )

        if regex is not None and format is not None:
            raise ValueError("Cannot specify both a regex and a format for a JSON string")

        if format is not None:
            regex = _get_format_pattern(format)

        elif regex is not None:
            # Sanitize the regex, removing unnecessary anchors that may cause problems later
            # NOTE/TODO: this could potentially be pushed further down into the lexeme function,
            # but it's not immediately clear whether anchors in other contexts are superfluous.
            regex = regex.lstrip("^").rstrip("$")

        elif regex is None:
            range_expr = f"{{{min_length},{max_length}}}" if max_length is not None else f"{{{min_length},}}"
            regex = f"(?s:.{range_expr})"

        return lm + lexeme(regex, contextual=True, json_string=True)


    @guidance(stateless=True)
    def object(
        self,
        lm,
        *,
        properties: Mapping[str, JSONSchema],
        additional_properties: JSONSchema,
        required: Sequence[str],
        base_uri: str,
    ):
        illegal_keys = set()
        property_grammars: dict[str, GrammarFunction] = {}
        for name, schema in properties.items():
            try:
                property_grammars[name] = self.json(json_schema=schema, base_uri=base_uri)
            except UnsatisfiableSchemaError as e:
                # We get here if the schema is a literal False or is otherwise determined to be unsatisfiable
                if name in required:
                    raise UnsatisfiableSchemaError(f"Required property {name!r} is unsatisfiable") from e
                # If the property is not required, we will just "blacklist" this key (e.g. if the schema was False)
                # Note that we're just dropping this exception.
                # Use json_dumps to properly quote / escape the key before adding it to the blacklist
                key = json_dumps(name)
                illegal_keys.add(key)

        additional_properties_grammar: Optional[GrammarFunction] = None
        try:
            additional_properties_grammar = self.json(json_schema=additional_properties, base_uri=base_uri)
        except UnsatisfiableSchemaError as e:
            if any(k not in properties for k in required):
                # "required" keys will be validated against "properties" if they're present, otherwise against "additionalProperties".
                # If "additionalProperties" is unsatisfiable, then required keys must be in "properties".
                raise UnsatisfiableSchemaError(
                    f"Required properties not in properties but additionalProperties is unsatisfiable."
                    f" Missing required properties: {list(r for r in required if r not in properties)}"
                ) from e
            else:
                # If "additionalProperties" is unsatisfiable but there are no required properties that need to be validated against it,
                # then we can safely ignore it. Note that this means that we are just going to drop this exception.
                pass

        keys: list[str] = []
        required_items: list[bool] = []
        item_grammars: list[GrammarFunction] = []
        # First iterate over the properties in order, then iterate over any missing required keys, using additional_properties as the schema
        for name in (*property_grammars.keys(), *(r for r in required if r not in properties)):
            # Use json_dumps to properly quote / escape the key
            key = json_dumps(name)
            keys.append(key)
            # Identify if the key is required
            required_items.append(name in required)
            # Build the grammar we'll use for this property
            item_grammars.append(f'{key}{self.key_separator}' + property_grammars.get(name, cast(GrammarFunction, additional_properties_grammar)))

        if additional_properties is not False:
            # Key for additionalProperties is a json string, but we need to disallow any properties that are already defined
            additional_key_grammar: GrammarFunction
            if len(keys) > 0:
                additional_key_grammar = as_regular_grammar(
                    And([
                        lexeme(r'"([^"\\]|\\["\\/bfnrt]|\\u[0-9a-fA-F]{4})*"'),
                        Not(lexeme('|'.join(map(quote_regex, (*keys, *illegal_keys))))),
                    ]),
                    lexeme = True,
                )
            else:
                additional_key_grammar = self.string()

            if additional_properties_grammar is not None:
                additional_item_grammar = additional_key_grammar + self.key_separator + additional_properties_grammar
                additional_items_grammar = sequence(additional_item_grammar + self.item_separator) + additional_item_grammar
                item_grammars.append(additional_items_grammar)
                required_items.append(False)

        return lm + "{" + self._join(
            elements = tuple(item_grammars),
            required = tuple(required_items),
        ) + "}"


    @guidance(stateless=True, cache=True)
    def _join(self, lm, *, elements: tuple[GrammarFunction, ...], required: tuple[bool, ...], prefixed: bool = False):
        if not elements:
            return lm

        elem, elements = elements[0], elements[1:]
        is_required, required = required[0], required[1:]

        if prefixed:
            if is_required:
                # If we know we have preceeding elements, we can safely just add a (',' + e)
                return lm + (self.item_separator + elem + self._join(elements=elements, required=required, prefixed=True))
            # If we know we have preceeding elements, we can safely just add an optional(',' + e)
            return lm + (optional(self.item_separator + elem) + self._join(elements=elements, required=required, prefixed=True))
        if is_required:
            # No preceding elements, and our element is required, so we just add the element
            return lm + (elem + self._join(elements=elements, required=required, prefixed=True))

        # No preceding elements, and our element is optional, so we add a select between the two options.
        # The first option is the recursive call with no preceding elements, the second is the recursive call
        # with the current element as a prefix.
        return lm + select([
            self._join(elements=elements, required=required, prefixed=False),
            elem + self._join(elements=elements, required=required, prefixed=True)
        ])


    @guidance(stateless=True)
    def array(
        self,
        lm,
        *,
        prefix_items_schema: Sequence[JSONSchema],
        item_schema: JSONSchema,
        min_items: int,
        max_items: Optional[int],
        base_uri: str,
    ):
        if max_items is not None and min_items > max_items:
            raise UnsatisfiableSchemaError(f"minItems ({min_items}) is greater than maxItems ({max_items})")

        items_grammar: Optional[GrammarFunction] = None
        try:
            items_grammar = self.json(json_schema=item_schema, base_uri=base_uri)
        except UnsatisfiableSchemaError as e:
            if len(prefix_items_schema) < min_items:
                raise UnsatisfiableSchemaError(
                    f"prefixItems has too few elements ({len(prefix_items_schema)}) to satisfy minItems ({min_items})"
                    f" but item schema is unsatisfiable"
                ) from e
            else:
                # If we've already satisfied min_items, we can just ignore the unsatisfiable item schema. This just means
                # that we can't generate any more items after the prefix items.
                # Note that this means that we are just going to drop this exception.
                pass

        required_items = []
        optional_items = []

        # If max_items is None, we can add an infinite tail of items later
        n_to_add = max(len(prefix_items_schema), min_items) if max_items is None else max_items
        for i in range(n_to_add):
            if i < len(prefix_items_schema):
                try:
                    item = self.json(json_schema=prefix_items_schema[i], base_uri=base_uri)
                except UnsatisfiableSchemaError as e:
                    # i corresponds to the number of items we've already satisfied
                    if i < min_items:
                        raise UnsatisfiableSchemaError(f"prefixItems[{i}] is unsatisfiable but min_items is {min_items}") from e
                    # Having an unsatisfiable prefix item is fine if we've already satisfied min_items, but this effectively sets max_items to i
                    # Note that this means that we are just going to drop this exception.
                    max_items = i
                    break

            elif items_grammar is not None:
                item = items_grammar
            else:
                assert i >= min_items
                break

            if i < min_items:
                required_items.append(item)
            else:
                optional_items.append(item)

        if max_items is None and items_grammar is not None:
            # Add an infinite tail of items
            optional_items.append(items_grammar + sequence(self.item_separator + items_grammar))

        lm += "["

        if required_items:
            first, *rest = required_items
            lm += first
            for item in rest:
                lm += self.item_separator + item

        if optional_items:
            # This is a bit subtle and would not be required if not for prefixItems -- the previous
            # must be present before the next one may be added, meaning we have nested optionals:
            # (first optional(,second optional(,third (optional(,...)))))
            first, *rest = optional_items
            tail: Union[str, GrammarFunction] = ""
            for item in reversed(rest):
                tail = optional(self.item_separator + item + tail)
            tail = first + tail

            if required_items:
                lm += optional(self.item_separator + tail)
            else:
                lm += optional(tail)

        lm += "]"
        return lm


    @guidance(stateless=True)
    def anyOf(
        self,
        lm,
        *,
        anyof_list: Sequence[JSONSchema],
        base_uri: str,
    ):
        if not anyof_list:
            raise UnsatisfiableSchemaError("anyOf has no schemas")

        options: list[GrammarFunction] = []
        for item in anyof_list:
            try:
                options.append(self.json(json_schema=item, base_uri=base_uri))
            except UnsatisfiableSchemaError:
                # No need to raise an error if one of the schemas is unsatisfiable. We'll check again at the end and raise if ALL
                # schemas are unsatisfiable. Note that this means that we are just going to drop this exception.
                pass
        if not options:
            # Can't really point to any one schema that's unsatisfiable, so let's include all the schemas in the error message
            raise UnsatisfiableSchemaError("all anyOf schemas are unsatisfiable: " + json_dumps(anyof_list))
        return lm + select(options)

    @guidance(stateless=True)
    def oneOf(
        self,
        lm,
        *,
        oneof_list: Sequence[JSONSchema],
        base_uri: str,
    ):
        if not oneof_list:
            raise UnsatisfiableSchemaError("oneOf has no schemas")
        if len(oneof_list) == 1:
            return lm + self.json(json_schema=oneof_list[0], base_uri=base_uri)
        warnings.warn("oneOf not fully supported, falling back to anyOf. This may cause validation errors in some cases.")
        return lm + self.anyOf(anyof_list=oneof_list, base_uri=base_uri)

    def push_sibling_keys(self, json_schema: dict[str, Any]) -> dict[str, Any]:
        """
        If sibling keys are present next to anyOf, oneOf, or $ref, we push them down into an allOf.
        """
        parent_schema = json_schema.copy()
        anyof_list = parent_schema.pop(Keyword.ANYOF, [])
        oneof_list = parent_schema.pop(Keyword.ONEOF, [])
        allof_list = parent_schema.pop(Keyword.ALLOF, [])
        ref = parent_schema.pop(Keyword.REF, None)

        common = []
        if VALID_KEYS.intersection(parent_schema) - set(IGNORED_KEYS):
            # If there are any sibling keys, we need to push them down into an allOf
            common.append(parent_schema)
        if allof_list:
            common.extend(allof_list)
        if ref:
            # TODO: $id / base_uri?
            common.append({Keyword.REF: ref})

        if anyof_list and oneof_list:
            return {
                "oneOf": [
                    {"allOf": common + [one_item, any_item]}
                    for one_item in oneof_list
                    for any_item in anyof_list
                ],
            }

        if oneof_list:
            if not common:
                return {"oneOf": oneof_list}
            return {
                "oneOf": [
                    {"allOf": common + [one_item]}
                    for one_item in oneof_list
                ],
            }

        if anyof_list:
            if not common:
                return {"anyOf": anyof_list}
            return {
                "anyOf": [
                    {"allOf": common + [any_item]}
                    for any_item in anyof_list
                ],
            }

        if len(common) == 1:
            return common[0]

        return {"allOf": common}

    def reduce_schema(self, orig_schema: dict[str, Any], base_uri: str) -> dict[str, Any]:
        types: list[set[str]] = []
        properties: defaultdict[str, list[JSONSchema]] = defaultdict(list)
        required: dict[str, None] = dict() # use a dict for ordered-set behavior
        additional_properties_list: list[tuple[JSONSchema, set[str]]] = []
        prefix_items: defaultdict[int, list[JSONSchema]] = defaultdict(list)
        items_list: list[tuple[JSONSchema, set[int]]] = []
        other_data: dict[str, Any] = {}
        enums: list[Sequence[Any]] = []
        consts: list[Any] = []

        def handle_keyword(key: str, value: Any, parent_schema: dict[str, Any], base_uri: str):
            if key == Keyword.ANYOF:
                raise NotImplementedError("anyOf in allOf not yet supported")

            elif key == Keyword.ONEOF:
                raise NotImplementedError("oneOf in allOf not yet supported")

            elif key == Keyword.ALLOF:
                value = cast(Sequence[JSONSchema], value)
                for schema in value:
                    add_schema(schema, base_uri)

            elif key == Keyword.REF:
                ref = cast(str, value)
                abspath = urijoin(base_uri, ref)
                resolved = self._resolver.lookup(abspath)
                add_schema(resolved.contents, base_uri=resolved.resolver._base_uri)

            elif key == Keyword.CONST:
                consts.append(value)

            elif key == Keyword.ENUM:
                value = cast(list[Any], value)
                enums.append(value)

            elif key == Keyword.TYPE:
                value = cast(Union[str, Sequence[str]], value)
                if isinstance(value, str):
                    value_set = {value}
                else:
                    value_set = set(value)
                types.append(value_set)

            elif key == ObjectKeywords.PROPERTIES:
                value = cast(dict[str, JSONSchema], value)
                for name, schema in value.items():
                    if isinstance(schema, dict):
                        this_base_uri = schema.get(Keyword.ID, base_uri)
                        if Keyword.REF in schema:
                            # Make the ref absolute so that it can be resolved in the right scope later
                            schema = schema.copy()
                            schema[Keyword.REF] = urijoin(this_base_uri, schema[Keyword.REF])
                    properties[name].append(schema)

            elif key == ObjectKeywords.REQUIRED:
                value = cast(Sequence[str], value)
                required.update({name: None for name in value})

            elif key == ObjectKeywords.ADDITIONAL_PROPERTIES:
                # TODO: unevaluatedProperties?
                value = cast(JSONSchema, value)
                # We need to keep track of which properties are exempt from this additionalProperties schema,
                # i.e. the ones defined in the parent schema
                exempt_properties: set[str] = set()
                if ObjectKeywords.PROPERTIES in parent_schema:
                    exempt_properties = set(parent_schema[ObjectKeywords.PROPERTIES])
                additional_properties_list.append(
                    (value, exempt_properties)
                )

            elif key == ArrayKeywords.PREFIX_ITEMS:
                value = cast(Sequence[JSONSchema], value)
                for i, schema in enumerate(value):
                    prefix_items[i].append(schema)

            elif key == ArrayKeywords.ITEMS:
                # TODO: unevaluatedItems?
                value = cast(JSONSchema, value)
                # We need to keep track of which prefixItems are exempt from this additionalItems schema,
                # i.e. the ones defined in the parent schema
                exempt_prefix_items: set[int] = set()
                if ArrayKeywords.PREFIX_ITEMS in parent_schema:
                    exempt_prefix_items = set(range(len(parent_schema[ArrayKeywords.PREFIX_ITEMS])))
                items_list.append(
                    (value, exempt_prefix_items)
                )

            elif key in other_data:
                if key in {
                    NumberKeywords.MINIMUM, NumberKeywords.EXCLUSIVE_MINIMUM,
                    StringKeywords.MIN_LENGTH, ArrayKeywords.MIN_ITEMS
                }:
                    other_data[key] = max(other_data[key], value)
                elif key in {
                    NumberKeywords.MAXIMUM, NumberKeywords.EXCLUSIVE_MAXIMUM,
                    StringKeywords.MAX_LENGTH, ArrayKeywords.MAX_ITEMS
                }:
                    other_data[key] = min(other_data[key], value)
                else:
                    raise NotImplementedError(f"Don't yet know how to reduce multiple values of {key!r} in allOf")

            else:
                other_data[key] = value

        def add_schema(schema: JSONSchema, base_uri: str):
            if schema is True:
                return
            if schema is False:
                raise UnsatisfiableSchemaError("allOf contains a 'false' schema")

            # Validate the schema's keys (we have only validated the parent schema's keys so far)
            # TODO: This will make us validate the parent twice... should probably be refactored
            validate_json_node_keys(schema)

            # Set the base_uri for this schema
            if Keyword.ID in schema:
                # TODO: avoid copies if possible..?
                schema = schema.copy()
                base_uri = urijoin(base_uri, schema.pop(Keyword.ID))

            for key, value in schema.items():
                if key in IGNORED_KEYS:
                    continue
                handle_keyword(key, value, schema, base_uri)

        add_schema(orig_schema, base_uri)

        combined_schema: dict[str, Any] = {}

        # Post-process additional_properties to make sure we apply the additional properties of one
        # schema to the properties of another schema
        for additional_schema, exempt_properties in additional_properties_list:
            for name in set(properties) - exempt_properties:
                properties[name].append(additional_schema)

        # Post-process items to make sure we apply the additional items of one schema to the prefix items of another schema
        for additional_schema, exempt_prefix_items in items_list:
            for i in set(prefix_items) - exempt_prefix_items:
                prefix_items[i].append(additional_schema)

        if properties:
            combined_schema[ObjectKeywords.PROPERTIES] = {}
            for name, schemas in properties.items():
                if len(schemas) == 1:
                    combined_schema[ObjectKeywords.PROPERTIES][name] = schemas[0]
                else:
                    combined_schema[ObjectKeywords.PROPERTIES][name] = {"allOf": schemas}

        if required:
            combined_schema[ObjectKeywords.REQUIRED] = list(required.keys())

        if additional_properties_list:
            if len(additional_properties_list) == 1:
                combined_schema[ObjectKeywords.ADDITIONAL_PROPERTIES], _ = additional_properties_list[0]
            else:
                combined_schema[ObjectKeywords.ADDITIONAL_PROPERTIES] = {"allOf": [schema for schema, _ in additional_properties_list]}

        if prefix_items:
            combined_schema[ArrayKeywords.PREFIX_ITEMS] = []
            for i in range(len(prefix_items)):
                schemas = prefix_items[i]
                if len(schemas) == 1:
                    combined_schema[ArrayKeywords.PREFIX_ITEMS].append(schemas[0])
                else:
                    combined_schema[ArrayKeywords.PREFIX_ITEMS].append({"allOf": schemas})

        if items_list:
            if len(items_list) == 1:
                combined_schema[ArrayKeywords.ITEMS], _ = items_list[0]
            else:
                combined_schema[ArrayKeywords.ITEMS] = {"allOf": [schema for schema, _ in items_list]}

        if enums:
            if len(enums) == 1:
                enum = enums[0]
            else:
                def reduce_enums(enum_a, enum_b):
                    try:
                        enum = list(set(enum_a) & set(enum_b))
                    except TypeError:
                        # Check on equality, not on hash
                        # Yes, this is O(n^2).
                        # Hope the items were unique.
                        # ¯\_(ツ)_/¯
                        enum = [a for a in enum_a for b in enum_b if a == b]
                    return enum
                enum = functools.reduce(reduce_enums, enums)
            if not enum:
                raise UnsatisfiableSchemaError(f"allOf has enums with no common values: {enums}")
            combined_schema[Keyword.ENUM] = enum

        if consts:
            const, *rest = consts
            for c in rest:
                if c != const:
                    raise UnsatisfiableSchemaError(f"allOf has consts with different values: {consts}")
            combined_schema[Keyword.CONST] = const

        if types:
            if len(types) == 1:
                type = list(types[0])
            else:
                def reduce_types(type_a: set[str], type_b: set[str]) -> set[str]:
                    common_types = type_a & type_b
                    # Integer is a "subtype" of number, so ensure we keep integer if we have "number" in one and "integer" in the other
                    if JSONType.INTEGER not in common_types and (
                        (JSONType.NUMBER in type_a and JSONType.INTEGER in type_b) or
                        (JSONType.INTEGER in type_a and JSONType.NUMBER in type_b)
                    ):
                        common_types.add(JSONType.INTEGER)
                    return common_types
                type = list(functools.reduce(reduce_types, types)) # type: ignore[arg-type]
                if not type:
                    raise UnsatisfiableSchemaError(f"allOf has conflicting types: {types}")
            combined_schema[Keyword.TYPE] = type

        assert not set(combined_schema) & set(other_data)
        combined_schema.update(other_data)
        return combined_schema


    @guidance(stateless=True)
    def allOf(
        self,
        lm,
        *,
        parent_schema: dict[str, Any],
        base_uri: str,
    ):
        reduced_schema = self.reduce_schema(parent_schema, base_uri)
        return lm + self.json(json_schema=reduced_schema, base_uri=base_uri)


    @guidance(stateless=True)
    def const(
        self,
        lm,
        *,
        value: Union[None, bool, int, float, str, Mapping, Sequence],
        parent_schema: JSONSchema,
    ):
        try:
            jsonschema.validate(
                instance=value,
                schema=parent_schema,
            )
        except jsonschema.ValidationError as e:
            raise UnsatisfiableSchemaError(f"const {value!r} is inconsistent with parent schema: {parent_schema}") from e
        # Base case
        if isinstance(value, (type(None), bool, int, float, str)):
            return lm + json_dumps(value)
        # Recursive cases
        # NOTE: we could potentially just use json_dumps in all cases, but this will ensure that we're
        # properly treating all parts as individual lexemes, which makes whitespace flexibility possible
        if isinstance(value, Mapping):
            return lm + self.json(
                json_schema={
                    "type": "object",
                    "properties": {k: {"const": v} for k, v in dict(value).items()},
                    "required": list(value.keys()),
                    "additionalProperties": False,
                },
                base_uri="", # dummy value -- we don't need to resolve anything
            )
        if isinstance(value, Sequence):
            return lm + self.json(
                json_schema={
                    "type": "array",
                    "prefixItems": [{"const": v} for v in list(value)],
                    "minItems": len(value),
                    "maxItems": len(value),
                    "items": False,
                },
                base_uri="", # dummy value -- we don't need to resolve anything
            )
        raise TypeError(f"Unsupported value type: {type(value)} for value: {value!r}")

    @guidance(stateless=True)
    def enum(
        self,
        lm,
        *,
        options: Sequence[Union[None, bool, int, float, str, Mapping, Sequence]],
        parent_schema: JSONSchema,
    ):
        if not options:
            raise UnsatisfiableSchemaError("enum has no options")
        all_opts: list[GrammarFunction] = []
        for instance in options:
            try:
                grm = self.const(value=instance, parent_schema=parent_schema)
            except UnsatisfiableSchemaError:
                # Like anyOf, we don't want to raise an error if one of the options is unsatisfiable. We'll check again at the end
                # and raise if ALL options are unsatisfiable. Note that this means that we are just going to drop this exception.
                continue
            all_opts.append(grm)
        if not all_opts:
            raise UnsatisfiableSchemaError(f"All enum options {options} are inconsistent with parent schema: {parent_schema}")
        return lm + select(options=all_opts)


    @guidance(stateless=True)
    def any(self, lm):
        return lm + select(
            [
                # Dummy base uris ok since we're not resolving anything
                self.json(json_schema={"type": "null"}, base_uri=""),
                self.json(json_schema={"type": "boolean"}, base_uri=""),
                self.json(json_schema={"type": "integer"}, base_uri=""),
                self.json(json_schema={"type": "number"}, base_uri=""),
                self.json(json_schema={"type": "string"}, base_uri=""),
                # Recursive cases
                self.json(
                    json_schema={
                        "type": "array",
                        "items": True,
                    },
                    base_uri="",
                ),
                self.json(
                    json_schema={
                        "type": "object",
                        "additionalProperties": True,
                    },
                    base_uri="",
                ),
            ]
        )


    @guidance(stateless=True)
    def json(
        self,
        lm,
        *,
        json_schema: JSONSchema,
        base_uri: str,
    ):
        if json_schema is True:
            json_schema = {}
        elif json_schema is False:
            raise UnsatisfiableSchemaError("No valid JSON can be generated from a schema of `false`")

        if json_schema == {}:
            return lm + self.any()

        # Early exit for simple cases
        if Keyword.CONST in json_schema:
            return lm + self.const(value=json_schema[Keyword.CONST], parent_schema=json_schema)

        if Keyword.ENUM in json_schema:
            return lm + self.enum(options=json_schema[Keyword.ENUM], parent_schema=json_schema)

        # More complex cases; validation needed
        validate_json_node_keys(json_schema)
        json_schema = self.push_sibling_keys(json_schema)

        if Keyword.ALLOF in json_schema:
            sibling_keys = get_sibling_keys(json_schema, Keyword.ALLOF)
            assert not sibling_keys
            return lm + self.allOf(parent_schema=json_schema, base_uri=base_uri)

        if Keyword.ANYOF in json_schema:
            sibling_keys = get_sibling_keys(json_schema, Keyword.ANYOF)
            assert not sibling_keys
            return lm + self.anyOf(anyof_list=json_schema[Keyword.ANYOF], base_uri=base_uri)

        if Keyword.ONEOF in json_schema:
            sibling_keys = get_sibling_keys(json_schema, Keyword.ONEOF)
            assert not sibling_keys
            return lm + self.oneOf(oneof_list=json_schema[Keyword.ONEOF], base_uri=base_uri)

        if Keyword.REF in json_schema:
            sibling_keys = get_sibling_keys(json_schema, Keyword.REF)
            assert not sibling_keys
            return lm + self.ref(reference=json_schema[Keyword.REF], base_uri=base_uri)

        if Keyword.TYPE in json_schema and isinstance(json_schema[Keyword.TYPE], str):
            target_type = json_schema[Keyword.TYPE]
            if target_type == JSONType.NULL:
                return  lm + "null"
            elif target_type == JSONType.BOOLEAN:
                return lm + select(["true", "false"])
            elif target_type in {JSONType.INTEGER, JSONType.NUMBER}:
                minimum = cast(Union[int, float, None], json_schema.get(NumberKeywords.MINIMUM, None))
                maximum = cast(Union[int, float, None], json_schema.get(NumberKeywords.MAXIMUM, None))
                # Older schemas (Draft4) may have exclusiveMinimum and exclusiveMaximum as booleans, but Draft202012+ should have them as numbers
                exclusive_minimum = cast(Union[int, float, None], json_schema.get(NumberKeywords.EXCLUSIVE_MINIMUM, None))
                exclusive_maximum = cast(Union[int, float, None], json_schema.get(NumberKeywords.EXCLUSIVE_MAXIMUM, None))
                # Internally, we'll use Draft4 style booleans
                exclusive_minimum_flag: bool = False
                exclusive_maximum_flag: bool = False

                if exclusive_minimum is not None:
                    if minimum is None or exclusive_minimum >= minimum:
                        minimum = exclusive_minimum
                        exclusive_minimum_flag = True

                if exclusive_maximum is not None:
                    if maximum is None or exclusive_maximum <= maximum:
                        maximum = exclusive_maximum
                        exclusive_maximum_flag = True

                if target_type == JSONType.INTEGER:
                    return lm + self.integer(
                        minimum=minimum,
                        maximum=maximum,
                        exclusiveMinimum=exclusive_minimum_flag,
                        exclusiveMaximum=exclusive_maximum_flag,
                    )
                else:
                    return lm + self.number(
                        minimum=minimum,
                        maximum=maximum,
                        exclusiveMinimum=exclusive_minimum_flag,
                        exclusiveMaximum=exclusive_maximum_flag,
                    )
            elif target_type == JSONType.STRING:
                return lm + self.string(
                    regex=json_schema.get(StringKeywords.PATTERN, None),
                    format=json_schema.get(StringKeywords.FORMAT, None),
                    min_length=json_schema.get(StringKeywords.MIN_LENGTH, 0),
                    max_length=json_schema.get(StringKeywords.MAX_LENGTH, None),
                )
            elif target_type == JSONType.ARRAY:
                return lm + self.array(
                    prefix_items_schema=json_schema.get(ArrayKeywords.PREFIX_ITEMS, []),
                    item_schema=json_schema.get(ArrayKeywords.ITEMS, True),
                    min_items=json_schema.get(ArrayKeywords.MIN_ITEMS, 0),
                    max_items=json_schema.get(ArrayKeywords.MAX_ITEMS, None),
                    base_uri=base_uri,
                )
            elif target_type == JSONType.OBJECT:
                return lm + self.object(
                    properties=json_schema.get(ObjectKeywords.PROPERTIES, {}),
                    additional_properties=json_schema.get(ObjectKeywords.ADDITIONAL_PROPERTIES, True),
                    required=json_schema.get(ObjectKeywords.REQUIRED, set()),
                    base_uri=base_uri,
                )
            else:
                raise ValueError(f"Unsupported type in schema: {target_type}")

        if Keyword.TYPE in json_schema:
            json_schema = json_schema.copy()
            target_types = cast(Sequence[JSONType], json_schema.pop(Keyword.TYPE))
        else:
            target_types = list(JSONType)

        assert Keyword.TYPE not in json_schema
        # Punt to anyOf if we have multiple types so that it can ignore an unsatisfiable subset
        return lm + self.anyOf(
            anyof_list = [
                {"type": target_type, **json_schema} for target_type in target_types
            ],
            base_uri=base_uri,
        )


@guidance(stateless=True)
def json(
    lm,
    name: Optional[str] = None,
    *,
    schema: Union[
        None,
        str,
        JSONSchema,
        Type["pydantic.BaseModel"],
        "pydantic.TypeAdapter",
    ] = None,
    temperature: float = 0.0,
    max_tokens: int = 100000000,
    separators: Optional[tuple[str, str]] = None,
    whitespace_flexible: bool = False,
    **kwargs,
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
            - A string representing a JSON schema which will be parsed using ``json.loads()``
            - A JSON schema object. This is a JSON schema string which has been passed to ``json.loads()``
            - A subclass of ``pydantic.BaseModel``
            - An instance of ``pydantic.TypeAdapter``
    """
    if "compact" in kwargs:
        warnings.warn("The 'compact' argument is deprecated and has no effect. It will be removed in a future release.", category=DeprecationWarning)
        kwargs.pop("compact")
    if kwargs:
        raise TypeError(f"Unexpected keyword arguments: {kwargs.keys()}")
    if schema is None:
        # Default schema is empty, "anything goes" schema
        # TODO: consider default being `{"type": "object"}`
        schema = {}
    elif isinstance(schema, (Mapping, bool, str)):
        if isinstance(schema, str):
            schema = cast(JSONSchema, json_loads(schema))
        # Raises jsonschema.exceptions.SchemaError or ValueError
        # if schema is not valid
        jsonschema.validators.Draft202012Validator.check_schema(schema)
    elif isinstance(schema, pydantic.TypeAdapter) or (isinstance(schema, type) and issubclass(schema, pydantic.BaseModel)):
        schema = pydantic_to_json_schema(schema)
    else:
        raise TypeError(f"Unsupported schema type: {type(schema)}")

    if whitespace_flexible:
        if separators is None:
            separators = (",", ":")
        skip_regex = r"[\x20\x0A\x0D\x09]+"
    else:
        skip_regex = None

    return lm + with_temperature(
        subgrammar(
            name,
            body=GenJson(schema=schema, separators=separators).root(),
            skip_regex=skip_regex,
            no_initial_skip=True,
            max_tokens=max_tokens,
        ),
        temperature=temperature,
    )
