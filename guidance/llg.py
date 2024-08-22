# Copied from other repo for now

from typing import List, Optional, Union, Literal, TypedDict


# This represents a collection of grammars, with a designated
# "start" grammar at first position.
# Grammars can refer to each other via GrammarRef nodes.
class TopLevelGrammar(TypedDict):
    grammars: List["GrammarWithLexer"]
    max_tokens: Optional[int]


DEFAULT_CONTEXTUAL: bool = True


# The start symbol is at nodes[0]
class GrammarWithLexer(TypedDict):
    nodes: List["NodeJSON"]

    # Only applies to greedy_lexer grammars.
    # This adds a new lexeme that will be ignored when parsing.
    greedy_skip_rx: Optional["RegexSpec"]

    # The default value for 'contextual' in Lexeme nodes.
    contextual: Optional[bool]

    # When set, the regexps can be referenced by their id (position in this list).
    rx_nodes: List["RegexJSON"]

    # If set, the grammar will allow skip_rx as the first lexeme.
    allow_initial_skip: Optional[bool]

    # Normally, when a sequence of bytes is forced by grammar, it is tokenized
    # canonically and forced as tokens.
    # With `no_forcing`, we let the model decide on tokenization.
    # This generally reduces both quality and speed, so should not be used
    # outside of testing.
    no_forcing: Optional[bool]

    # If set, the grammar will allow invalid utf8 byte sequences.
    # Any Unicode regex will cause an error.
    allow_invalid_utf8: Optional[bool]


class NodeProps(TypedDict, total=False):
    max_tokens: Optional[int]
    name: Optional[str]
    capture_name: Optional[str]


class NodeString(NodeProps):
    # Force generation of the specific string.
    literal: str


class NodeGen(NodeProps):
    # Regular expression matching the body of generation.
    body_rx: "RegexSpec"

    # The whole generation must match `body_rx + stop_rx`.
    # Whatever matched `stop_rx` is discarded.
    # If `stop_rx` is empty, it's assumed to be EOS.
    stop_rx: "RegexSpec"

    # When set, the string matching `stop_rx` will be output as a capture
    # with the given name.
    stop_capture_name: Optional[str]

    # Lazy gen()s take the shortest match. Non-lazy take the longest.
    # If not specified, the gen() is lazy if stop_rx is non-empty.
    lazy: Optional[bool]

    # Override sampling temperature.
    temperature: Optional[float]


class NodeLexeme(NodeProps):
    # The regular expression that will greedily match the input.
    rx: "RegexSpec"

    # If false, all other lexemes are excluded when this lexeme is recognized.
    # This is normal behavior for keywords in programming languages.
    # Set to true for eg. a JSON schema with both `/"type"/` and `/"[^"]*"/` as lexemes,
    # or for "get"/"set" contextual keywords in C#.
    contextual: Optional[bool]

    # Override sampling temperature.
    temperature: Optional[float]

    # When set, the lexeme will be quoted as a JSON string.
    # For example, /[a-z"]+/ will be quoted as /([a-z]|\\")+/
    json_string: Optional[bool]

    # It lists the allowed escape sequences, typically one of:
    # "nrbtf\\\"u" - to allow all JSON escapes, including \u00XX for control characters
    #     this is the default
    # "nrbtf\\\"" - to disallow \u00XX control characters
    # "nrt\\\"" - to also disallow unusual escapes (\f and \b)
    # "" - to disallow all escapes
    # Note that \uXXXX for non-control characters (code points above U+001F) are never allowed,
    # as they never have to be quoted in JSON.
    json_allowed_escapes: Optional[str]

    # When set and json_string is also set, "..." will not be added around the regular expression.
    json_raw: Optional[bool]


class NodeGenGrammar(NodeProps):
    grammar: "GrammarId"

    # Override sampling temperature.
    temperature: Optional[float]


class NodeSelect(NodeProps):
    among: List["NodeId"]


class NodeJoin(NodeProps):
    sequence: List["NodeId"]


class NodeStringJSON(TypedDict):
    """Force generation of the specific string."""

    String: "NodeString"


class NodeGenJSON(TypedDict):
    """Generate according to regex."""

    Gen: "NodeGen"


class NodeLexemeJSON(TypedDict):
    """Lexeme in a greedy grammar."""

    Lexeme: "NodeLexeme"


class NodeGenGrammarJSON(TypedDict):
    """Generate according to specified grammar."""

    GenGrammar: "NodeGenGrammar"


class NodeSelectJSON(TypedDict):
    """Generate one of the options."""

    Select: "NodeSelect"


class NodeJoinJSON(TypedDict):
    """Generate all of the nodes in sequence."""

    Join: "NodeJoin"


# Define the main NodeJSON type
NodeJSON = Union[
    NodeStringJSON,
    NodeGenJSON,
    NodeLexemeJSON,
    NodeGenGrammarJSON,
    NodeSelectJSON,
    NodeJoinJSON,
]


from typing import TypedDict, Literal, Union, List, Optional


class RegexAndJSON(TypedDict):
    """Intersection of the regexes."""

    And: List["RegexId"]


class RegexOrJSON(TypedDict):
    """Union of the regexes."""

    Or: List["RegexId"]


class RegexConcatJSON(TypedDict):
    """Concatenation of the regexes."""

    Concat: List["RegexId"]


class RegexLookAheadJSON(TypedDict):
    """Matches the regex; should be at the end of the main regex. The length of the lookahead can be recovered from the engine."""

    LookAhead: "RegexId"


class RegexNotJSON(TypedDict):
    """Matches everything the regex doesn't match. Can lead to invalid UTF-8."""

    Not: "RegexId"


class RegexRepeatJSON(TypedDict):
    """Repeat the regex at least min times, at most max times."""

    Repeat: Union["RegexId", int, Optional[int]]


class RegexLiteralJSON(TypedDict):
    """Matches this string only."""

    Literal: str


class RegexByteLiteralJSON(TypedDict):
    """Matches this string of bytes only. Can lead to invalid UTF-8."""

    ByteLiteral: List[int]


class RegexByteJSON(TypedDict):
    """Matches this byte only. If byte is not in 0..127, it may lead to invalid UTF-8."""

    Byte: int


class RegexByteSetJSON(TypedDict):
    """Matches any byte in the set, expressed as bitset. Can lead to invalid UTF-8 if the set is not a subset of 0..127."""

    ByteSet: List[int]


class RegexRegexJSON(TypedDict):
    """Compile the regex using the regex_syntax crate."""

    Regex: str


# Define the main RegexJSON type
RegexJSON = Union[
    RegexAndJSON,
    RegexOrJSON,
    RegexConcatJSON,
    RegexLookAheadJSON,
    RegexNotJSON,
    RegexRepeatJSON,
    RegexLiteralJSON,
    RegexByteLiteralJSON,
    RegexByteJSON,
    RegexByteSetJSON,
    RegexRegexJSON,
    # Matches the empty string. Same as Concat([]).
    Literal["EmptyString"],
    # Matches nothing. Same as Or([]).
    Literal["NoMatch"],
]

# The actual wire format allows for direct strings, but we always use nodes
RegexSpec = Union[str, "RegexId"]

GrammarId = int
NodeId = int
RegexId = int


class BytesOutput(TypedDict):
    hex: str
    str: str


class OutCapture(BytesOutput):
    object: Literal["capture"]
    name: str
    log_prob: float


class OutFinalText(BytesOutput):
    object: Literal["final_text"]


class OutText(BytesOutput):
    object: Literal["text"]
    log_prob: float
    num_tokens: int
    is_generated: bool
    stats: "ParserStats"


ParserOutput = Union[OutCapture, OutFinalText, OutText]


class ParserStats(TypedDict):
    runtime_us: int
    rows: int
    definitive_bytes: int
    lexer_ops: int
    all_items: int
    hidden_bytes: int


# AICI stuff:


class RunUsageResponse(TypedDict):
    sampled_tokens: int
    ff_tokens: int
    cost: float


class InitialRunResponse(TypedDict):
    id: str
    object: Literal["initial-run"]
    created: int
    model: str


class RunResponse(TypedDict):
    object: Literal["run"]
    forks: List["RunForkResponse"]
    usage: RunUsageResponse


class RunForkResponse(TypedDict):
    index: int
    finish_reason: Optional[str]
    text: str
    error: str
    logs: str
    storage: List[object]
    micros: int


AssistantPromptRole = Literal["system", "user", "assistant"]


class AssistantPrompt(TypedDict):
    role: AssistantPromptRole
    content: str


class RunRequest(TypedDict):
    controller: str
    controller_arg: dict[str, TopLevelGrammar]
    prompt: Optional[str]  # Optional with a default value
    messages: Optional[List[AssistantPrompt]]  # Optional with a default value
    temperature: Optional[float]  # Optional with a default value of 0.0
    top_p: Optional[float]  # Optional with a default value of 1.0
    top_k: Optional[int]  # Optional with a default value of -1
    max_tokens: Optional[int]  # Optional with a default value based on context size
