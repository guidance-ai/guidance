import pytest

from guidance import (
    byte_range,
    char_range,
    commit_point,
    select,
    string,
    token_limit,
    with_temperature,
)
from guidance._grammar import (
    Byte,
    ByteRange,
    GrammarFunction,
    Join,
    ModelVariable,
    Select,
)


def compare_grammars(g1: GrammarFunction, g2: GrammarFunction) -> bool:
    """Recursively compare two GrammarFunction objects for equivalence."""
    
    if type(g1) != type(g2):
        return False

    # Compare attributes based on type
    if isinstance(g1, ModelVariable):
        return (
            g1.name == g2.name
            and g1.hidden == g2.hidden
            and g1.commit_point == g2.commit_point
            and g1.capture_name == g2.capture_name
        )
    elif isinstance(g1, (Byte, ByteRange)):
        return (
            g1.name == g2.name
            and g1.hidden == g2.hidden
            and g1.commit_point == g2.commit_point
            and g1.capture_name == g2.capture_name
            and g1.temperature == g2.temperature
            and g1.max_tokens == g2.max_tokens
        )
    elif isinstance(g1, (Join, Select)):
        return (
            g1.name == g2.name
            and g1.hidden == g2.hidden
            and g1.commit_point == g2.commit_point
            and g1.capture_name == g2.capture_name
            and g1.max_tokens == g2.max_tokens
            and len(g1.values) == len(g2.values) # Check both have same number of child nodes
            and all(compare_grammars(v1, v2) for v1, v2 in zip(g1.values, g2.values)) # Recursively compare child nodes
        )

    else:
        raise ValueError(f"Unsupported grammar type: {type(g1)}")


@pytest.mark.parametrize(
    "grammar",
    [
        string("Hello, world!"),
        Byte(b"a"),
        byte_range(b"\x00", b"\xff"),
        char_range("a", "z"),
        select(["option1", "option2", "option3"]),
        commit_point(string("commit"), hidden=True),
        token_limit(string("limited"), max_tokens=5),
        with_temperature(string("temp"), temperature=0.5),
        ModelVariable("my_variable"),
        Join([string("part1"), string("part2")]),
        select(
            [
                string("option1"),
                Join([string("part1"), string("part2")]),
            ]
        ),
    ],
)
def test_grammar_protobuf_roundtrip(grammar: GrammarFunction):
    """Test that grammars can be round-tripped through protobuf serialization."""

    # Serialize the grammar to a protobuf message
    serialized_grammar = grammar.serialize()

    # Deserialize the grammar from the protobuf message
    deserialized_grammar = GrammarFunction.deserialize(serialized_grammar)

    # Recursively compare the grammars
    assert compare_grammars(
        grammar, deserialized_grammar
    ), f"Deserialized grammar does not match original:\nOriginal: {grammar}\nDeserialized: {deserialized_grammar}\n"

