import pytest
from itertools import chain

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
    if isinstance(g1, (Byte, ByteRange, ModelVariable)):
        slots = chain.from_iterable(getattr(cls, '__slots__', []) for cls in type(g1).mro())
        return all(getattr(g1, slot) == getattr(g2, slot) for slot in slots)
    elif isinstance(g1, (Join, Select)):
        slots = chain.from_iterable(getattr(cls, '__slots__', []) for cls in type(g1).mro())
        return (all(getattr(g1, slot) == getattr(g2, slot) for slot in slots if 'values' not in slot)
            and len(g1.values) == len(g2.values) # Check both have same number of child nodes
            and all(compare_grammars(v1, v2) for v1, v2 in zip(g1.values, g2.values)) # Recursively compare child nodes
        )
    else:
        raise ValueError(f"Unsupported grammar type: {type(g1)}")


@pytest.mark.xfail(reason="No protobuf support for new lazy grammars")
@pytest.mark.parametrize(
    "grammar",
    [
        string("Hello, world!"),
        Byte(b"a"),
        byte_range(b"\x00", b"\xff"),
        char_range("a", "z"),
        select(["option1", "option2", "option3"]),
        commit_point(string("commit"), hidden=False),
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
    serialized_grammar = grammar.serialize()
    deserialized_grammar = GrammarFunction.deserialize(serialized_grammar)

    # Recursively compare the grammars
    assert compare_grammars(
        grammar, deserialized_grammar
    ), f"Deserialized grammar does not match original:\nOriginal: {grammar}\nDeserialized: {deserialized_grammar}\n"

