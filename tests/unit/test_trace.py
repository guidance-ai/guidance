import pytest

from guidance.trace._trace import WeakRefList, TraceHandler, LiteralInput, TextOutput, RoleCloserInput
from guidance.trace import (
    TraceNode,
    StatelessGuidanceInput,
    StatefulGuidanceInput,
    ImageInput,
    EmbeddedInput,
    RoleOpenerInput,
    ImageOutput,
    CaptureOutput,
)


def test_weak_ref_list():
    class EmptyClass:
        pass

    a = EmptyClass()
    b = EmptyClass()
    li = WeakRefList()
    li.append(a)
    li.append(b)

    del a
    with pytest.raises(ReferenceError):
        _ = li[0]

    # Does not remove dead entries
    _ = li[1]
    assert len(li) == 2

    # Remove works as expected
    li.remove(b)
    assert len(li) == 1

    # Iter goes over live entries only
    for el in li:
        _ = el


def test_trace_node():
    root = TraceNode()
    child1 = TraceNode()
    child2 = TraceNode()
    root.add_child(child1)
    root.add_child(child2)

    assert root.root() is root
    assert list(root.ancestors()) == []
    assert list(root.path()) == [root]
    assert list(root.traverse()) == [root, child1, child2]

    assert child1.root() is root
    assert list(child1.ancestors()) == [root]
    assert list(child1.path()) == [root, child1]
    assert list(child1.traverse()) == [child1]


def test_trace_handler():
    trace_handler = TraceHandler()
    root = trace_handler.update_node(0, None, None)
    child1 = trace_handler.update_node(1, 0, None)
    inp = LiteralInput(value="")
    out = TextOutput(value="")
    pre_child2 = trace_handler.update_node(2, 0, inp)
    child2 = trace_handler.update_node(2, 0, out)

    assert pre_child2 == child2
    assert child2.input == inp
    assert child2.output == out
    assert child2.root() == root
    assert child1 not in child2.path()


@pytest.mark.parametrize(
    "node",
    [
        StatelessGuidanceInput(value=None),
        StatefulGuidanceInput(value=None),
        LiteralInput(value=""),
        ImageInput(value=b""),
        EmbeddedInput(value=""),
        RoleOpenerInput(name=""),
        RoleCloserInput(name=""),
        TextOutput(value=""),
        ImageOutput(value=b""),
        CaptureOutput(name=""),
    ],
)
def test_node_format_smoke(node):
    node.__repr__()
    node.__str__()
