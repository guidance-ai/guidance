"""State of models, including state tree that represents a guidance trace within Python.

The first implementation aims for simplicity.
Once benchmark figures are out, we'll figure out what to optimize.
"""
# TODO(nopdive): Deal with weak referencing for GC

from dataclasses import dataclass, field
from itertools import count
from typing import Dict, Any, Optional
import weakref


class StateHandler:
    def __init__(self) -> None:
        self._model_node_map: Dict[int, StateNode] = {}
        self._node_model_map: Dict[StateNode, int] = {}

    def update_node(self, model_id: int, parent_id: int, node_attr: "NodeAttr" = None) -> None:
        node = self._model_node_map.get(model_id, None)
        if node is None:
            node = StateNode()

            self._model_node_map[model_id] = node
            self._node_model_map[node] = model_id

            parent_node = self._model_node_map.get(parent_id, None)
            if parent_node is not None:
                parent_node.add_child(node)

        if node_attr is not None:
            if isinstance(node_attr, InputAttr):
                assert node.input is None
                node.input = node_attr
            elif isinstance(node_attr, OutputAttr):
                assert node.output is None
                node.output = node_attr
            else:
                raise ValueError(f"Unexpected node attr: {node_attr}")


@dataclass
class StateNode:
    identifier: int = field(default_factory=count().__next__)
    parent: Optional["StateNode"] = None
    children: list["StateNode"] = field(default_factory=list)
    input: Optional["InputAttr"] = None
    output: Optional["OutputAttr"] = None

    def add_child(self, child: "StateNode") -> None:
        child.parent = self
        self.children.append(child)

    def remove_child(self, child: "StateNode") -> None:
        child.parent = None
        self.children.remove(child)

    def __repr__(self):
        return f"{self.identifier}:{'' if self.input is None else repr(self.input)}:{'' if self.output is None else repr(self.output)}"

    def __hash__(self):
        return hash(self.identifier)


def visualize_text(state_builder: StateHandler, node: StateNode) -> None:
    def visit(visitor: StateNode):
        if visitor.parent is not None:
            visit(visitor.parent)

        if visitor.output is not None and isinstance(visitor.output, OutputAttr):
            print(visitor.output._value, end='')

    visit(node)


def visualize(state_builder: StateHandler, node: StateNode) -> None:
    from anytree import Node, RenderTree

    def visit(visitor: StateNode, viz_parent=None):
        nonlocal state_builder

        if viz_parent is None:
            viz_node = Node(f"{state_builder._node_model_map[visitor]}:{repr(visitor)}")
        else:
            viz_node = Node(f"{state_builder._node_model_map[visitor]}:{repr(visitor)}", parent=viz_parent)

        for child in visitor.children:
            visit(child, viz_node)
        return viz_node
    viz_root = visit(node)

    for pre, fill, node in RenderTree(viz_root):
        tree_str = u"%s%s" % (pre, node.name)
        print(tree_str)


class NodeAttr:
    pass


class InputAttr(NodeAttr):
    def __init__(self) -> None:
        super().__init__()


class OutputAttr(NodeAttr):
    def __init__(self) -> None:
        super().__init__()


class StatelessGuidanceInput(InputAttr):
    def __init__(self, value: Any):
        self._value = value
        super().__init__()

    def __repr__(self):
        return f"{self.__class__.__name__}:{self._value}"

    def __str__(self):
        return str(self._value)


class StatefulGuidanceInput(InputAttr):
    def __init__(self, value: Any):
        self._value = value
        super().__init__()

    def __repr__(self):
        return f"{self.__class__.__name__}:{self._value}"

    def __str__(self):
        return str(self._value)


class LiteralInput(InputAttr):
    def __init__(self, value: str):
        self._value = value
        super().__init__()

    def __repr__(self):
        return f"{self.__class__.__name__}:{repr(self._value)}"

    def __str__(self):
        return self._value


class TextOutput(OutputAttr):
    def __init__(self, value: str):
        self._value = value
        super().__init__()

    def __repr__(self):
        return f"{self.__class__.__name__}:{repr(self._value)}"

    def __str__(self):
        return self._value


class EmbeddedInput(InputAttr):
    def __init__(self, value: str):
        self._value = value
        super().__init__()

    def __repr__(self):
        return f"{self.__class__.__name__}:{repr(self._value)}"

    def __str__(self):
        return self._value

class RoleOpenerInput(InputAttr):
    def __init__(self, value: str):
        self._value = value
        super().__init__()

    def __repr__(self):
        return f"{self.__class__.__name__}:{repr(self._value)}"

    def __str__(self):
        return self._value


class RoleCloserInput(InputAttr):
    def __init__(self, value: str):
        self._value = value
        super().__init__()

    def __repr__(self):
        return f"{self.__class__.__name__}:{repr(self._value)}"

    def __str__(self):
        return self._value

