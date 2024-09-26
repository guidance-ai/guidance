"""State of models, including state tree that represents a guidance trace within Python.

The first implementation aims for simplicity.
Once benchmark figures are out, we'll figure out what to optimize.
"""
# TODO(nopdive): Deal with weak referencing for GC

from typing import Dict, Any
import weakref


class StateHandler:
    def __init__(self) -> None:
        self._model_node_map: Dict[int, StateNode] = {}
        self._node_model_map: Dict[StateNode, int] = {}
    
    def _remove_node(self, model_id: int) -> None:
        node = self._model_node_map.get(model_id, None)
        for child in node.children:
            child.parent = None
        if node.parent is not None:
            node.parent.remove_child(node)

        del self._model_node_map[model_id]
        del self._node_model_map[node]

    def update_node(self, model_id: int, parent_id: int, node: "StateNode") -> None:
        # Replace node if required
        existing_node = self._model_node_map.get(model_id, None)
        if existing_node is not None:
            self._remove_node(model_id)

        self._model_node_map[model_id] = node
        self._node_model_map[node] = model_id
        parent_node = self._model_node_map.get(parent_id, None)
        if parent_node is not None:
            parent_node.add_child(node)

_id_counter = 0
class StateNode:
    def __init__(self) -> None:
        self._id = self.__class__.gen_id()
        self._parent = None
        self._children = []

    def add_child(self, child: "StateNode") -> None:
        child._parent = self
        self._children.append(child)

    def remove_child(self, child: "StateNode") -> None:
        child._parent = None
        self._children.remove(child)

    @property
    def children(self):
        return self._children

    @property
    def parent(self):
        return self._parent

    @parent.setter
    def parent(self, value):
        self._parent = value

    @classmethod
    def gen_id(cls):
        global _id_counter
        _id = _id_counter
        _id_counter += 1
        return _id

    def __repr__(self):
        return self.__str__()


def visualize_text(state_builder: StateHandler, node: StateNode) -> None:
    def visit(node: StateNode, buffer=[]):
        if node.parent is not None:
            visit(node._parent)
        if isinstance(node, TextOutput):
            print(node._value, end='')
    visit(node)


def visualize(state_builder: StateHandler, node: StateNode) -> None:
    from anytree import Node, RenderTree

    def visit(node: StateNode, viz_parent=None):
        nonlocal state_builder

        if viz_parent is None:
            viz_node = Node(f"{state_builder._node_model_map[node]}:{node}")
        else:
            viz_node = Node(f"{state_builder._node_model_map[node]}:{node}", parent=viz_parent)

        for child in node.children:
            visit(child, viz_node)
        return viz_node
    viz_root = visit(node)

    for pre, fill, node in RenderTree(viz_root):
        tree_str = u"%s%s" % (pre, node.name)
        print(tree_str)


class InputNode(StateNode):
    def __init__(self) -> None:
        super().__init__()


class OutputNode(StateNode):
    def __init__(self) -> None:
        super().__init__()


class Tracker(StateNode):
    def __init__(self):
        super().__init__()

    def __str__(self):
        return f"{self._id}:Tracker"


class StatelessGuidanceInput(InputNode):
    def __init__(self, value: Any):
        self._value = value
        super().__init__()

    def __str__(self):
        return f"{self._id}:{self.__class__.__name__}:{self._value}"


class StatefulGuidanceInput(InputNode):
    def __init__(self, value: Any):
        self._value = value
        super().__init__()

    def __str__(self):
        return f"{self._id}:{self.__class__.__name__}:{self._value}"
    

class LiteralInput(InputNode):
    def __init__(self, value: str):
        self._value = value
        super().__init__()

    def __str__(self):
        return f"{self._id}:{self.__class__.__name__}:{repr(self._value)}"


class TextOutput(OutputNode):
    def __init__(self, value: str):
        self._value = value
        super().__init__()

    def __str__(self):
        return f"{self._id}:{self.__class__.__name__}:{repr(self._value)}"


class EmbeddedInput(InputNode):
    def __init__(self, value: str):
        self._value = value
        super().__init__()

    def __str__(self):
        return f"{self._id}:{self.__class__.__name__}:{repr(self._value)}"


class RoleOpenerInput(InputNode):
    def __init__(self, value: str):
        self._value = value
        super().__init__()

    def __str__(self):
        return f"{self._id}:{self.__class__.__name__}:{repr(self._value)}"


class RoleCloserInput(InputNode):
    def __init__(self, value: str):
        self._value = value
        super().__init__()

    def __str__(self):
        return f"{self._id}:{self.__class__.__name__}:{repr(self._value)}"

