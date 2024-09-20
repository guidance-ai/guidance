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

    def add_node(self, model_id: int, parent_id: int, node: "StateNode") -> None:
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

    @classmethod
    def gen_id(cls):
        global _id_counter
        _id = _id_counter
        _id_counter += 1
        return _id


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


class TrackingNode(StateNode):
    def __init__(self) -> None:
        super().__init__()


class Nop(TrackingNode):
    def __init__(self):
        super().__init__()

    def __str__(self):
        return f"{self._id}:Nop"


class GuidanceInput(StateNode):
    def __init__(self, value: Any):
        self._value = value
        super().__init__()

    def __str__(self):
        return f"{self._id}:{self.__class__.__name__}:{self._value.__class__.__name__}:{self._value}"


class LiteralInput(InputNode):
    def __init__(self, value: str):
        self._value = value
        super().__init__()

    def __str__(self):
        return f"{self._id}:{self.__class__.__name__}:{self._value}"


class TextOutput(OutputNode):
    def __init__(self, value: str):
        self._value = value
        super().__init__()

    def __str__(self):
        return f"{self._id}:{self.__class__.__name__}:{self._value}"


class EmbeddedInput(InputNode):
    def __init__(self, value: str):
        self._value = value
        super().__init__()

    def __str__(self):
        return f"{self._id}:{self.__class__.__name__}:{self._value}"


# class Env:
#     def __init__(self, parent: Optional["Env"] = None) -> None:
#         self._symbol_table = {}
#         self._parent = parent
#
#     def __getitem__(self, item: Any) -> Any:
#         current_env = self
#         while current_env._parent is not None:
#             if item in current_env._symbol_table:
#                 return current_env[item]
#         raise KeyError(item)


