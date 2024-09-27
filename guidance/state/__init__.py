"""State of models, including state tree that represents a guidance trace within Python.

The first implementation aims for simplicity.
Once benchmark figures are out, we'll figure out what to optimize.
"""
# TODO(nopdive): Deal with weak referencing for GC

from dataclasses import dataclass, field, fields
from itertools import count
from typing import Dict, Any, Optional
import weakref


def _dataclass_repr(obj, target_fields=None):
    if target_fields is None:
        records = (
            f'{field.name}={getattr(obj, field.name)!r}'
            for field in fields(obj)
            if getattr(obj, field.name) != field.default
        )
    else:
        records = (
            f'{field.name}={getattr(obj, field.name)!r}'
            for field in fields(obj)
            if getattr(obj, field.name) != field.default and field.name in target_fields
        )
    out = f'{type(obj).__name__}({", ".join(records)})'
    return out


def _dataclass_str(obj, target_fields=None):
    if target_fields is None:
        records = (
            f'{getattr(obj, field.name)!s}'
            for field in fields(obj)
            if getattr(obj, field.name) != field.default
        )
    else:
        records = (
            f'{getattr(obj, field.name)!s}'
            for field in fields(obj)
            if getattr(obj, field.name) != field.default and field.name in target_fields
        )
    out = ", ".join(records)
    return out


@dataclass
class ModelTraceHandler:
    model_node_map: Dict[int, "TraceNode"] = field(default_factory=dict)
    node_model_map: Dict["TraceNode", int] = field(default_factory=dict)

    def update_node(self, model_id: int, parent_id: int, node_attr: "NodeAttr" = None) -> None:
        node = self.model_node_map.get(model_id, None)
        if node is None:
            node = TraceNode()

            self.model_node_map[model_id] = node
            self.node_model_map[node] = model_id

            parent_node = self.model_node_map.get(parent_id, None)
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
class TraceNode:
    identifier: int = field(default_factory=count().__next__)
    parent: Optional["TraceNode"] = None
    children: list["TraceNode"] = field(default_factory=list)
    input: Optional["InputAttr"] = None
    output: Optional["OutputAttr"] = None

    def add_child(self, child: "TraceNode") -> None:
        child.parent = self
        self.children.append(child)

    def remove_child(self, child: "TraceNode") -> None:
        child.parent = None
        self.children.remove(child)

    def __repr__(self):
        return _dataclass_repr(self, ["identifier", "input", "output"])

    def __hash__(self):
        return hash(self.identifier)


def visualize_text(state_builder: ModelTraceHandler, node: TraceNode) -> None:
    def visit(visitor: TraceNode):
        if visitor.parent is not None:
            visit(visitor.parent)

        if visitor.output is not None and isinstance(visitor.output, OutputAttr):
            print(visitor.output, end='')

    visit(node)


def visualize(state_builder: ModelTraceHandler, node: TraceNode) -> None:
    from anytree import Node, RenderTree

    def visit(visitor: TraceNode, viz_parent=None):
        nonlocal state_builder

        if viz_parent is None:
            viz_node = Node(f"{state_builder.node_model_map[visitor]}:{visitor !r}")
        else:
            viz_node = Node(f"{state_builder.node_model_map[visitor]}:{visitor !r}", parent=viz_parent)

        for child in visitor.children:
            visit(child, viz_node)
        return viz_node
    viz_root = visit(node)

    for pre, fill, node in RenderTree(viz_root):
        tree_str = u"%s%s" % (pre, node.name)
        print(tree_str)


@dataclass(frozen=True)
class NodeAttr:
    def __repr__(self):
        return _dataclass_repr(self)

    def __str__(self):
        return _dataclass_str(self)


@dataclass(frozen=True)
class InputAttr(NodeAttr):
    pass


@dataclass(frozen=True)
class OutputAttr(NodeAttr):
    pass


@dataclass(frozen=True)
class StatelessGuidanceInput(InputAttr):
    value: Any


@dataclass(frozen=True)
class StatefulGuidanceInput(InputAttr):
    value: Any

    def __repr__(self):
        return f"{self.__class__.__name__}({repr(self.value)})"


@dataclass(frozen=True)
class LiteralInput(InputAttr):
    value: str


@dataclass(frozen=True)
class EmbeddedInput(InputAttr):
    value: str


@dataclass(frozen=True)
class RoleOpenerInput(InputAttr):
    value: str


@dataclass(frozen=True)
class RoleCloserInput(InputAttr):
    value: str


@dataclass(frozen=True)
class TextOutput(OutputAttr):
    value: str
