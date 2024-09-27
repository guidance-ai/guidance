"""Trace tree of inputs & outputs generated from a guidance program.

The first implementation aims for simplicity.
Once benchmark figures are out, we'll figure out what to optimize.
"""
# TODO(nopdive): Deal with weak referencing for GC

from pydantic import BaseModel, Field
from itertools import count
from typing import Dict, Any, Optional
from guidance._utils import pydantic_no_default_repr, pydantic_no_default_str


class NodeAttr(BaseModel):
    def __repr__(self):
        return pydantic_no_default_repr(self)

    def __str__(self):
        return pydantic_no_default_str(self)


class InputAttr(NodeAttr):
    pass


class OutputAttr(NodeAttr):
    pass


class StatelessGuidanceInput(InputAttr):
    value: Any

    def __repr__(self):
        return f"{self.__class__.__name__}({self.value})"


class StatefulGuidanceInput(InputAttr):
    value: Any

    def __repr__(self):
        return f"{self.__class__.__name__}({self.value})"


class LiteralInput(InputAttr):
    value: str


class EmbeddedInput(InputAttr):
    value: str


class RoleOpenerInput(InputAttr):
    name: str


class RoleCloserInput(InputAttr):
    name: str


class TextOutput(OutputAttr):
    value: str
    is_generated: bool = False
    token_count: int = 0
    prob: float = 0.0


class CaptureOutput(OutputAttr):
    name: str
    value: Optional[str] = None
    is_append: bool = False
    log_probs: float = 0.0

    def __str__(self):
        return f"{self.name}{'+=' if self.is_append else '='}{self.value.__str__()}"


class TraceNode(BaseModel):
    identifier: int = Field(default_factory=count().__next__)
    parent: Optional["TraceNode"] = None
    children: list["TraceNode"] = []
    input: Optional[InputAttr] = None
    output: Optional[OutputAttr] = None

    def add_child(self, child: "TraceNode") -> None:
        child.parent = self
        self.children.append(child)

    def remove_child(self, child: "TraceNode") -> None:
        child.parent = None
        self.children.remove(child)

    def __repr__(self):
        return f"{self.identifier}:{self.input!r}:{self.output!r}"

    def __hash__(self):
        return hash(self.identifier)


class TraceHandler(BaseModel):
    id_node_map: Dict[int, "TraceNode"] = {}
    node_id_map: Dict["TraceNode", int] = {}

    def update_node(self, identifier: int, parent_id: int, node_attr: "NodeAttr" = None) -> None:
        node = self.id_node_map.get(identifier, None)
        if node is None:
            node = TraceNode()

            self.id_node_map[identifier] = node
            self.node_id_map[node] = identifier

            parent_node = self.id_node_map.get(parent_id, None)
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


def visualize_text(state_builder: TraceHandler, node: TraceNode) -> None:
    def visit(visitor: TraceNode):
        if visitor.parent is not None:
            visit(visitor.parent)

        if visitor.output is not None and isinstance(visitor.output, OutputAttr):
            print(visitor.output, end='')

    visit(node)


def visualize(state_builder: TraceHandler, node: TraceNode) -> None:
    from anytree import Node, RenderTree

    def visit(visitor: TraceNode, viz_parent=None):
        nonlocal state_builder

        if viz_parent is None:
            viz_node = Node(f"{state_builder.node_id_map[visitor]}:{visitor!r}")
        else:
            viz_node = Node(f"{state_builder.node_id_map[visitor]}:{visitor!r}", parent=viz_parent)

        for child in visitor.children:
            visit(child, viz_node)
        return viz_node
    viz_root = visit(node)

    for pre, fill, node in RenderTree(viz_root):
        tree_str = u"%s%s" % (pre, node.name)
        print(tree_str)


