"""Trace tree of inputs & outputs generated from a guidance program.

The first implementation aims for simplicity.
Once benchmark figures are out, we'll figure out what to optimize.

The most critical class is the trace handler. See its documentation for trace design & motivations.
"""
# TODO(nopdive): Deal with weak referencing for GC. Take care of cyclical dependencies from parent & children fields.
# TODO(nopdive): Benchmark (expected heap fragmentation issue). Likely need memory pooling (via rust/ctypes/Cython).
# NOTE(nopdive): Should we be explicit on capture variables? Right now they are within guidance grammars.

from pydantic import BaseModel, Field
from itertools import count
from typing import Dict, Any, Optional
from guidance._utils import pydantic_no_default_repr, pydantic_no_default_str


class NodeAttr(BaseModel):
    """Attributes of a trace node."""

    def __repr__(self):
        return pydantic_no_default_repr(self)

    def __str__(self):
        return pydantic_no_default_str(self)


class InputAttr(NodeAttr):
    """Input for a guidance program (i.e. literal or guidance grammar)."""
    pass


class OutputAttr(NodeAttr):
    """Output for a guidance program (i.e. text output)."""
    pass


class StatelessGuidanceInput(InputAttr):
    """Stateless guidance input (light wrapper)."""
    value: Any

    def __repr__(self):
        return f"{self.__class__.__name__}({self.value})"


class StatefulGuidanceInput(InputAttr):
    """Stateful guidance input (light wrapper)."""
    value: Any

    def __repr__(self):
        return f"{self.__class__.__name__}({self.value})"


class LiteralInput(InputAttr):
    """Text string as a literal."""
    value: str


class EmbeddedInput(InputAttr):
    """Text string with embedded guidance input."""
    value: str


class RoleOpenerInput(InputAttr):
    """Appears when a role is opened (i.e. user / system).

    This usually occurs as a role context and __enter__ is called.
    """
    name: str


class RoleCloserInput(InputAttr):
    """Appears when a role is closed (i.e. user / system).

    This usually occurs as a role context and __exit__ is called.
    """
    name: str


class TextOutput(OutputAttr):
    """Text string."""
    value: str
    is_generated: bool = False
    token_count: int = 0
    prob: float = 0.0


class CaptureOutput(OutputAttr):
    """Capture variable output as a string.

    If `value` is set to None, this means it's a reset (needed for append capture group outputs).
    """
    name: str
    value: Optional[str] = None
    is_append: bool = False
    log_probs: float = 0.0

    def __str__(self):
        return f"{self.name}{'+=' if self.is_append else '='}{self.value.__str__()}"


class TraceNode(BaseModel):
    """Trace node which associates inputs and outputs of a guidance program."""
    identifier: int = Field(default_factory=count().__next__)
    parent: Optional["TraceNode"] = None
    children: list["TraceNode"] = []
    input: Optional[InputAttr] = None
    output: Optional[OutputAttr] = None

    def add_child(self, child: "TraceNode") -> None:
        """ Add a child node to the trace node.

        Args:
            child: The child node to add.
        """
        child.parent = self
        self.children.append(child)

    def remove_child(self, child: "TraceNode") -> None:
        """ Remove a child node from the trace node.

        Args:
            child: The child node to remove.
        """
        child.parent = None
        self.children.remove(child)

    def __repr__(self):
        return f"{self.identifier}:{self.input!r}:{self.output!r}"

    def __hash__(self):
        return hash(self.identifier)


class TraceHandler(BaseModel):
    """Trace handler that will own a tree of trace nodes.

    This will primarily be owned by a model's engine.
    Each guidance model corresponds to one trace node.
    All guidance models emitted from an engine is to be included as their own paths within the tree.

    The requirement for holding all live traces ensures downstream consumers such as UI providers
    can do near-real-time partial updates.
    """
    id_node_map: Dict[int, TraceNode] = {}
    node_id_map: Dict[TraceNode, int] = {}

    def update_node(self, identifier: int, parent_id: int, node_attr: NodeAttr = None) -> None:
        """ Update the trace node with the given identifier.

        If the trace node does not exist, it will be created.
        Both parent id and node attributes can be updated only once until further notice.

        Args:
            identifier: User-defined identifier for the trace node.
            parent_id: User-defined parent identifier for the trace node.
            node_attr: Input or output node attribute to be updated on the trace node.
        """
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


def visualize_output(node: TraceNode) -> None:
    """ Visualize output attributes of a trace node up to the root.

    Users should not be accessing this function. For debugging purposes.

    Args:
        node: The trace node to visualize.
    """
    def visit(visitor: TraceNode):
        if visitor.parent is not None:
            visit(visitor.parent)

        if visitor.output is not None and isinstance(visitor.output, TextOutput):
            print(visitor.output, end='')
    visit(node)


def visualize(trace_handler: TraceHandler, node: TraceNode) -> None:
    """ Visualize tree of a trace node going down to all its leaves.

    Users should not be accessing this function. For debugging purposes.

    Args:
        trace_handler: Trace handler needed to pull user-defined identifiers of trace nodes.
        node: Trace node that will function as the root.
    """
    from anytree import Node, RenderTree

    def visit(visitor: TraceNode, viz_parent=None):
        nonlocal trace_handler

        if viz_parent is None:
            viz_node = Node(f"{trace_handler.node_id_map[visitor]}:{visitor!r}")
        else:
            viz_node = Node(f"{trace_handler.node_id_map[visitor]}:{visitor!r}", parent=viz_parent)

        for child in visitor.children:
            visit(child, viz_node)
        return viz_node
    viz_root = visit(node)

    for pre, fill, node in RenderTree(viz_root):
        tree_str = u"%s%s" % (pre, node.name)
        print(tree_str)


