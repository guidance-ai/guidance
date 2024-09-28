from ..trace import TextOutput, TraceNode, TraceHandler


def trace_node_to_str(node: TraceNode) -> str:
    """ Visualize output attributes of a trace node up to the root.

    Users should not be accessing this function. For debugging purposes.

    Args:
        node: The trace node to visualize.
    Returns:
        Output as string.
    """
    def visit(visitor: TraceNode, buffer: list):
        if visitor.parent is not None:
            visit(visitor.parent, buffer)

        if visitor.output is not None and isinstance(visitor.output, TextOutput):
            buffer.append(str(visitor.output))

    results = []
    visit(node, results)
    return ''.join(results)


def trace_node_to_tree(trace_handler: TraceHandler, node: TraceNode) -> None:
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
