""" Visualization related to trace. """

import json
from typing import Optional, Dict

from ..trace import (
    TextOutput,
    TokenOutput,
    TraceNode,
    TraceHandler,
    RoleOpenerInput,
    RoleCloserInput,
    ImageOutput,
    TokenOutput
)
import html


def trace_node_to_html(
    node: TraceNode, prettify_roles=False
) -> str:
    """Represents trace path as html string.

    Args:
        node: Trace node that designates the end of a trace path for HTML output.
        prettify_roles: Enables prettier formatting for roles.

    Returns:
        HTML string of trace path as html.
    """
    buffer = []
    node_path = list(node.path())
    active_role: Optional[TraceNode] = None

    for node in node_path:
        if isinstance(node.input, RoleOpenerInput):
            active_role = node
        elif isinstance(node.input, RoleCloserInput):
            active_role = node
        if isinstance(node.output, TextOutput):
            if active_role is not None:
                if prettify_roles and isinstance(active_role.input, RoleOpenerInput) and (role_name := active_role.input.name) is not None:
                    fmt = f"<div style='display: flex; border-bottom: 1px solid rgba(127, 127, 127, 0.2);  justify-content: center; align-items: center;'><div style='flex: 0 0 80px; opacity: 0.5;'>{role_name.lower()}</div><div style='flex-grow: 1; padding: 5px; padding-top: 10px; padding-bottom: 10px; margin-top: 0px; white-space: pre-wrap; margin-bottom: 0px;'>"
                    buffer.append(fmt)
                if not prettify_roles:
                    buffer.append("<span style='background-color: rgba(255, 180, 0, 0.3); border-radius: 3px;'>")

            if not (active_role and prettify_roles):
                attr = node.output
                latency = f"{attr.latency_ms:.2f}"
                chunk_text = attr.value

                if not isinstance(attr, TokenOutput):
                    if attr.is_generated:
                        fmt = f"<span style='background-color: rgba({0}, {255}, {0}, 0.15); border-radius: 3ps;' title='Chunk: {chunk_text}\nlatency_ms: {latency}'>{html.escape(chunk_text)}</span>"
                    elif attr.is_force_forwarded:
                        fmt = f"<span style='background-color: rgba({0}, {0}, {255}, 0.15); border-radius: 3ps;' title='Chunk: {chunk_text}\nlatency_ms: {latency}'>{html.escape(chunk_text)}</span>"
                    else:
                        fmt = f"<span style='background-color: rgba({255}, {255}, {255}, 0.15); border-radius: 3ps;' title='Chunk: {chunk_text}\nlatency_ms: {latency}'>{html.escape(chunk_text)}</span>"
                else:
                    token = attr.token
                    token_str = token.token
                    # assert token_str == chunk_text

                    prob = token.prob # TODO: what if nan?
                    top_k: dict[str, str] = {}
                    # find the correct token
                    for _token in (attr.top_k or []):
                        top_k[f"{_token.token}"] = f"{_token.prob} - Masked: {_token.masked}"
                    top_k_repr = json.dumps(top_k, indent=2)

                    if attr.is_generated:
                        fmt = f"<span style='background-color: rgba({0}, {127 + int(127 * prob)}, {0}, 0.15); border-radius: 3ps;' title='Token: \"{token_str}\" : {prob}\nTop-k: {top_k_repr}\nlatency_ms: {latency}'>{html.escape(token_str)}</span>"
                    elif attr.is_force_forwarded:
                        fmt = f"<span style='background-color: rgba({0}, {0}, {127 + int(127 * prob)}, 0.15); border-radius: 3ps;' title='Token: \"{token_str}\" : {prob}\nTop-k: {top_k_repr}\nlatency_ms: {latency}'>{html.escape(token_str)}</span>"
                    else:
                        fmt = f"<span style='background-color: rgba({255}, {255}, {255}, 0.15); border-radius: 3ps;' title='Token: \"{token_str}\" : {prob}\nTop-k: {top_k_repr}'>{html.escape(token_str)}</span>"

                buffer.append(fmt)

            if active_role is not None:
                if not prettify_roles:
                    buffer.append("</span>")
                if isinstance(active_role.input, RoleCloserInput) and prettify_roles:
                    buffer.append(f"</div></div>")
                active_role = None
        elif isinstance(node.output, ImageOutput):
            buffer.append(
                f'<img src="data:image/png;base64,{node.output.value.decode()}" style="max-width" 400px; vertical-align: middle; margin: 4px;">'
            )

    buffer.insert(
        0,
        "<pre style='margin: 0px; padding: 0px; vertical-align: middle; padding-left: 8px; margin-left: -8px; border-radius: 0px; border-left: 1px solid rgba(127, 127, 127, 0.2); white-space: pre-wrap; font-family: ColfaxAI, Arial; font-size: 15px; line-height: 23px;'>",
    )
    buffer.append("</pre>")
    return "".join(buffer)


def trace_node_to_str(node: TraceNode) -> str:
    """Visualize output attributes of a trace node up to the root.

    Users should not be accessing this function. For debugging purposes.

    Args:
        node: The trace node to visualize.
    Returns:
        Output as string.
    """
    buffer = []
    for node in node.path():
        if isinstance(node.output, TextOutput):
            buffer.append(str(node.output))
    return "".join(buffer)


def display_trace_tree(trace_handler: TraceHandler) -> None:
    """Visualize tree of a trace node going down to all its leaves.

    Users should not normally be accessing this function. For debugging purposes.

    Args:
        trace_handler: Trace handler needed to pull user-defined identifiers of trace nodes.
    """
    from anytree import Node, RenderTree # type: ignore[import-untyped]

    root = trace_handler.root()
    trace_viz_map: Dict[TraceNode, Node] = {}
    for node in root.traverse(bfs=False):
        viz_parent = trace_viz_map.get(node.parent, None)
        viz_node = Node(f"{trace_handler.node_id_map[node]}:{node!r}", parent=viz_parent)
        trace_viz_map[node] = viz_node
    viz_root = trace_viz_map[root]

    for pre, fill, node in RenderTree(viz_root):
        tree_str = "%s%s" % (pre, node.name)
        print(tree_str)
