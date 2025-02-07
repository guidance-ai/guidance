""" Visualization related to trace. """

import base64
import json
from typing import Optional, Dict
from .._schema import GenToken

from ..visual._message import TokensMessage
from ..trace import (
    TextOutput,
    TraceNode,
    TraceHandler,
    RoleOpenerInput,
    RoleCloserInput,
    ImageOutput,
)
import html


def trace_node_to_html(
    node: TraceNode, prettify_roles=False, complete_msg: TokensMessage = None
) -> str:
    """Represents trace path as html string.

    Args:
        node: Trace node that designates the end of a trace path for HTML output.
        prettify_roles: Enables prettier formatting for roles.
        complete_msg: Output message received on completion of engine.

    Returns:
        HTML string of trace path as html.
    """
    buffer = []
    node_path = list(node.path())
    active_role: Optional[TraceNode] = None

    prob_idx = 0
    # remaining_text = ""
    full_text = ""
    if complete_msg:
        for token in complete_msg.tokens:
            full_text += token.text

    for i, node in enumerate(node_path):
        if isinstance(node.input, RoleOpenerInput):
            active_role = node
        elif isinstance(node.input, RoleCloserInput):
            active_role = node

        if isinstance(node.output, TextOutput):
            if active_role is not None:
                if isinstance(active_role.input, RoleOpenerInput) and prettify_roles:
                    role_name = active_role.input.name
                    fmt = f"<div style='display: flex; border-bottom: 1px solid rgba(127, 127, 127, 0.2);  justify-content: center; align-items: center;'><div style='flex: 0 0 80px; opacity: 0.5;'>{role_name.lower()}</div><div style='flex-grow: 1; padding: 5px; padding-top: 10px; padding-bottom: 10px; margin-top: 0px; white-space: pre-wrap; margin-bottom: 0px;'>"
                    buffer.append(fmt)
                if not prettify_roles:
                    buffer.append("<span style='background-color: rgba(255, 180, 0, 0.3); border-radius: 3px;'>")

            if not (active_role and prettify_roles):
                attr = node.output

                fmt = ""
                if not complete_msg:
                    if attr.is_generated:
                        # fmt = f"<span style='background-color: rgba({165 * (1 - attr.prob)}, {165 * attr.prob}, 0, 0.15); border-radius: 3ps;' title='{attr.prob}'>{html.escape(attr.value)}</span>"
                        fmt = f"<span style='background-color: rgba({0}, {255}, {0}, 0.15); border-radius: 3ps;' title='{attr.prob}'>{html.escape(attr.value)}</span>"
                    elif attr.is_force_forwarded:
                        fmt = f"<span style='background-color: rgba({0}, {0}, {255}, 0.15); border-radius: 3ps;' title='{attr.prob}'>{html.escape(attr.value)}</span>"
                    else:
                        # fmt = f"{html.escape(attr.value)}"
                        fmt += f"<span style='background-color: rgba({255}, {255}, {255}, 0.15); border-radius: 3ps;' title='{attr.prob}'>{html.escape(attr.value)}</span>"
                else:
                    # switch to token-based
                    # cell_tokens = attr.tokens
                    # for token in cell_tokens:
                    #     # assert token.token == complete_msg.tokens[prob_idx].token, f"Token mismatch {token.token} != {complete_msg.tokens[prob_idx].token}"
                    #     if token.token_id != complete_msg.tokens[prob_idx].token_id:
                    #         if remaining_text + token.text != complete_msg.tokens[prob_idx].text:
                    #             remaining_text += token.text
                    #             continue
                    #         else:
                    #             remaining_text = ""

                    #     token_str = complete_msg.tokens[prob_idx].text
                    #     prob = complete_msg.tokens[prob_idx].prob
                    #     top_k = {}
                    #     # find the correct token
                    #     for _item in complete_msg.tokens[prob_idx].top_k:
                    #         top_k[f"{_item.text}"] = f"{_item.prob} - Masked: {_item.is_masked}"
                    #     top_k = json.dumps(top_k, indent=2)

                    #     latency = f"{complete_msg.tokens[prob_idx].latency_ms:.2f}"

                    #     if complete_msg.tokens[prob_idx].is_generated:
                    #         fmt += f"<span style='background-color: rgba({0}, {127 + int(127 * prob)}, {0}, 0.15); border-radius: 3ps;' title='Token: \"{token_str}\" : {prob}\nTop-k: {top_k}\nlatency_ms: {latency}'>{html.escape(token_str)}</span>"
                    #     elif complete_msg.tokens[prob_idx].is_force_forwarded:
                    #         fmt += f"<span style='background-color: rgba({0}, {0}, {127 + int(127 * prob)}, 0.15); border-radius: 3ps;' title='Token: \"{token_str}\" : {prob}\nTop-k: {top_k}\nlatency_ms: {latency}'>{html.escape(token_str)}</span>"
                    #     else:
                    #         fmt += f"<span style='background-color: rgba({255}, {255}, {255}, 0.15); border-radius: 3ps;' title='Token: \"{token_str}\" : {prob}\nTop-k: {top_k}'>{html.escape(token_str)}</span>"

                    #     full_text = full_text[len(token_str) :]
                    #     prob_idx += 1

                    chunk_text = attr.value
                    # find tokens in complete message that cover this chunk
                    tokens: list[GenToken] = []
                    _idx = prob_idx
                    tokens_text = ""
                    while _idx < len(complete_msg.tokens):
                        _token = complete_msg.tokens[_idx]
                        tokens_text += _token.text
                        tokens.append(_token)
                        if chunk_text in tokens_text:
                            break
                        _idx += 1

                    assert (
                        chunk_text in tokens_text
                    ), f"Token mismatch {tokens_text} != {chunk_text}"

                    start_idx = tokens_text.index(chunk_text)
                    remaining_text = tokens_text[start_idx + len(chunk_text) :]

                    if remaining_text:
                        # remove the last tokens
                        tokens.pop(-1)

                    # update prob_idx
                    prob_idx += len(tokens)

                    for token in tokens:
                        token_str = token.text
                        prob = token.prob
                        top_k = {}
                        # find the correct token
                        for _item in token.top_k:
                            top_k[f"{_item.text}"] = f"{_item.prob} - Masked: {_item.is_masked}"
                        top_k = json.dumps(top_k, indent=2)

                        latency = f"{token.latency_ms:.2f}"

                        if token.is_generated:
                            fmt += f"<span style='background-color: rgba({0}, {127 + int(127 * prob)}, {0}, 0.15); border-radius: 3ps;' title='Token: \"{token_str}\" : {prob}\nTop-k: {top_k}\nlatency_ms: {latency}'>{html.escape(token_str)}</span>"
                        elif token.is_force_forwarded:
                            fmt += f"<span style='background-color: rgba({0}, {0}, {127 + int(127 * prob)}, 0.15); border-radius: 3ps;' title='Token: \"{token_str}\" : {prob}\nTop-k: {top_k}\nlatency_ms: {latency}'>{html.escape(token_str)}</span>"
                        else:
                            fmt += f"<span style='background-color: rgba({255}, {255}, {255}, 0.15); border-radius: 3ps;' title='Token: \"{token_str}\" : {prob}\nTop-k: {top_k}'>{html.escape(token_str)}</span>"

                buffer.append(fmt)

            if active_role is not None:
                if not prettify_roles:
                    buffer.append("</span>")
                if isinstance(active_role.input, RoleCloserInput) and prettify_roles:
                    buffer.append(f"</div></div>")
                active_role = None
        elif isinstance(node.output, ImageOutput):
            encoded = base64.b64encode(node.output.value).decode()
            buffer.append(
                f'<img src="data:image/png;base64,{encoded}" style="max-width" 400px; vertical-align: middle; margin: 4px;">'
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
    from anytree import Node, RenderTree

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
