import logging
from typing import Optional
from pydantic import BaseModel

from ..trace import TraceHandler
from ..visual import Message, TraceMessage, ResetDisplayMessage
from ._trace import trace_node_to_html

logger = logging.getLogger(__name__)

try:
    from IPython.display import clear_output, display, HTML
    from IPython import get_ipython

    ipython_imported = True
except ImportError:
    ipython_imported = False


class RenderUpdate(BaseModel):
    messages: list = []
    need_new_display: bool = False


class UpdateController:
    def __init__(self, trace_handler: TraceHandler):
        self._trace_handler = trace_handler
        self._messages: list[Message] = []

        self._prev_trace_id: Optional[int] = None
        self._prev_cell_msg_id: Optional[int] = None
        self._prev_cell_exec_count: Optional[int] = None

    def update(self, message: Message) -> RenderUpdate:
        if not isinstance(message, TraceMessage):
            return RenderUpdate()

        # if type(message.node_attr) not in [TextOutput, RoleOpenerInput, RoleCloserInput]:
        #     return RenderUpdate()

        trace_node = message.trace_node
        need_reset = False
        need_new_display = False

        if len(self._messages) == 0:
            # If no messages, reset
            logger.debug("NEED_RESET:empty")
            need_reset = True
            need_new_display = True
        else:
            # If we diverge from the model path, truncate and reset
            last_trace_message = next(
                x for x in reversed(self._messages) if isinstance(x, TraceMessage)
            )
            last_trace_node = last_trace_message.trace_node

            if trace_node not in last_trace_node.path():
                logger.debug(f"NEED_RESET:divergence:{last_trace_node}:{trace_node}")
                need_reset = True

                # Truncate path that is no longer used by current trace node
                ancestor_idx = -1
                ancestors = set(trace_node.ancestors())
                for idx, prev_message in enumerate(self._messages):
                    if isinstance(prev_message, TraceMessage):
                        if prev_message.trace_node in ancestors:
                            ancestor_idx = idx
                if ancestor_idx == -1:
                    logger.debug(f"PARENT_NOT_FOUND:{trace_node}")
                    self._messages.clear()
                else:
                    self._messages = self._messages[:ancestor_idx]

        # If we are in a new Jupyter cell or execution, reset
        if ipython_imported:
            ipy = get_ipython()
            cell_msg_id = ipy.get_parent()["msg_id"]
            cell_exec_count = ipy.execution_count
            if (
                cell_msg_id != self._prev_cell_msg_id
                or cell_exec_count != self._prev_cell_exec_count
            ):
                need_reset = True
                need_new_display = True
                logger.debug(f"NEED_RESET:jupyter:{cell_msg_id}|{cell_exec_count}")
                self._prev_cell_msg_id = cell_msg_id
                self._prev_cell_exec_count = cell_exec_count
                self._messages = []

        out_messages = []
        # Add previous messages if reset required
        if need_reset:
            out_messages.append(ResetDisplayMessage())
            out_messages.extend(self._messages)
        # Add current message
        out_messages.append(message)
        self._messages.append(message)

        return RenderUpdate(messages=out_messages, need_new_display=need_new_display)


class Renderer:
    """Renders guidance model to a visual medium."""

    def update(self, message: Message) -> None:
        raise NotImplementedError("Update not implemented.")


class JupyterHtmlRenderer(Renderer):
    def __init__(self, trace_handler: TraceHandler) -> None:
        self._update_controller = UpdateController(trace_handler)
        self._formatted = []

    def update(self, message: Message) -> None:
        render_update = self._update_controller.update(message)

        last_trace_node = None
        for out_message in render_update.messages:
            if isinstance(out_message, TraceMessage):
                last_trace_node = out_message.trace_node

        if last_trace_node is not None:
            clear_output(wait=True)
            display(HTML(trace_node_to_html(last_trace_node, prettify_roles=False)))


class AutoRenderer(Renderer):
    def __init__(self, trace_handler: TraceHandler):
        # TODO(nopdive): Add autodetect logic.
        self._renderer = JupyterHtmlRenderer(trace_handler=trace_handler)

    def update(self, message: Message) -> None:
        if self._renderer is not None:
            self._renderer.update(message)
