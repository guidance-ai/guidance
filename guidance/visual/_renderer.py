import html
import logging
from typing import Optional
from pydantic import BaseModel
from ..trace import RoleOpenerInput, RoleCloserInput, TextOutput, TraceHandler
from ..visual import Message, TraceMessage, ResetDisplayMessage

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
            last_trace_message = next(x for x in reversed(self._messages) if isinstance(x, TraceMessage))
            last_trace_node = last_trace_message.trace_node

            if trace_node != last_trace_node and last_trace_node not in trace_node.ancestors():
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
            cell_msg_id = ipy.get_parent()['msg_id']
            cell_exec_count = ipy.execution_count
            if cell_msg_id != self._prev_cell_msg_id or cell_exec_count != self._prev_cell_exec_count:
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
        raise NotImplementedError('Update not implemented.')


class JupyterHtmlRenderer(Renderer):
    def __init__(self, trace_handler: TraceHandler) -> None:
        self._update_controller = UpdateController(trace_handler)
        self._formatted = []

    def update(self, message: Message) -> None:
        render_update = self._update_controller.update(message)

        formatted = []
        for out_message in render_update.messages:
            if isinstance(out_message, ResetDisplayMessage):
                self._formatted.clear()
                formatted.clear()
            elif isinstance(out_message, TraceMessage):
                if isinstance(out_message.node_attr, TextOutput):
                    x = out_message.node_attr
                    fmt = f"<span style='background-color: rgba({165 * (1 - x.prob)}, {165 * x.prob}, 0, 0.15); border-radius: 3px;'>{html.escape(x.value)}</span>"
                    formatted.append(fmt)
                elif isinstance(out_message.node_attr, RoleOpenerInput):
                    # TODO(nopdive): Implement HTML for role opener
                    pass
                elif isinstance(out_message.node_attr, RoleCloserInput):
                    # TODO(nopdive): Implement HTML for role closer. Looks like there's a bug somewhere on order of trace nodes (text-output for closer is after role-closer-input).
                    pass
        self._formatted.extend(formatted)

        # For Jupyter HTML rendering, there are no partial updates.
        if len(self._formatted) > 0:
            clear_output(wait=True)
            display(HTML("".join(self._formatted)))


class AutoRenderer(Renderer):
    def __init__(self, trace_handler: TraceHandler):
        # TODO(nopdive): Add autodetect logic.
        self._renderer = JupyterHtmlRenderer(trace_handler=trace_handler)

    def update(self, message: Message) -> None:
        if self._renderer is not None:
            self._renderer.update(message)
