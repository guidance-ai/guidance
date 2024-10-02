import logging
import weakref
from typing import Optional, Callable
from pydantic import BaseModel
from pydantic_core import from_json
import json

from ..trace import TraceHandler
from ..visual import GuidanceMessage, TraceMessage, ResetDisplayMessage
from ._trace import trace_node_to_html

try:
    from IPython.display import clear_output, display, HTML
    from IPython import get_ipython

    ipython_imported = True
except ImportError:
    ipython_imported = False

try:
    import stitch

    stitch_installed = True
except ImportError:
    stitch_installed = False

logger = logging.getLogger(__name__)


class RenderUpdate(BaseModel):
    messages: list = []
    need_new_display: bool = False


class UpdateController:
    def __init__(self, trace_handler: TraceHandler):
        self._trace_handler = trace_handler
        self._messages: list[GuidanceMessage] = []

        self._prev_trace_id: Optional[int] = None
        self._prev_cell_msg_id: Optional[int] = None
        self._prev_cell_exec_count: Optional[int] = None

    def update(self, message: GuidanceMessage) -> RenderUpdate:
        if not isinstance(message, TraceMessage):
            return RenderUpdate()

        trace_node = self._trace_handler[message.trace_id]
        need_reset = False
        need_new_display = False

        if len(self._messages) == 0:
            # If no messages, reset
            logger.debug("NEED_RESET:empty")
            need_reset = True
            need_new_display = True
        else:
            # If we diverge from the model path, truncate and reset
            *_, last_trace_message = (x for x in reversed(self._messages) if isinstance(x, TraceMessage))
            last_trace_node = self._trace_handler[last_trace_message.trace_id]

            if last_trace_node not in trace_node.path():
                logger.debug(f"NEED_RESET:divergence:curr:{trace_node}")
                logger.debug(f"NEED_RESET:divergence:prev:{last_trace_node}")
                need_reset = True

                # Truncate path that is no longer used by current trace node
                ancestor_idx = -1
                ancestors = set(trace_node.ancestors())
                for idx, prev_message in enumerate(self._messages):
                    if isinstance(prev_message, TraceMessage):
                        if self._trace_handler[prev_message.trace_id] in ancestors:
                            ancestor_idx = idx
                if ancestor_idx == -1:
                    logger.debug(f"PARENT_NOT_FOUND:{trace_node}")
                    self._messages.clear()
                else:
                    self._messages = self._messages[:ancestor_idx]

        # If we are in a new Jupyter cell or execution, reset
        if ipython_imported and get_ipython() is not None:
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

    def __init__(self):
        self._observers = []

    def notify(self, message: GuidanceMessage):
        for observer in self._observers:
            observer(message)

    def subscribe(self, callback: Callable[[GuidanceMessage], None]) -> None:
        self._observers.append(weakref.proxy(callback, self._observers.remove))

    def update(self, message: GuidanceMessage) -> None:
        raise NotImplementedError("Update not implemented.")


class LegacyHtmlRenderer(Renderer):
    """Original HTML renderer for guidance."""

    def __init__(self, trace_handler: TraceHandler) -> None:
        self._trace_handler = trace_handler
        super().__init__()

    def update(self, message: GuidanceMessage) -> None:
        if not isinstance(message, TraceMessage):
            return

        trace_node = self._trace_handler[message.trace_id]
        if trace_node is not None:
            clear_output(wait=True)
            display(HTML(trace_node_to_html(trace_node, prettify_roles=False)))


def _create_stitch_widget():
    from stitch import StitchWidget
    import pkg_resources

    if _create_stitch_widget.src_doc_template is None:
        with open(
            pkg_resources.resource_filename("guidance", "resources/graphpaper-inline.html"), "r"
        ) as f:
            _create_stitch_widget.src_doc_template = f.read()
    w = StitchWidget()
    w.initial_width = "100%"
    w.initial_height = "auto"
    w.srcdoc = _create_stitch_widget.src_doc_template

    return w


_create_stitch_widget.src_doc_template = None


class JupyterWidgetRenderer(Renderer):
    def __init__(self, trace_handler: TraceHandler) -> None:
        self._jupyter_widget = None
        self._update_controller = UpdateController(trace_handler)
        self._message_buffer = []
        self._client_ready = False

        super().__init__()

    def update(self, message: GuidanceMessage) -> None:
        display_update = self._update_controller.update(message)

        if display_update.need_new_display:
            logger.debug(f"NEED_NEW_DISPLAY:new widget")
            self._jupyter_widget = _create_stitch_widget()
            self._jupyter_widget.observe(self._client_msg_cb, names='clientmsg')

            # clear_output(wait=True)
            display(self._jupyter_widget)

        for out_message in display_update.messages:
            message_json = out_message.model_dump_json(indent=2, serialize_as_any=True)
            self._message_buffer.append(message_json)
        self._send_messages_to_client()

    def _client_msg_cb(self, change: dict) -> None:
        # NOTE(nopdive): Widget callbacks do not print to stdout/stderr nor module log.

        new_val = change['new']
        try:
            msg_di = json.loads(new_val)
            if msg_di.get('class_name', None) == 'HeartbeatMessage':
                logger.debug("CLIENT_READY")
                self._client_ready = True
                self._send_messages_to_client()
            msg = from_json(new_val)
            self.notify(msg)
        except Exception as e:
            self._jupyter_widget.log.error(f"Failed to process client message:{new_val}:{repr(e)}")

    def _send_messages_to_client(self) -> None:
        if not self._client_ready:
            return

        while len(self._message_buffer) != 0:
            message = self._message_buffer.pop(0)
            self._jupyter_widget.kernelmsg = message


class AutoRenderer(Renderer):
    def __init__(self, trace_handler: TraceHandler):
        if stitch_installed:
            self._renderer = JupyterWidgetRenderer(trace_handler=trace_handler)
        else:
            self._renderer = LegacyHtmlRenderer(trace_handler=trace_handler)
        super().__init__()

    def notify(self, message: GuidanceMessage):
        self._renderer.notify(message)

    def subscribe(self, callback: Callable[[GuidanceMessage], None]) -> None:
        self._renderer.subscribe(callback)

    def update(self, message: GuidanceMessage) -> None:
        self._renderer.update(message)
