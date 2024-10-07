import logging
from typing import Optional, Callable
from pydantic import BaseModel
from asyncio import Queue

from ._message import JupyterCellExecutionCompletedMessage, deserialize_message, serialize_message
from ..trace import TraceHandler
from ..visual import GuidanceMessage, TraceMessage, ResetDisplayMessage, ClientReadyMessage
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


class JupyterChangeDetector:
    def __init__(self):
        self._prev_cell_msg_id: Optional[int] = None
        self._prev_cell_exec_count: Optional[int] = None

    def has_changed(self) -> bool:
        if ipython_imported and get_ipython() is not None:
            ipy = get_ipython()
            cell_msg_id = ipy.get_parent()["msg_id"]
            cell_exec_count = ipy.execution_count
            if (
                    cell_msg_id != self._prev_cell_msg_id
                    or cell_exec_count != self._prev_cell_exec_count
            ):
                self._prev_cell_msg_id = cell_msg_id
                self._prev_cell_exec_count = cell_exec_count
                logger.debug(f"JUPYTER:changed:{cell_msg_id}|{cell_exec_count}")
                return True
            else:
                return False
        else:
            return False


class UpdateController:
    def __init__(self, trace_handler: TraceHandler):
        self._trace_handler = trace_handler
        self._jupyter_change_detector = JupyterChangeDetector()
        self._messages: list[GuidanceMessage] = []
        self._prev_trace_id: Optional[int] = None

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
        # NOTE(nopdive): Consider refactoring out to Jupyter renderer.
        if self._jupyter_change_detector.has_changed():
            need_reset = True
            need_new_display = True
            self._messages.clear()

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
        self._observers.append(callback)

    def update(self, message: GuidanceMessage) -> None:
        raise NotImplementedError("Update not implemented.")


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


from ._async import run_async_task, ThreadSafeAsyncCondVar, async_loop


class JupyterWidgetRenderer(Renderer):
    def __init__(self, trace_handler: TraceHandler, wait_for_client=True) -> None:
        self._update_controller = UpdateController(trace_handler)
        self._jupyter_widget = None
        self._jupyter_change_detector = JupyterChangeDetector()
        self._is_alive_cell_cb = False
        self._last_trace_id = None
        self._loop = async_loop()

        self._send_queue: Optional[Queue] = None
        self._recv_queue: Optional[Queue] = None
        self._wait_for_client = wait_for_client
        self._client_ready = ThreadSafeAsyncCondVar(async_loop())
        self._cell_executed = ThreadSafeAsyncCondVar(async_loop())

        # Wait for queues to instantiate
        future = run_async_task(self._create_queues())
        future.result()

        # Start send/recv message loops
        run_async_task(self._handle_recv_messages())
        run_async_task(self._handle_send_messages())

        super().__init__()

    async def _create_queues(self) -> None:
        self._send_queue = Queue()
        self._recv_queue = Queue()
        return None

    def update(self, message: GuidanceMessage) -> None:
        # Handle Jupyter cell completion
        self._handle_jupyter_cell_completion()

        # Handle message
        display_update = self._update_controller.update(message)

        if display_update.need_new_display:
            logger.debug(f"NEED_NEW_DISPLAY:new widget")
            self._jupyter_widget = _create_stitch_widget()
            self._jupyter_widget.observe(self._client_msg_cb, names='clientmsg')

            clear_output(wait=True)
            display(self._jupyter_widget)

        for out_message in display_update.messages:
            if isinstance(out_message, TraceMessage):
                self._last_trace_id = out_message.trace_id
            self._loop.call_soon_threadsafe(self._send_queue.put_nowait, out_message)

    def _handle_jupyter_cell_completion(self):
        if self._jupyter_change_detector.has_changed():
            if self._is_alive_cell_cb:
                get_ipython().events.unregister('post_execute', self._cell_completion_cb)
                self._is_alive_cell_cb = False
            get_ipython().events.register('post_execute', self._cell_completion_cb)
            self._is_alive_cell_cb = True

    def _cell_completion_cb(self):
        try:
            message = JupyterCellExecutionCompletedMessage(
                last_trace_id=self._last_trace_id
            )
            logger.debug(f"CELL:executed:{message}")

            # Message Python observers
            self.notify(message)
            # Message client observers
            self._loop.call_soon_threadsafe(self._send_queue.put_nowait, message)
        finally:
            get_ipython().events.unregister('post_execute', self._cell_completion_cb)
            self._is_alive_cell_cb = False

    def _client_msg_cb(self, change: dict) -> None:
        # NOTE(nopdive): Widget callbacks do not print to stdout/stderr nor module log.
        self._loop.call_soon_threadsafe(self._recv_queue.put_nowait, change['new'])

    async def _handle_recv_messages(self):
        logger.debug("RECV:init")

        while True:
            try:
                value = await self._recv_queue.get()
                logger.debug(f"RECV:raw:{value}")
                message = deserialize_message(value)
                logger.debug(f"RECV:msg:{message}")
                if isinstance(message, ClientReadyMessage):
                    logger.debug("RECV:clientready")
                    self._client_ready.notify()
                self.notify(message)
                self._recv_queue.task_done()
            except Exception as e:
                logger.error(f"RECV:err:{repr(e)}")

    async def _handle_send_messages(self):
        logger.debug("SEND:init")

        # Wait until ready
        if self._wait_for_client:
            await self._client_ready.wait()
        logger.debug("SEND:ready")

        while True:
            try:
                message = await self._send_queue.get()
                logger.debug(f"SEND:msg:{message}")
                message_json = serialize_message(message)
                # logger.debug(f"SEND:json:{message_json}")
                if self._jupyter_widget is not None:
                    self._jupyter_widget.kernelmsg = message_json
                else:
                    logger.debug(f"SEND:jupyter:send but no widget")
                self._send_queue.task_done()
            except Exception as e:
                logger.error(f"SEND:err:{repr(e)}")


class LegacyHtmlRenderer(JupyterWidgetRenderer):
    """Original HTML renderer for guidance."""

    def __init__(self, trace_handler: TraceHandler) -> None:
        self._trace_handler = trace_handler
        super().__init__(trace_handler, wait_for_client=False)

    def update(self, message: GuidanceMessage) -> None:
        # Handle Jupyter cell completion
        self._handle_jupyter_cell_completion()

        if isinstance(message, TraceMessage):
            trace_node = self._trace_handler[message.trace_id]
            self._last_trace_id = message.trace_id
            if trace_node is not None:
                clear_output(wait=True)
                display(HTML(trace_node_to_html(trace_node, prettify_roles=False)))
        elif isinstance(message, JupyterCellExecutionCompletedMessage):
            logger.debug("renderer:cell executed")
        else:
            pass


class AutoRenderer(Renderer):
    def __init__(self, trace_handler: TraceHandler):
        if stitch_installed:
            self._renderer = JupyterWidgetRenderer(trace_handler=trace_handler)
        else:
            self._renderer = LegacyHtmlRenderer(trace_handler=trace_handler)

        # self._renderer = LegacyHtmlRenderer(trace_handler=trace_handler)
        super().__init__()

    def notify(self, message: GuidanceMessage):
        self._renderer.notify(message)

    def subscribe(self, callback: Callable[[GuidanceMessage], None]) -> None:
        self._renderer.subscribe(callback)

    def update(self, message: GuidanceMessage) -> None:
        self._renderer.update(message)
