""" Renderer is responsible for displaying output.

Our main focus is on jupyter notebooks and later terminal.
"""
# TODO(nopdive): Implementation for terminals & append-only text displays.
# NOTE(nopdive): Testing this notebook related components is tricky. Should figure this out at some point.

import asyncio
import logging
import weakref
from typing import Optional, Callable
from pydantic import BaseModel
from asyncio import Queue

from ._environment import Environment
from ._message import ExecutionCompletedMessage, ExecutionCompletedOutputMessage, \
    deserialize_message, serialize_message, MetricMessage, ClientReadyAckMessage
from ..trace import TraceHandler, TextOutput
from ..visual import GuidanceMessage, TraceMessage, ResetDisplayMessage, ClientReadyMessage
from ._trace import trace_node_to_html
from ._async import run_async_coroutine, async_loop, async_task
from warnings import warn

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


# NOTE(nopdive): Temporary solution until we iron out all issues with widgets.
_legacy_mode = False  # pragma: no cover
def legacy_mode(flag: Optional[bool] = None) -> bool:  # pragma: no cover
    """ Sets visualizations to legacy mode (how Guidance used to be).

    This method call will likely deprecate when a fresher API for
    renderers is defined.

    Args:
        flag: True if legacy is wanted.

    Returns:
        Whether legacy mode is enabled.
    """
    global _legacy_mode
    if flag is not None:
        _legacy_mode = flag
    return _legacy_mode


class RenderUpdate(BaseModel):
    """ This informs which new messages are to render, along with if a new display is needed."""

    messages: list = []
    need_new_display: bool = False


class JupyterChangeDetector:
    """ Detects if a change has occurred on a Jupyter cell."""

    def __init__(self):
        self._prev_cell_msg_id: Optional[int] = None
        self._prev_cell_exec_count: Optional[int] = None

    def has_changed(self) -> bool:
        """ Detects if Jupyter cell has been updated.

        Returns:
            True if cell has a higher exec count or if we're in a different cell to before.
        """
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
    """ Responsible for producing a RenderUpdate object, that has messages to be sent to client."""

    def __init__(self, trace_handler: TraceHandler):
        """ Initializes.

        Args:
            trace_handler: Trace handler of an engine.
        """
        self._trace_handler = trace_handler
        self._jupyter_change_detector = JupyterChangeDetector()
        self._messages: list[GuidanceMessage] = []
        self._prev_trace_id: Optional[int] = None
        self._completed = False

    def update(self, message: GuidanceMessage) -> RenderUpdate:
        """ Determines which messages need to be sent to a client.

        This can include resets, as well as full message replays when the jupyter
        cell changes, or a new model is displaying.

        Args:
            message: Incoming message.

        Returns:
            Render update that has messages to be sent to client alongside a reset flag.
        """
        if isinstance(message, MetricMessage) and not self._completed:
            return RenderUpdate(messages=[message])
        elif isinstance(message, ExecutionCompletedOutputMessage) and not self._completed:
            logger.debug("RENDER:execution completed out")
            return RenderUpdate(messages=[message])
        elif isinstance(message, ExecutionCompletedMessage):
            logger.debug("RENDER:execution completed")
            self._completed = True
            return RenderUpdate()
        elif not isinstance(message, TraceMessage):
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
        """Initializes. """
        self._observers = []

    def notify(self, message: GuidanceMessage):
        """ Notifies all observers of the renderer of an incoming message.

        Args:
            message: Incoming message.
        """
        for observer in self._observers:
            observer(message)

    def subscribe(self, callback: Callable[[GuidanceMessage], None]) -> None:
        """ Subscribes to incoming messages.

        Args:
            callback: Callback to handle incoming messages.
        """
        self._observers.append(callback)

    def unsubscribe(self, callback: Callable[[GuidanceMessage], None]) -> None:
        self._observers.remove(callback)

    def cleanup(self):
        pass

    def update(self, message: GuidanceMessage) -> None:
        """ Updates renderer with incoming message.

        Args:
            message: Incoming message.
        """
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


class JupyterWidgetRenderer(Renderer):
    """Jupyter widget renderer that is implemented via stitch package."""

    def __init__(self, trace_handler: TraceHandler) -> None:
        """ Initializes.

        Args:
            trace_handler: Trace handler of an engine.
        """
        self._update_controller = UpdateController(trace_handler)
        self._jupyter_widget = None
        self._jupyter_change_detector = JupyterChangeDetector()
        self._is_alive_cell_cb = False
        self._last_trace_id = None
        self._loop = async_loop()

        # Create queue and wait for instantiation
        self._send_queue: Optional[Queue] = None
        self._recv_queue: Optional[Queue] = None
        future = run_async_coroutine(self._create_queues())
        future.result()

        # Start send/recv message loops
        recv_coroutine = self._handle_recv_messages()
        send_coroutine = self._handle_send_messages()
        self._recv_task = run_async_coroutine(async_task(recv_coroutine)).result()
        self._send_task = run_async_coroutine(async_task(send_coroutine)).result()

        weakref.finalize(self, self.cleanup)
        super().__init__()

    def cleanup(self):
        # NOTE(nopdive): This might not be called on GC due potential circular refs.
        # Follow-up work is dependent on overall memory review for engine.

        logger.debug("RENDERER:cleanup")
        self._loop.call_soon_threadsafe(self._send_queue.put_nowait, None)
        self._loop.call_soon_threadsafe(self._recv_queue.put_nowait, None)

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
            message = ExecutionCompletedMessage(
                last_trace_id=self._last_trace_id
            )
            logger.debug(f"CELL:executed:{message}")

            # Message Python observers
            self.notify(message)
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
                # logger.debug(f"RECV:raw:{value}")

                if value is None:
                    logger.debug("RECV:closing")
                    break

                message = deserialize_message(value)
                # logger.debug(f"RECV:msg:{message}")

                if isinstance(message, ClientReadyMessage):
                    logger.debug("RECV:clientready")
                    self._loop.call_soon_threadsafe(self._send_queue.put_nowait, ClientReadyAckMessage())

                self.notify(message)
                self._recv_queue.task_done()
            except Exception as e:
                logger.error(f"RECV:err:{repr(e)}")

    async def _handle_send_messages(self):
        logger.debug("SEND:init")

        # TODO(nopdive): Find better alternative at some point for client readiness.
        # NOTE(nopdive): Waiting on client cb does not work, client messages received on cell completion.

        # What if we only used 1% of our brain?
        await asyncio.sleep(200 / 1000.)
        logger.debug("SEND:ready")

        while True:
            try:
                message = await self._send_queue.get()
                # logger.debug(f"SEND:msg:{message}")

                if message is None:
                    logger.debug("SEND:closing")
                    break

                message_json = serialize_message(message)
                # logger.debug(f"SEND:json:{message_json}")

                if self._jupyter_widget is not None:
                    self._jupyter_widget.kernelmsg = message_json
                else:
                    logger.debug(f"SEND:jupyter:send but no widget")
                self._send_queue.task_done()
            except Exception as e:
                logger.error(f"SEND:err:{repr(e)}")


class DoNothingRenderer(Renderer):
    """ It does nothing. Placeholder for future renderers."""

    def __init__(self, trace_handler: TraceHandler) -> None:
        """ Initializes.

        Args:
            trace_handler: Trace handler of an engine.
        """
        self._trace_handler = trace_handler
        super().__init__()

    def update(self, message: GuidanceMessage) -> None:
        pass


class LegacyHtmlRenderer(JupyterWidgetRenderer):
    """Original HTML renderer for guidance."""

    def __init__(self, trace_handler: TraceHandler) -> None:
        """ Initializes.

        Args:
            trace_handler: Trace handler of an engine.
        """

        self._trace_handler = trace_handler
        super().__init__(trace_handler)

    def update(self, message: GuidanceMessage) -> None:
        # Handle Jupyter cell completion
        self._handle_jupyter_cell_completion()

        if isinstance(message, TraceMessage) or isinstance(message, ExecutionCompletedOutputMessage):
            complete_msg = None
            if isinstance(message, ExecutionCompletedOutputMessage):
                complete_msg = message

            trace_node = self._trace_handler[message.trace_id]
            self._last_trace_id = message.trace_id
            if trace_node is not None:
                clear_output(wait=True)
                display(HTML(trace_node_to_html(trace_node, prettify_roles=False, complete_msg=complete_msg)))
        elif isinstance(message, ExecutionCompletedMessage):
            logger.debug("RENDERER:cell executed")
        else:
            pass


class AutoRenderer(Renderer):
    """ Automatically detects which renderer to use based on environment."""

    def __init__(self, trace_handler: TraceHandler):
        """ Initializes.

        Args:
            trace_handler: Trace handler of an engine.
        """
        self._env = Environment()

        if legacy_mode():
            self._renderer = LegacyHtmlRenderer(trace_handler=trace_handler)
        else:
            if self._env.is_notebook():
                if stitch_installed:
                    self._renderer = JupyterWidgetRenderer(trace_handler=trace_handler)
                else:
                    self._renderer = LegacyHtmlRenderer(trace_handler=trace_handler)
            elif self._env.is_terminal():
                # TODO(nopdive): When IPython events are figured out (cell completion)
                #                hook up terminal interface separate to non-interactive
                #                shell.
                self._renderer = DoNothingRenderer(trace_handler=trace_handler)
            else:  # pragma: no cover
                logger.error("Env detection has failed. This is a bug.")
                warn("Env detection has failed. No renderer will be provided.")
                self._renderer = None

        super().__init__()

    def notify(self, message: GuidanceMessage):
        self._renderer.notify(message)

    def subscribe(self, callback: Callable[[GuidanceMessage], None]) -> None:
        self._renderer.subscribe(callback)

    def update(self, message: GuidanceMessage) -> None:
        self._renderer.update(message)

    def cleanup(self):
        self._renderer.cleanup()

    def renderer_type(self) -> type:
        """Type of renderer that has been selected based on environment.

        Returns:
            Type of selected renderer.
        """
        return type(self._renderer)