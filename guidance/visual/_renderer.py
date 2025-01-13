""" Renderer is responsible for displaying output.

Our main focus is on jupyter notebooks and later terminal.
"""
# TODO(nopdive): Implementation for terminals & append-only text displays.
# NOTE(nopdive): Testing this notebook related components is tricky. Should figure this out at some point.

import asyncio
import logging
import weakref
from typing import Optional, Callable, Tuple, Any
from asyncio import Queue
from functools import partial

from . import MetricMessage, TokensMessage
from ._environment import Environment
from ._jupyter import ipy_handle_event_once
from ._message import ExecutionCompletedMessage, \
    deserialize_message, serialize_message, ClientReadyAckMessage, ExecutionStartedMessage
from .._utils import log_cleanup
from ..trace import TraceHandler
from ..visual import GuidanceMessage, TraceMessage, ResetDisplayMessage, ClientReadyMessage
from ._trace import trace_node_to_html
from ._async import run_async_coroutine, async_task, call_soon_threadsafe
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
        """ Unsubscribes from incoming messages.

        Args:
            callback: Callback to remove.
        """
        self._observers.remove(callback)

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
    weakref.finalize(w, log_cleanup, f"stitch({id(w)})")

    return w


_create_stitch_widget.src_doc_template = None


def _cleanup(recv_queue: Optional[Queue], send_queue: Optional[Queue], log_msg: str) -> None:
    log_cleanup(log_msg)
    if recv_queue is not None:
        call_soon_threadsafe(send_queue.put_nowait, None)
    if send_queue is not None:
        call_soon_threadsafe(recv_queue.put_nowait, None)


async def _create_queue() -> Queue:
    # This will run in the visual thread.
    return Queue()


def _on_stitch_clientmsg(recv_queue_weakref: weakref.ReferenceType["Queue"], change: dict) -> None:
    # NOTE(nopdive): Widget callbacks do not print to stdout/stderr nor module log.
    recv_queue = recv_queue_weakref()
    if recv_queue is not None:
        call_soon_threadsafe(recv_queue.put_nowait, change['new'])


def _on_cell_completion(renderer_weakref: weakref.ReferenceType["Renderer"], info) -> None:
    logger.debug(f"CELL:executed")
    try:
        renderer = renderer_weakref()
        if renderer is None:
            return

        message = ExecutionCompletedMessage(
            last_trace_id=renderer._last_trace_id,
            is_err=info.error_in_exec is not None,
        )
        renderer.update(message)
    except Exception as e:
        logger.error(repr(e))


async def _handle_recv_messages(renderer_weakref: weakref.ReferenceType["Renderer"], queue_weakref: weakref.ReferenceType["Queue"]) -> None:
    logger.debug("RECV:init")

    while True:
        try:
            queue = queue_weakref()
            if queue is None:
                break
            value = await queue.get()

            # logger.debug(f"RECV:raw:{value}")

            if value is None:
                logger.debug("RECV:closing")
                break

            message = deserialize_message(value)
            # logger.debug(f"RECV:msg:{message}")

            renderer = renderer_weakref()
            if renderer is None:
                logger.debug("RECV:renderer early clean")
                break

            if isinstance(message, ClientReadyMessage):
                logger.debug("RECV:clientready")
                call_soon_threadsafe(renderer._send_queue.put_nowait, ClientReadyAckMessage())

            renderer.notify(message)
            renderer._recv_queue.task_done()
        except Exception as e:
            logger.error(f"RECV:err:{repr(e)}")


async def _handle_send_messages(renderer_weakref: weakref.ReferenceType["Renderer"], queue_weakref: weakref.ReferenceType["Queue"]) -> None:
    logger.debug("SEND:init")
    # NOTE(nopdive): Waiting on client cb does not work, client messages received on cell completion.
    #                Currently, we do a replay of messages on completion for client if client
    #                first receives non-zero message identifier.

    # What if we only used 1% of our brain?
    await asyncio.sleep(200 / 1000.)
    logger.debug("SEND:ready")

    while True:
        try:
            queue = queue_weakref()
            if queue is None:
                break
            message = await queue.get()
            # logger.debug(f"SEND:msg:{message}")

            if message is None:
                logger.debug("SEND:closing")
                break

            message_json = serialize_message(message)
            # logger.debug(f"SEND:json:{message_json}")

            renderer = renderer_weakref()
            if renderer is None:
                break
            if renderer._stitch_widget is not None:
                renderer._stitch_widget.kernelmsg = message_json
            else:
                logger.debug(f"SEND:jupyter:send but no widget")
            renderer._send_queue.task_done()
        except Exception as e:
            logger.error(f"SEND:err:{repr(e)}")


class JupyterWidgetRenderer(Renderer):
    """Jupyter widget renderer that is implemented via stitch package."""

    def __init__(self, trace_handler: TraceHandler) -> None:
        """ Initializes.

        Args:
            trace_handler: Trace handler of an engine.
        """
        super().__init__()

        self._trace_handler = trace_handler
        self._messages: list[GuidanceMessage] = []
        self._completed = False
        self._running = False
        self._need_reset = False
        self._stitch_widget = None
        self._stitch_on_clientmsg = None
        self._cell_event_handler = None
        self._last_trace_id = None

        # Create queue and wait for instantiation
        self._send_queue: Queue = run_async_coroutine(_create_queue()).result()
        self._recv_queue: Queue = run_async_coroutine(_create_queue()).result()

        # Start send/recv message loops
        recv_coroutine = _handle_recv_messages(weakref.ref(self), weakref.ref(self._recv_queue))
        send_coroutine = _handle_send_messages(weakref.ref(self), weakref.ref(self._send_queue))
        self._recv_task = run_async_coroutine(async_task(recv_coroutine)).result()
        self._send_task = run_async_coroutine(async_task(send_coroutine)).result()

        weakref.finalize(self, _cleanup, self._recv_queue, self._send_queue, f"renderer({id(self)})")


    def has_divergence(self, message: GuidanceMessage) -> Tuple[bool, int]:
        """Checks if message has divergence with current path.

        Args:
            message: Incoming message.

        Returns:
            Tuple of (has diverged, shared ancestor index). Index will be -1 if no divergence.

        Raises:
            Exception if there is no shared ancestor (including root). This should not happen.
        """
        if not isinstance(message, TraceMessage):
            return False, -1

        # If we diverge from the model path, truncate and reset
        message_trace_node = self._trace_handler[message.trace_id]

        prev_trace_messages = [x for x in self._messages if isinstance(x, TraceMessage)]
        trace_messages_len = len(prev_trace_messages)
        if trace_messages_len == 0:
            return False, -1
        elif trace_messages_len == 1:
            if isinstance(self._messages[0], TraceMessage):
                last_trace_node = self._trace_handler[self._messages[0].trace_id]
                if message_trace_node.parent == last_trace_node:
                    return False, -1
                else:
                    return True, 0
            else:
                return False, -1
        else:
            last_trace_message = prev_trace_messages[-1]
            last_trace_node = self._trace_handler[last_trace_message.trace_id]

            if last_trace_node not in message_trace_node.path():
                logger.debug(f"DIVERGENCE:curr:{message_trace_node}")
                logger.debug(f"DIVERGENCE:prev:{last_trace_node}")

                # Truncate path that is no longer used by current trace node
                ancestor_idx = -1
                ancestors = set(message_trace_node.ancestors())
                for idx, prev_message in enumerate(self._messages):
                    if isinstance(prev_message, TraceMessage):
                        prev_trace_node = self._trace_handler[prev_message.trace_id]
                        if prev_trace_node in ancestors:
                            ancestor_idx = idx

                if ancestor_idx == -1:
                    if message_trace_node.parent == last_trace_node.root():  # pragma: no cover
                        ancestor_idx = 0
                    else:
                        raise Exception(f"Parent missing for trace node: {message_trace_node}")

                return True, ancestor_idx
            else:
                return False, -1


    def update(self, message: GuidanceMessage) -> None:
        out_messages = []

        if isinstance(message, ExecutionCompletedMessage):
            logger.debug("RENDERER:execution end")
            self._completed = True
            self._running = False
            self.notify(message)

            if message.is_err:
                out_messages.append(MetricMessage(name="status", value="⚠"))
            else:
                out_messages.append(MetricMessage(name="status", value="✓"))

        if not self._running and isinstance(message, TraceMessage):
            logger.debug("RENDERER:execution start")
            started_msg = ExecutionStartedMessage()
            out_messages.append(started_msg)
            out_messages.append(MetricMessage(name="status", value='⟳'))

            ipy_handle_event_once(
                partial(_on_cell_completion, weakref.ref(self)),
                'post_run_cell'
            )
            self._need_reset = True
            self._running = True
            self._completed = False

            # TODO(nopdive): Fire off execution immediately to renderer subscribers. Review later.
            call_soon_threadsafe(self._recv_queue.put_nowait, serialize_message(started_msg))

        # Check if message has diverged from prev messages
        diverged, shared_ancestor_idx = self.has_divergence(message)
        if diverged:
            out_messages.append(ResetDisplayMessage())
            out_messages[len(out_messages):] = self._messages[:shared_ancestor_idx]
            self._messages.clear()

        # Reset if needed
        if self._need_reset:
            logger.debug("RENDERER:reset")

            # Clear messages
            self._messages = []

            if self._stitch_widget is not None:
                self._stitch_widget.unobserve(self._stitch_on_clientmsg, names='clientmsg')

            self._stitch_widget = _create_stitch_widget()
            self._stitch_on_clientmsg = partial(_on_stitch_clientmsg, weakref.ref(self._recv_queue))
            self._stitch_widget.observe(self._stitch_on_clientmsg, names='clientmsg')

            # Redraw
            # clear_output(wait=True)
            display(self._stitch_widget)

            self._need_reset = False

        # Append current message to outgoing
        out_messages.append(message)

        # Send outgoing messages to client
        for out_message in out_messages:
            # logger.debug(f"RENDERER:out:{out_message}")
            if isinstance(out_message, TraceMessage):
                self._last_trace_id = out_message.trace_id

            self._messages.append(out_message)
            call_soon_threadsafe(self._send_queue.put_nowait, out_message)

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
        pass
        # # Handle Jupyter cell completion
        # self._handle_jupyter_cell_completion()
        #
        # if isinstance(message, TraceMessage) or isinstance(message, ExecutionCompletedOutputMessage):
        #     complete_msg = None
        #     if isinstance(message, ExecutionCompletedOutputMessage):
        #         complete_msg = message
        #
        #     trace_node = self._trace_handler[message.trace_id]
        #     self._last_trace_id = message.trace_id
        #     if trace_node is not None:
        #         clear_output(wait=True)
        #         display(HTML(trace_node_to_html(trace_node, prettify_roles=False, complete_msg=complete_msg)))
        # elif isinstance(message, ExecutionCompletedMessage):
        #     logger.debug("RENDERER:cell executed")
        # else:
        #     pass


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
                self._renderer = DoNothingRenderer(trace_handler=trace_handler)

        super().__init__()

    def notify(self, message: GuidanceMessage):
        self._renderer.notify(message)

    def subscribe(self, callback: Callable[[GuidanceMessage], None]) -> None:
        self._renderer.subscribe(callback)

    def unsubscribe(self, callback: Callable[[GuidanceMessage], None]) -> None:
        self._renderer.unsubscribe(callback)

    def update(self, message: GuidanceMessage) -> None:
        self._renderer.update(message)

    def renderer_type(self) -> type:
        """Type of renderer that has been selected based on environment.

        Returns:
            Type of selected renderer.
        """
        return type(self._renderer)