"""Renderer is responsible for displaying output.

Our main focus is on jupyter notebooks and later terminal.
"""
# TODO(nopdive): Implementation for terminals & append-only text displays.
# NOTE(nopdive): Testing this notebook related components is tricky. Should figure this out at some point.

import asyncio
import logging
import threading
import traceback
import weakref
from asyncio import Queue
from functools import lru_cache, partial
from typing import TYPE_CHECKING, Callable, Optional
from warnings import warn

from .._topics import DEFAULT_TOPIC
from .._utils import log_cleanup
from ..trace import TraceHandler
from ..visual import (
    ClientReadyMessage,
    ExecutionCompletedAckMessage,
    GuidanceMessage,
    ResetDisplayMessage,
    TraceMessage,
)
from . import MetricMessage
from ._environment import Environment
from ._jupyter import ipy_handle_event_once
from ._message import (
    ClientReadyAckMessage,
    ExecutionCompletedMessage,
    ExecutionStartedMessage,
    OutputRequestMessage,
    deserialize_message,
    serialize_message,
)

try:
    from IPython import get_ipython
    from IPython.display import HTML, clear_output, display

    ipython_imported = True
except ImportError:
    ipython_imported = False


try:
    import stitch  # type: ignore[import-untyped]

    stitch_installed = True
except ImportError:
    stitch_installed = False

if TYPE_CHECKING:
    from stitch import StitchWidget

logger = logging.getLogger(__name__)

# Uncomment the following lines to enable file logging
import datetime

log_filename = f"widget_debug_{datetime.datetime.now().strftime('%H%M%S')}.log"
file_handler = logging.FileHandler(log_filename)
file_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s.%(msecs)03d - %(name)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.setLevel(logging.DEBUG)

DEFAULT_TOPIC = "default"


class Renderer:
    """Renders guidance model to a visual medium."""

    def __init__(self):
        """Initializes."""

    def update(self, message: GuidanceMessage, topic: str = DEFAULT_TOPIC) -> None:
        """Updates renderer with incoming message.

        Args:
            message: Incoming message.
            topic: Topic to update.
        """
        raise NotImplementedError("Update not implemented.")


@lru_cache(maxsize=1)
def _get_src_doc_template() -> str:
    """Returns the source document template for the stitch widget."""
    import importlib.resources as resources

    import guidance

    path = resources.files(guidance) / "resources" / "graphpaper-inline.html"
    with path.open("r", encoding="utf-8") as f:
        return f.read()


def _create_stitch_widget() -> "StitchWidget":
    from stitch import StitchWidget

    w = StitchWidget()
    w.initial_width = "100%"
    w.initial_height = "auto"
    w.srcdoc = _get_src_doc_template()
    weakref.finalize(w, log_cleanup, f"stitch({id(w)})")

    return w


def _cleanup(recv_queue: Optional[Queue], send_queue: Optional[Queue], log_msg: str, exchange_cb) -> None:
    from ..registry import get_bg_async, get_exchange

    log_cleanup(log_msg)
    if send_queue is not None:
        get_bg_async().call_soon_threadsafe(send_queue.put_nowait, None)
    if recv_queue is not None:
        get_bg_async().call_soon_threadsafe(recv_queue.put_nowait, None)

    get_exchange().unsubscribe(exchange_cb)


async def _create_queue() -> Queue:
    # This will run in the visual thread.
    return Queue()


def _on_stitch_clientmsg(recv_queue_weakref: weakref.ReferenceType["Queue"], change: dict) -> None:
    from ..registry import get_bg_async

    # NOTE(nopdive): Widget callbacks do not print to stdout/stderr nor module log.
    recv_queue = recv_queue_weakref()
    if recv_queue is not None:
        get_bg_async().call_soon_threadsafe(recv_queue.put_nowait, change["new"])


def _on_cell_completion(renderer_weakref: weakref.ReferenceType["JupyterWidgetRenderer"], info) -> None:
    logger.debug(f"CELL:executed")
    try:
        renderer = renderer_weakref()
        if renderer is None:
            return

        message = ExecutionCompletedMessage(
            last_trace_id=renderer.last_trace_id,
            is_err=info.error_in_exec is not None,
        )
        renderer.update(message)
    except Exception as _:
        logger.error(f"CELL_COMPLETE:{traceback.format_exc()}")


async def _handle_recv_messages(
    renderer_weakref: weakref.ReferenceType["JupyterWidgetRenderer"], queue_weakref: weakref.ReferenceType["Queue"]
) -> None:
    logger.debug("RECV:init")
    from ..registry import get_bg_async, get_exchange

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
                get_bg_async().call_soon_threadsafe(renderer.send_queue.put_nowait, ClientReadyAckMessage())
            elif isinstance(message, OutputRequestMessage):
                logger.debug("RECV:outputrequest")
            elif isinstance(message, ExecutionCompletedAckMessage):
                logger.debug(f"RECV:ExecutionCompletedAck (id={message.message_id})")
                if renderer._ack_received_event is not None:
                    logger.debug(f"RECV:setting ack received event, event={id(renderer._ack_received_event)}")
                    renderer._ack_received_event.set()
                else:
                    logger.debug(f"RECV:ack received but no event to set")

            get_exchange().publish(message, topic=f"{DEFAULT_TOPIC}")
            renderer.recv_queue.task_done()
        except Exception as _:
            logger.error(f"RECV:err:{traceback.format_exc()}")


async def _handle_send_messages(
    renderer_weakref: weakref.ReferenceType["JupyterWidgetRenderer"], queue_weakref: weakref.ReferenceType["Queue"]
) -> None:
    logger.debug("SEND:init")
    # NOTE(nopdive): Waiting on client cb does not work, client messages received on cell completion.
    #                Currently, we do a replay of messages on completion for client if client
    #                first receives non-zero message identifier.

    # What if we only used 1% of our brain?
    await asyncio.sleep(200 / 1000.0)
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
            if isinstance(renderer, JupyterWidgetRenderer) and renderer.stitch_widget is not None:
                # NOTE(nopdive): This at random times, appears to fire two changes instead of one change event.
                renderer.stitch_widget.kernelmsg = message_json
            else:
                logger.debug(f"SEND:jupyter:send but no widget")
            renderer.send_queue.task_done()
        except Exception as _:
            logger.error(f"SEND:err:{traceback.format_exc()}")


class JupyterWidgetRenderer(Renderer):
    """Jupyter widget renderer that is implemented via stitch package."""

    def __init__(self, trace_handler: TraceHandler) -> None:
        """Initializes.

        Args:
            trace_handler: Trace handler of an engine.
        """
        from ..registry import get_bg_async, get_exchange

        super().__init__()

        self.stitch_widget: Optional[StitchWidget] = None
        self.last_trace_id: Optional[int] = None

        self._trace_handler = trace_handler
        self._messages: list[GuidanceMessage] = []
        self._completed = False
        self._running = False
        self._new_widget_needed = False
        self.stitch_widget_observed = False
        self._stitch_on_clientmsg: Optional[Callable[[], None]] = None
        self.last_cell_session_id: Optional[str] = None

        # Debug tracking
        self._debug_enabled = False
        self._debug_messages: list[GuidanceMessage] = []

        # For waiting on execution completion acknowledgment
        self._ack_received_event = None

        # Create queue and wait for instantiation
        self.send_queue: Queue = get_bg_async().run_async_coroutine(_create_queue()).result()
        self.recv_queue: Queue = get_bg_async().run_async_coroutine(_create_queue()).result()

        # Start send/recv message loops
        recv_coroutine = _handle_recv_messages(weakref.ref(self), weakref.ref(self.recv_queue))
        send_coroutine = _handle_send_messages(weakref.ref(self), weakref.ref(self.send_queue))
        self._recv_task = get_bg_async().run_async_coroutine(get_bg_async().async_task(recv_coroutine)).result()
        self._send_task = get_bg_async().run_async_coroutine(get_bg_async().async_task(send_coroutine)).result()

        # Start recv on exchange
        get_exchange().subscribe(self._on_exchange)
        weakref.finalize(self, _cleanup, self.recv_queue, self.send_queue, f"renderer({id(self)})", self._on_exchange)

    def _on_exchange(self, message: GuidanceMessage) -> None:
        if not isinstance(message, MetricMessage):  # NOTE(nopdive): Metrics spam at fixed intervals.
            logger.debug(f"ON_EXCHANGE:{message}")

        if isinstance(message, MetricMessage):
            self.update(message)
        elif isinstance(message, OutputRequestMessage):
            self.update(message)
        elif isinstance(message, TraceMessage):
            self.update(message)

    def has_divergence(self, message: GuidanceMessage) -> tuple[bool, int]:
        """Checks if message has divergence with current path.

        Args:
            message: Incoming message.

        Returns:
            tuple of (has diverged, shared ancestor index). Index will be -1 if no divergence.

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
                try:
                    last_trace_node = self._trace_handler[self._messages[0].trace_id]
                    if message_trace_node.parent == last_trace_node:
                        return False, -1
                    else:
                        return True, 0
                except KeyError:
                    # Trace node was garbage collected, treat as divergence
                    return True, 0
            else:
                return False, -1
        else:
            last_trace_message = prev_trace_messages[-1]
            try:
                last_trace_node = self._trace_handler[last_trace_message.trace_id]
            except KeyError:
                # Trace node was garbage collected, treat as divergence
                return True, 0

            if last_trace_node not in message_trace_node.path():
                logger.debug(f"DIVERGENCE:curr:{message_trace_node}")
                logger.debug(f"DIVERGENCE:prev:{last_trace_node}")

                # Truncate path that is no longer used by current trace node
                ancestor_idx = -1
                ancestors = set(message_trace_node.ancestors())
                for idx, prev_message in enumerate(self._messages):
                    if isinstance(prev_message, TraceMessage):
                        try:
                            prev_trace_node = self._trace_handler[prev_message.trace_id]
                            if prev_trace_node in ancestors:
                                ancestor_idx = idx
                        except KeyError:
                            # Trace node was garbage collected, skip it
                            continue
                if ancestor_idx == -1:
                    if message_trace_node.parent == last_trace_node.root():  # pragma: no cover
                        ancestor_idx = 0
                    else:
                        logger.debug(f"DIVERGENCE:full_reset")
                        ancestor_idx = 0

                return True, ancestor_idx
            else:
                return False, -1

    def update(self, message: GuidanceMessage, topic=DEFAULT_TOPIC) -> None:
        from ..registry import get_bg_async, get_exchange

        out_messages: list[GuidanceMessage] = []

        # Metrics
        if isinstance(message, MetricMessage):
            if self._running:
                # logger.debug(f"RENDERER:metric:{message}")
                pass
            else:
                return

        if isinstance(message, ExecutionCompletedMessage):
            # Execution completed
            logger.debug("RENDERER:execution end")
            self._completed = True
            self._running = False

            if message.is_err:
                out_messages.append(MetricMessage(name="status", value="Error"))
            else:
                out_messages.append(MetricMessage(name="status", value="Done"))

            # Set up event for waiting on acknowledgment BEFORE sending the message
            self._ack_received_event = threading.Event()
            logger.debug(
                f"RENDERER:setup ack event for ExecutionCompletedMessage, event={id(self._ack_received_event)}"
            )
        elif not self._running and isinstance(message, TraceMessage):
            # Execution started
            logger.debug(f"RENDERER:execution start, currently have {len(self._messages)} stored messages")

            # Clear any pending ack event from previous execution
            if self._ack_received_event is not None:
                logger.debug("RENDERER:clearing previous ack event")
                self._ack_received_event = None

            started_msg = ExecutionStartedMessage()
            out_messages.append(started_msg)
            out_messages.append(MetricMessage(name="status", value="Running"))

            _, self.last_cell_session_id = ipy_handle_event_once(
                partial(_on_cell_completion, weakref.ref(self)), "post_run_cell"
            )
            self._new_widget_needed = True
            self._running = True
            self._completed = False

            # TODO(nopdive): Fire off execution immediately to renderer subscribers. Review later.
            get_bg_async().call_soon_threadsafe(self.recv_queue.put_nowait, serialize_message(started_msg))

        # Check if message has diverged from prev messages
        diverged, shared_ancestor_idx = self.has_divergence(message)
        if diverged:
            logger.debug(f"RENDERER:diverged, keeping {shared_ancestor_idx} messages out of {len(self._messages)}")
            out_messages.append(ResetDisplayMessage())
            out_messages[len(out_messages) :] = self._messages[:shared_ancestor_idx]
            logger.debug(
                f"RENDERER:cleared {len(self._messages)} messages, keeping {len(self._messages[:shared_ancestor_idx])}"
            )
            self._messages.clear()

        # Check if requested to reset and replay
        if isinstance(message, OutputRequestMessage):
            logger.debug(f"RENDERER:replay:{message}")
            out_messages.append(ResetDisplayMessage())
            out_messages[len(out_messages) :] = self._messages[:]

        # Reset if needed
        if self._new_widget_needed:
            logger.debug("RENDERER:new widget needed")
            # Store existing messages before clearing
            existing_messages = self._messages[:]

            # Clear messages
            self._messages = []

            if self.stitch_widget is not None and self.stitch_widget_observed:
                self.stitch_widget.unobserve(self._stitch_on_clientmsg, names="clientmsg")
                self.stitch_widget_observed = False
                logger.debug("RENDERER:widget unobserved (new)")

            self.stitch_widget = _create_stitch_widget()
            self._stitch_on_clientmsg = partial(_on_stitch_clientmsg, weakref.ref(self.recv_queue))
            self.stitch_widget.observe(self._stitch_on_clientmsg, names="clientmsg")
            self.stitch_widget_observed = True
            logger.debug("RENDERER:widget observed (new)")

            # Redraw
            display(self.stitch_widget)

            self._new_widget_needed = False

            # Replay existing messages to the new widget
            if existing_messages:
                logger.debug(f"RENDERER:replaying {len(existing_messages)} existing messages")
                # Add all existing messages to the outgoing queue
                out_messages.append(ResetDisplayMessage())
                out_messages.extend(existing_messages)

        # Append current message to outgoing
        out_messages.append(message)
        logger.debug(
            f"RENDERER:appending message {message.class_name}, will have {len(self._messages) + len(out_messages)} total messages"
        )

        # Send outgoing messages to client
        for out_message in out_messages:
            if isinstance(out_message, TraceMessage):
                # logger.debug(f"RENDERER:out:{out_message}")
                self.last_trace_id = out_message.trace_id

            self._messages.append(out_message)

            # Track for debug if enabled
            if self._debug_enabled:
                self._debug_messages.append(out_message)

            get_bg_async().call_soon_threadsafe(self.send_queue.put_nowait, out_message)

        # Wait for ExecutionCompletedAck if we set up an event
        if self._ack_received_event is not None:
            logger.debug("RENDERER:waiting for ExecutionCompletedAck...")

            # Wait up to 2 seconds for acknowledgment
            ack_received = self._ack_received_event.wait(timeout=2.0)

            if ack_received:
                logger.debug("RENDERER:ExecutionCompletedAck received - continuing")
                # Clear the event since we got the ack
                self._ack_received_event = None
            else:
                logger.warning("RENDERER:ExecutionCompletedAck timeout - continuing anyway")
                # Don't clear the event - let it stay for late acks

    def enable_debug(self) -> None:
        """Enable debug mode in the widget to capture message history."""
        from ..registry import get_bg_async

        self._debug_enabled = True
        self._debug_messages = []  # Clear previous messages

    def clear_debug_data(self) -> None:
        """Clear captured debug messages."""
        self._debug_messages = []
        logger.info("Debug messages cleared")

    def get_debug_data(self) -> Optional[str]:
        """Get debug data as a JSON string.

        Returns:
            JSON string containing debug data, or None if no data available
        """
        import datetime
        import json

        if not self._debug_enabled:
            logger.warning("Debug mode not enabled - call enable_debug() first")
            return None

        if not self._debug_messages:
            logger.warning("No debug messages captured yet")
            return None

        def make_json_serializable(obj):
            """Convert objects to JSON serializable format."""
            if isinstance(obj, bytes):
                # Convert bytes to base64 string for JSON serialization
                import base64

                return base64.b64encode(obj).decode("utf-8")
            elif isinstance(obj, dict):
                return {k: make_json_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_json_serializable(item) for item in obj]
            elif hasattr(obj, "model_dump"):
                # Pydantic models
                return make_json_serializable(obj.model_dump())
            else:
                return obj

        debug_data = {
            "timestamp": datetime.datetime.now().isoformat(),
            "messageCount": len(self._debug_messages),
            "messages": [
                make_json_serializable({"message_id": msg.message_id, "class_name": msg.class_name, **msg.model_dump()})
                for msg in self._debug_messages
            ],
        }

        return json.dumps(debug_data, indent=2)


class DoNothingRenderer(Renderer):
    """It does nothing."""

    def __init__(self, trace_handler: TraceHandler) -> None:
        """Initializes.

        Args:
            trace_handler: Trace handler of an engine.
        """
        self._trace_handler = trace_handler
        super().__init__()

    def update(self, message: GuidanceMessage, topic: str = DEFAULT_TOPIC) -> None:
        pass


class AutoRenderer(Renderer):
    """Automatically detects which renderer to use based on environment."""

    def __init__(self, trace_handler: TraceHandler):
        """Initializes.

        Args:
            trace_handler: Trace handler of an engine.
        """
        self._env = Environment()

        self._renderer: Renderer
        if self._env.is_notebook():
            if stitch_installed:
                self._renderer = JupyterWidgetRenderer(trace_handler=trace_handler)
            else:
                self._renderer = DoNothingRenderer(trace_handler=trace_handler)
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

    def update(self, message: GuidanceMessage, topic: str = DEFAULT_TOPIC) -> None:
        self._renderer.update(message, topic=topic)

    def renderer_type(self) -> type:
        """Type of renderer that has been selected based on environment.

        Returns:
            Type of selected renderer.
        """
        return type(self._renderer)
