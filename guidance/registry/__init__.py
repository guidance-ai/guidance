"""Registry module that contains singletons."""
# NOTE(nopdive): Consider moving singleton factories to registry static class.

import threading
from ..trace import TraceHandler
from ..visual import AutoRenderer, Renderer, TopicExchange

_exchange_lock = threading.Lock()
_exchange = None

_trace_handler_lock = threading.Lock()
_trace_handler = None

_renderer_lock = threading.Lock()
_renderer = None


def get_exchange() -> TopicExchange:
    global _exchange
    with _exchange_lock:
        if _exchange is None:
            _exchange = TopicExchange()
    return _exchange


def get_trace_handler() -> TraceHandler:
    global _trace_handler
    with _trace_handler_lock:
        if _trace_handler is None:
            _trace_handler = TraceHandler()
    return _trace_handler


def get_renderer(trace_handler: TraceHandler) -> Renderer:
    # TODO(nopdive): Remove trace handler argument.

    global _renderer
    with _renderer_lock:
        if _renderer is None:
            _renderer = AutoRenderer(trace_handler)
    return _renderer


def set_renderer(renderer: Renderer) -> None:
    global _renderer
    with _renderer_lock:
        _renderer = renderer


__all__ = [
    "get_renderer",
    "set_renderer",
    "get_trace_handler",
    "get_exchange",
]