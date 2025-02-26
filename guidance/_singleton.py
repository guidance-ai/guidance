# NOTE(nopdive): Code adjusted to handle set/get with thread safety.

import threading
from .trace import TraceHandler
from .visual import AutoRenderer, Renderer


_trace_handler_lock = threading.Lock()
_trace_handler = None

_renderer_lock = threading.Lock()
_renderer = None


def get_trace_handler() -> TraceHandler:
    global _trace_handler
    with _trace_handler_lock:
        if _trace_handler is None:
            _trace_handler = TraceHandler()
    return _trace_handler


def get_renderer(trace_handler: TraceHandler) -> Renderer:
    global _renderer
    with _renderer_lock:
        if _renderer is None:
            _renderer = AutoRenderer(trace_handler)
    return _renderer


def set_renderer(renderer: Renderer) -> None:
    global _renderer
    with _renderer_lock:
        _renderer = renderer