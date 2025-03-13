# NOTE(nopdive): Consider moving singleton factories to registry static class.

import threading

from .._schema import GuidanceEngineMetrics
from ..metrics import Monitor, PeriodicMetricsGenerator
from ..trace import TraceHandler
from ..visual import AutoRenderer, Renderer, TopicExchange
from .._bg import BackgroundAsync

_monitor_lock = threading.Lock()
_monitor = None
_periodic_metrics_gen = None

_bg_async_lock = threading.Lock()
_bg_async = None

_exchange_lock = threading.Lock()
_exchange = None

_trace_handler_lock = threading.Lock()
_trace_handler = None

_renderer_lock = threading.Lock()
_renderer = None


def get_monitor() -> Monitor:
    global _monitor
    global _monitor_lock
    global _periodic_metrics_gen

    with _monitor_lock:
        if _monitor is None:
            _monitor = Monitor(GuidanceEngineMetrics())
            _monitor.start()
            _periodic_metrics_gen = PeriodicMetricsGenerator(_monitor)
            _periodic_metrics_gen.start()
    return _monitor


def get_bg_async() -> BackgroundAsync:
    global _bg_async
    global _bg_async_lock

    with _bg_async_lock:
        if _bg_async is None:
            _bg_async = BackgroundAsync()
    return _bg_async


def get_exchange() -> TopicExchange:
    global _exchange
    global _exchange_lock

    with _exchange_lock:
        if _exchange is None:
            _exchange = TopicExchange()
    return _exchange


def get_trace_handler() -> TraceHandler:
    global _trace_handler
    global _trace_handler_lock

    with _trace_handler_lock:
        if _trace_handler is None:
            _trace_handler = TraceHandler()
    return _trace_handler


def get_renderer() -> Renderer:
    global _renderer
    global _renderer_lock

    with _renderer_lock:
        trace_handler = get_trace_handler()
        if _renderer is None:
            _renderer = AutoRenderer(trace_handler)
    return _renderer


def set_renderer(renderer: Renderer) -> None:
    global _renderer
    global _renderer_lock

    with _renderer_lock:
        _renderer = renderer
