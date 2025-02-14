from functools import lru_cache

from .trace import TraceHandler
from .visual import AutoRenderer, Renderer


@lru_cache(maxsize=1)
def get_trace_handler() -> TraceHandler:
    return TraceHandler()


@lru_cache(maxsize=1)
def get_renderer(trace_handler: TraceHandler) -> Renderer:
    return AutoRenderer(trace_handler)
