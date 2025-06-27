"""Trace tree of inputs & outputs generated from a guidance program.

The first implementation aims for simplicity.
Once benchmark figures are out, we'll figure out what to optimize.

The most critical class is the trace handler. See its documentation for trace design & motivations.
"""

from ._trace import (
    AudioOutput,
    Backtrack,
    CaptureOutput,
    EmbeddedInput,
    ImageInput,
    ImageOutput,
    InputAttr,
    LiteralInput,
    NodeAttr,
    OutputAttr,
    RoleCloserInput,
    RoleOpenerInput,
    StatefulGuidanceInput,
    StatelessGuidanceInput,
    TextOutput,
    Token,
    TokenOutput,
    TraceHandler,
    TraceNode,
    VideoOutput,
)

__all__ = [
    "TraceNode",
    "TraceHandler",
    "InputAttr",
    "OutputAttr",
    "NodeAttr",
    "LiteralInput",
    "StatefulGuidanceInput",
    "StatelessGuidanceInput",
    "RoleOpenerInput",
    "RoleCloserInput",
    "EmbeddedInput",
    "ImageInput",
    "AudioOutput",
    "ImageOutput",
    "VideoOutput",
    "CaptureOutput",
    "TextOutput",
    "TokenOutput",
    "Token",
    "Backtrack",
]
