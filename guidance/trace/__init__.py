"""Trace tree of inputs & outputs generated from a guidance program.

The first implementation aims for simplicity.
Once benchmark figures are out, we'll figure out what to optimize.

The most critical class is the trace handler. See its documentation for trace design & motivations.
"""

from ._trace import NodeAttr, InputAttr, OutputAttr, StatefulGuidanceInput, StatelessGuidanceInput
from ._trace import LiteralInput, EmbeddedInput, ImageInput, RoleCloserInput, RoleOpenerInput
from ._trace import TextOutput, ImageOutput, CaptureOutput, TraceNode, TraceHandler
