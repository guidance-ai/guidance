from typing import Optional
from pydantic import BaseModel
from ..trace import NodeAttr


class GuidanceMessage(BaseModel):
    class_name: str = ""

    def __init__(self, **kwargs):
        kwargs["class_name"] = self.__class__.__name__
        super().__init__(**kwargs)


class TraceMessage(GuidanceMessage):
    trace_id: int
    parent_trace_id: Optional[int]
    node_attr: Optional[NodeAttr]


class ResetDisplayMessage(GuidanceMessage):
    pass


class HeartbeatMessage(GuidanceMessage):
    pass