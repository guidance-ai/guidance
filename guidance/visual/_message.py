from typing import Optional
from pydantic import BaseModel
from ..trace import NodeAttr, TraceNode


class Message(BaseModel):
    message_type: str = ""

    def __init__(self, **kwargs):
        kwargs["message_type"] = self.__class__.__name__
        super().__init__(**kwargs)


class TraceMessage(Message):
    trace_id: int
    parent_trace_id: Optional[int]
    trace_node: TraceNode
    node_attr: Optional[NodeAttr]


class ResetDisplayMessage(Message):
    pass
