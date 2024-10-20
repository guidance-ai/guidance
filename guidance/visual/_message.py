from typing import Optional, Dict, Union

from pydantic import BaseModel, Field, PrivateAttr

from guidance._schema import GenTokenExtra
from ..trace import NodeAttr
import json


_msg_counter: int = -1
class GuidanceMessage(BaseModel):
    message_id: int = Field(default=None)
    class_name: str = ""

    def __init__(self, **kwargs):
        global _msg_counter

        kwargs["class_name"] = self.__class__.__name__
        _msg_counter += 1
        kwargs["message_id"] = _msg_counter
        super().__init__(**kwargs)


class TraceMessage(GuidanceMessage):
    trace_id: int
    parent_trace_id: Optional[int]
    node_attr: Optional[NodeAttr]


class MetricMessage(GuidanceMessage):
    name: str
    value: Union[float, str, list[float], list[str]]
    scalar: bool = True


class ExecutionCompletedMessage(GuidanceMessage):
    last_trace_id: Optional[int]


class ExecutionCompletedOutputMessage(GuidanceMessage):
    trace_id: int
    text: str
    tokens: list[GenTokenExtra]


class ResetDisplayMessage(GuidanceMessage):
    pass


class ClientReadyMessage(GuidanceMessage):
    pass


class ClientReadyAckMessage(GuidanceMessage):
    pass


class OutputRequestMessage(GuidanceMessage):
    pass


model_registry: Dict[str, type(GuidanceMessage)] = {
    'TraceMessage': TraceMessage,
    'ExecutionCompleted': ExecutionCompletedMessage,
    'ExecutionCompletedOutputMessage': ExecutionCompletedOutputMessage,
    'ResetDisplayMessage': ResetDisplayMessage,
    'ClientReadyMessage': ClientReadyMessage,
    'ClientReadyAckMessage': ClientReadyAckMessage,
    'OutputRequestMessage': OutputRequestMessage,
    'MetricMessage': MetricMessage,
}


def serialize_message(message: GuidanceMessage) -> str:
    message_json = message.model_dump_json(indent=2, serialize_as_any=True)
    return message_json


def deserialize_message(data: str) -> GuidanceMessage:
    data_json = json.loads(data)
    class_name = data_json.get("class_name")
    model_class = model_registry.get(class_name)
    if not model_class:
        raise ValueError(f"Unknown class_name: {class_name}")
    return model_class.model_validate_json(data)