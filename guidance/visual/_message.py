"""Messages that used between server (usually Jupyter Python kernel) and client.

Messages are required to be added to the model registry for serialization.
"""
from typing import Optional, Dict, Union

from pydantic import BaseModel, Field

from guidance._schema import GenTokenExtra
from ..trace import NodeAttr
import json


_msg_counter: int = -1
class GuidanceMessage(BaseModel):
    """Message sent within Guidance layer."""

    message_id: int = Field(default=None)
    class_name: str = ""

    def __init__(self, **kwargs):
        global _msg_counter

        kwargs["class_name"] = self.__class__.__name__
        if kwargs.get("message_id") is None:
            _msg_counter += 1
            kwargs["message_id"] = _msg_counter
        super().__init__(**kwargs)


class TraceMessage(GuidanceMessage):
    """Update on a trace node."""
    trace_id: int
    parent_trace_id: Optional[int] = None
    node_attr: Optional[NodeAttr] = None


class MetricMessage(GuidanceMessage):
    """Metric that has been emitted."""
    name: str
    value: Union[float, str, list[float], list[str]] = Field(union_mode='left_to_right')
    scalar: bool = True


class ExecutionStartedMessage(GuidanceMessage):
    """Fired when renderer has started trace messages."""


class ExecutionCompletedMessage(GuidanceMessage):
    """Fired when renderer has completed trace messages.

    This functions as the last message sent to client.
    """
    last_trace_id: Optional[int] = None
    is_err: bool = False


class TokensMessage(GuidanceMessage):
    """Fired when trace messages are completed, with tokens for client."""
    trace_id: int
    text: str
    tokens: list[GenTokenExtra]

    def __str__(self):
        return f"message_id={self.message_id} class_name={self.class_name} trace_id={self.trace_id}"


class ResetDisplayMessage(GuidanceMessage):
    """Instructs client to reset the display, removing all output."""
    pass


class ClientReadyMessage(GuidanceMessage):
    """Fired when client is ready to receive messages."""
    pass


class ClientReadyAckMessage(GuidanceMessage):
    """Fired when server acknowledges client readiness."""
    pass


class OutputRequestMessage(GuidanceMessage):
    """Fired when client requests tokens from server."""
    pass


model_registry: Dict[str, type(GuidanceMessage)] = {
    'TraceMessage': TraceMessage,
    'ExecutionStartedMessage': ExecutionStartedMessage,
    'ExecutionCompletedMessage': ExecutionCompletedMessage,
    'ExecutionCompletedOutputMessage': TokensMessage,
    'ResetDisplayMessage': ResetDisplayMessage,
    'ClientReadyMessage': ClientReadyMessage,
    'ClientReadyAckMessage': ClientReadyAckMessage,
    'OutputRequestMessage': OutputRequestMessage,
    'MetricMessage': MetricMessage,
    'TokensMessage': TokensMessage,
}


def serialize_message(message: GuidanceMessage) -> str:
    """ Serializes guidance message.

    Args:
        message: Message to be serialized.

    Returns:
        Serialized message in JSON format.
    """
    message_json = message.model_dump_json(indent=2, serialize_as_any=True)
    return message_json


def deserialize_message(data: str) -> GuidanceMessage:
    """ Deserializes string into a guidance message.

    Args:
        data: JSON string to be deserialized.

    Returns:
        Guidance message.
    """
    data_json = json.loads(data)
    class_name = data_json.get("class_name")
    model_class = model_registry.get(class_name)
    if not model_class:
        raise ValueError(f"Unknown class_name: {class_name}")
    return model_class.model_validate_json(data)