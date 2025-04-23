"""Messages that used between server (usually Jupyter Python kernel) and client.

Messages are required to be added to the model registry for serialization.
"""
from typing import Optional, Dict, Union, Any, MutableMapping

from pydantic import BaseModel, Field, model_validator

from ..trace import NodeAttr
import json
import base64

def byte_to_base64(b: bytes) -> str:
    return base64.b64encode(b).decode('utf-8')

def base64_to_bytes(base64_str: str) -> bytes:
    return base64.b64decode(base64_str)


def recursive_model_dump_json(model):
    if isinstance(model, BaseModel):
        model_dict = model.model_dump()
        for field_name, field_value in model_dict.items():
            if isinstance(field_value, bytes):
                model_dict[field_name] = byte_to_base64(field_value)
            elif isinstance(field_value, dict):
                model_dict[field_name] = recursive_model_dump_json(field_value)
            elif isinstance(field_value, list):
                model_dict[field_name] = [recursive_model_dump_json(item) if isinstance(item, BaseModel) else item for item in field_value]
        return model_dict
    return model


def recursive_model_parse_raw(model, data):
    if isinstance(model, BaseModel):
        model_instance = model.model_validate(data)
        for field_name, field_value in model_instance.model_dump().items():
            if isinstance(field_value, str):
                field_type = model.__annotations__.get(field_name)
                if field_type == bytes:
                    model_instance.__setattr__(field_name, base64_to_bytes(field_value))
            elif isinstance(field_value, dict):
                model_instance.__setattr__(field_name, recursive_model_parse_raw(model, field_value))
            elif isinstance(field_value, list):
                model_instance.__setattr__(field_name, [recursive_model_parse_raw(model, item) if isinstance(item, dict) else item for item in field_value])
        return model_instance
    return model


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
    
    class Config:
        json_encoders = { bytes: byte_to_base64 }
    
    @model_validator(mode="before")
    def decode_base64_to_bytes(cls, values):
        if isinstance(values, dict):
            return _decode_base64_to_bytes(cls, values)
        return values

def _decode_base64_to_bytes(cls, values: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
    for field_name, field_value in values.items():
        field_type = cls.__annotations__.get(field_name)
        if field_type == bytes and isinstance(field_value, str):
            try:
                values[field_name] = base64_to_bytes(field_value)
            except Exception as e:  # pragma: no cover
                raise ValueError(f"Failed to decode base64 string for '{field_name}': {e}")
    return values

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


model_registry: Dict[str, type[GuidanceMessage]] = {
    'TraceMessage': TraceMessage,
    'ExecutionStartedMessage': ExecutionStartedMessage,
    'ExecutionCompletedMessage': ExecutionCompletedMessage,
    'ResetDisplayMessage': ResetDisplayMessage,
    'ClientReadyMessage': ClientReadyMessage,
    'ClientReadyAckMessage': ClientReadyAckMessage,
    'OutputRequestMessage': OutputRequestMessage,
    'MetricMessage': MetricMessage,
}


def serialize_message(message: GuidanceMessage) -> str:
    """ Serializes guidance message.

    Args:
        message: Message to be serialized.

    Returns:
        Serialized message in JSON format.
    """
    # message_json = message.model_dump_json(indent=2, serialize_as_any=True)
    message_json = recursive_model_dump_json(message)
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
    message = recursive_model_parse_raw(model_class, data_json)
    return message