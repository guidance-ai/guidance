"""Messages that used between server (usually Jupyter Python kernel) and client.

Messages are required to be added to the model registry for serialization.
"""

from itertools import count
from typing import Annotated, ClassVar, Optional, Union

from pydantic import BaseModel, Discriminator, Field, Tag, TypeAdapter, computed_field, model_validator

from ..trace import NodeAttr


class GuidanceMessage(BaseModel):
    """Message sent within Guidance layer."""

    message_id: int = Field(default_factory=count().__next__)

    _subclasses: ClassVar[set[type["GuidanceMessage"]]] = set()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._subclasses.add(cls)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def class_name(self) -> str:
        """Class name of the message."""
        return self.__class__.__name__

    @model_validator(mode="before")
    def validate_class_name(cls, data):
        if isinstance(data, dict):
            if "class_name" in data and data["class_name"] != cls.__name__:
                raise ValueError(f"mismatched class name: {data['class_name']}, expected: {cls.__name__}")
        return data

    @classmethod
    def as_discriminated_union(cls) -> type["GuidanceMessage"]:
        return Annotated[
            Union[tuple(Annotated[tp, Tag(tp.__name__)] for tp in cls._subclasses)],
            Discriminator(
                lambda x: x["class_name"] if isinstance(x, dict) else x.class_name,
            ),
        ]  # type: ignore[return-value]


class TraceMessage(GuidanceMessage):
    """Update on a trace node."""

    trace_id: int
    parent_trace_id: Optional[int] = None
    node_attr: Optional[NodeAttr.as_discriminated_union()] = None  # type: ignore


class MetricMessage(GuidanceMessage):
    """Metric that has been emitted."""

    name: str
    value: Union[float, str, list[float], list[str]] = Field(union_mode="left_to_right")
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


class ExecutionCompletedAckMessage(GuidanceMessage):
    """Fired when client acknowledges execution completion."""
    pass

def serialize_message(message: GuidanceMessage) -> str:
    """Serializes guidance message.

    Args:
        message: Message to be serialized.

    Returns:
        Serialized message in JSON format.
    """
    return message.model_dump_json()


def deserialize_message(data: str) -> GuidanceMessage:
    """Deserializes string into a guidance message.

    Args:
        data: JSON string to be deserialized.

    Returns:
        Guidance message.
    """
    return TypeAdapter(GuidanceMessage.as_discriminated_union()).validate_json(data)
