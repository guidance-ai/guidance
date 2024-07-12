from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, Field, RootModel
from typing_extensions import Annotated

from . import _serialization_pb2


# TODO: pydantic rather than protobuf?
class EngineCallResponse:
    new_bytes: bytes
    is_generated: bool
    new_bytes_prob: float
    capture_groups: dict
    capture_group_log_probs: dict
    new_token_count: int

    def __init__(
        self,
        new_bytes,
        is_generated,
        new_bytes_prob,
        capture_groups,
        capture_group_log_probs,
        new_token_count,
    ):
        self.new_bytes = new_bytes
        self.is_generated = is_generated
        self.new_bytes_prob = new_bytes_prob
        self.capture_groups = capture_groups
        self.capture_group_log_probs = capture_group_log_probs
        self.new_token_count = new_token_count

    def _to_proto(self):
        """Converts an EngineCallResponse object to its Protobuf representation.

        Returns:
            engine_response_pb2.EngineCallResponse: The Protobuf equivalent of this object.
        """
        groups = {}
        group_log_probs = {}

        def to_protobuf_value(
            v: Union[str, bytes, float, list]
        ) -> _serialization_pb2.Value:
            """Convert Python values to Protobuf Value messages."""
            value = _serialization_pb2.Value()
            if isinstance(v, str):
                value.string_value = v
            elif isinstance(v, bytes):
                value.bytes_value = v
            elif isinstance(v, float):
                value.float_value = v
            elif isinstance(v, list):
                for item in v:
                    value.list_value.values.append(to_protobuf_value(item))
            else:
                raise TypeError(f"Unsupported type: {type(v)}")
            return value

        for k, v in self.capture_groups.items():
            groups[k] = to_protobuf_value(v)

        for k, v in self.capture_group_log_probs.items():
            group_log_probs[k] = to_protobuf_value(v)

        return _serialization_pb2.EngineCallResponse(
            new_bytes=self.new_bytes,
            is_generated=self.is_generated,
            new_bytes_prob=self.new_bytes_prob,
            capture_groups=groups,
            capture_group_log_probs=group_log_probs,
            new_token_count=self.new_token_count,
        )

    def encode(self, charset):
        """Used to support FastAPI encoding of EngineCallResponse objects."""
        return self.serialize()

    def serialize(self):
        proto = self._to_proto()
        return proto.SerializeToString()

    @staticmethod
    def deserialize(byte_data):
        proto = _serialization_pb2.EngineCallResponse()
        proto.ParseFromString(byte_data)

        def from_protobuf_value(
            value: _serialization_pb2.Value,
        ) -> Union[str, bytes, float, list]:
            """Convert Protobuf Value message to Python values"""
            if value.HasField("string_value"):
                return value.string_value
            elif value.HasField("bytes_value"):
                return value.bytes_value
            elif value.HasField("float_value"):
                return value.float_value
            elif value.HasField("list_value"):
                return [from_protobuf_value(item) for item in value.list_value.values]
            else:
                raise ValueError("Protobuf Value message has no recognized field set")

        groups = {}
        for k, v in proto.capture_groups.items():
            groups[k] = from_protobuf_value(v)

        group_log_probs = {}
        for k, v in proto.capture_group_log_probs.items():
            group_log_probs[k] = from_protobuf_value(v)

        return EngineCallResponse(
            new_bytes=proto.new_bytes,
            is_generated=proto.is_generated,
            new_bytes_prob=proto.new_bytes_prob,
            capture_groups=groups,
            capture_group_log_probs=group_log_probs,
            new_token_count=proto.new_token_count,
        )


class LLProgressCapture(BaseModel):
    object: Literal["capture"]
    name: str
    hex: str
    log_prob: float


class LLProgressText(BaseModel):
    object: Literal["text"]
    hex: str
    num_tokens: int
    log_prob: float
    is_generated: bool


class LLProgressFinalText(BaseModel):
    object: Literal["final_text"]
    # we don't need to handle this for now


LLProgressItem = Annotated[
    Union[LLProgressCapture, LLProgressText, LLProgressFinalText],
    Field(discriminator="object"),
]


class LLProgress(RootModel):
    root: list[LLProgressItem]

    def to_engine_call_response(self) -> EngineCallResponse:
        new_bytes = b""
        new_token_count = 0
        new_bytes_prob = 0.0
        is_generated = False
        capture_groups: dict[str, Any] = {}
        capture_group_log_probs: dict[str, Any] = {}
        num_text_entries = 0

        for j in self.root:
            if isinstance(j, LLProgressCapture):
                is_generated = True
                cname = j.name
                data = bytes.fromhex(j.hex)
                if cname.startswith("__LIST_APPEND:"):
                    cname = cname[14:]
                    if cname not in capture_groups or not isinstance(
                        capture_groups[cname], list
                    ):
                        capture_groups[cname] = []
                        capture_group_log_probs[cname] = []
                    capture_groups[cname].append(data)
                    capture_group_log_probs[cname].append(j.log_prob)
                else:
                    capture_groups[cname] = data
                    capture_group_log_probs[cname] = j.log_prob
            elif isinstance(j, LLProgressText):
                # it actually should only happen once per round...
                new_bytes += bytes.fromhex(j.hex)
                new_token_count += j.num_tokens
                new_bytes_prob += j.log_prob
                is_generated |= j.is_generated
                num_text_entries += 1
        if num_text_entries > 0:
            new_bytes_prob /= num_text_entries

        return EngineCallResponse(
            new_bytes=new_bytes,
            new_token_count=new_token_count,
            new_bytes_prob=new_bytes_prob,
            is_generated=is_generated,
            capture_groups=capture_groups,
            capture_group_log_probs=capture_group_log_probs,
        )


class LLInterpreterResponse(BaseModel):
    progress: LLProgress
    stop: bool
    temperature: Optional[float]
