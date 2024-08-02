from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, Field, NonNegativeInt, RootModel, model_validator, computed_field
from typing_extensions import Annotated
from functools import cached_property


class GuidanceEngineMetrics(BaseModel):
    engine_input_tokens: NonNegativeInt = 0
    engine_output_tokens: NonNegativeInt = 0


class EngineCallResponse(BaseModel):
    new_bytes: bytes
    is_generated: bool
    new_bytes_prob: float
    capture_groups: dict
    capture_group_log_probs: dict
    new_token_count: NonNegativeInt


class GenData(BaseModel):
    tokens: list[int]
    mask: bytes
    temperature: float

    @computed_field  # type: ignore[misc]
    @cached_property
    def valid_next_tokens(self) -> list[int]:
        return [i for i, b in enumerate(self.mask) if b != 0]


class LLProgressCapture(BaseModel):
    object: Literal["capture"]
    name: str
    hex: str
    log_prob: float
    list_append: bool = False

    @model_validator(mode="before")
    def strip_list_append_prefix(cls, values):
        name = values["name"]
        if name.startswith("__LIST_APPEND:"):
            values["name"] = name[14:]
            # Override whatever was set
            values["list_append"] = True
        return values


class LLProgressText(BaseModel):
    object: Literal["text"]
    hex: str
    num_tokens: NonNegativeInt
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
                if j.list_append:
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
