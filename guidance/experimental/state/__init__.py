from typing import Union

from ...models.base._state import ChatState, CompletionState, RoleEnd, RoleStart
from .openai import OpenAIState
from .transformers import (
    BaseTransformersChatState,
    Llama3TransformersState,
)

State = Union[ChatState, CompletionState]

__all__ = [
    "State",
    "ChatState",
    "CompletionState",
    "RoleEnd",
    "RoleStart",
    "OpenAIState",
    "TransformersUnstructuredState",
    "Llama3TransformersState",
    "BaseTransformersChatState",
]
