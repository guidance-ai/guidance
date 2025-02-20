from typing import Union

from ...models._base._state import BaseChatState, BaseCompletionState, RoleEnd, RoleStart
from .openai import OpenAIState
from .transformers import (
    BaseTransformersChatState,
    Llama3TransformersState,
)

State = Union[BaseChatState, BaseCompletionState]

__all__ = [
    "State",
    "BaseChatState",
    "BaseCompletionState",
    "RoleEnd",
    "RoleStart",
    "OpenAIState",
    "TransformersUnstructuredState",
    "Llama3TransformersState",
    "BaseTransformersChatState",
]
