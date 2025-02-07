import json
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Iterable

from typing_extensions import assert_never

from guidance.models import Transformers

from .ast import ContentChunk, Node
from .state import ChatState, CompletionState, State


class Client(ABC):
    @abstractmethod
    def run(self, state: State, node: Node) -> Iterable[ContentChunk]:
        pass

    def format_state(self, state: State) -> str:
        return json.dumps(state.get_state(), indent=2)


class TransformersClient(Client):
    def __init__(self, model_id: str = "microsoft/Phi-3-mini-4k-instruct"):
        guidance_model = Transformers(model_id)
        self.engine = guidance_model.engine

    def run(self, state: State, node: Node) -> Iterable[ContentChunk]:
        if isinstance(node, str):
            yield node
        else:
            if isinstance(state, ChatState):
                chat_state = state.get_state()
                prompt = apply_chat_template(
                    chat_state["messages"],
                    chat_state["prefill"],
                    None,
                    None,
                    self.engine.tokenizer._orig_tokenizer,
                )
            elif isinstance(state, CompletionState):
                completion_state = state.get_state()
                prompt = completion_state["prompt"]
            else:
                if TYPE_CHECKING:
                    assert_never(state)
                raise TypeError(f"Expected ChatState or CompletionState, got {type(state)}")

            engine_gen = self.engine(
                prompt,
                node,
                ensure_bos_token=False,
                echo=False,
            )
            for response in engine_gen:
                # breakpoint()
                yield response.new_bytes.decode("utf-8")


from typing import Any, Optional

from .state.transformers import TransformersMessage


def apply_chat_template(
    messages: list[TransformersMessage],
    prefill: Optional[TransformersMessage],
    tools: Optional[list[Any]],
    chat_template: Optional[str],
    tokenizer: Any,
) -> str:
    if prefill is None:
        sentinel_value = None
    elif prefill["content"]:
        sentinel_value = None
        messages = messages + [prefill]
    else:
        # This is a hack to get around the fact that apply_chat_template won't properly continue the final message
        # if it is empty. We add a sentinel value to the final message, and then remove it after the fact.
        sentinel_value = "<|FINAL_MESSAGE_SENTINEL_VALUE|>"
        messages = messages + [dict(role=prefill["role"], content=sentinel_value)]
    prompt = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        chat_template=chat_template,
        continue_final_message=True,
        add_generation_prompt=False,
        tokenize=False,
    )
    if sentinel_value is not None:
        prompt = prompt[: prompt.rindex(sentinel_value)]
    return prompt
