from abc import ABC, abstractmethod
from typing import Iterable

from guidance.models import Transformers

from .ast import ContentChunk, Node
from .state import APIState


class Client(ABC):
    @abstractmethod
    def run(self, state: APIState, node: Node) -> Iterable[ContentChunk]:
        pass


class TransformersClient(Client):
    def __init__(self, model_id: str = "microsoft/Phi-3-mini-4k-instruct"):
        guidance_model = Transformers(model_id)
        self.engine = guidance_model.engine

    def run(self, state: APIState, node: Node) -> Iterable[ContentChunk]:
        if isinstance(node, str):
            yield node
        else:
            state_dict = state.get_state()
            prefill = state.get_active_message()
            prompt = apply_chat_template(
                state_dict["messages"], prefill, None, None, self.engine.tokenizer._orig_tokenizer
            )
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
