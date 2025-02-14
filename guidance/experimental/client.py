import json
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic, Iterable, Optional, TypeVar, Union, cast

from openai import OpenAI
from openai.types.chat import ChatCompletionChunk as OpenAIChatCompletionChunk
from typing_extensions import assert_never

from guidance._grammar import Function, Gen, Join
from guidance.models import Transformers
from guidance.models._model import Engine
from guidance.trace._trace import LiteralInput

from .ast import CaptureOutput, ContentChunk, ImageBlob, Node, TextOutput
from .state import BaseTransformersChatState, ChatState, CompletionState, State
from .state.openai import OpenAIState

S = TypeVar("S", bound=State)


class Client(ABC, Generic[S]):
    @abstractmethod
    def run(self, state: S, node: Node) -> Iterable[ContentChunk]:
        pass

    @abstractmethod
    def initial_state(self) -> S:
        pass

    def format_state(self, state: S) -> str:
        return json.dumps(state.get_state(), indent=2)


class OpenAIClient(Client[OpenAIState]):
    def __init__(self, model: str = "gpt-4o-mini", api_key: Optional[str] = None):
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def initial_state(self) -> OpenAIState:
        return OpenAIState.from_openai_model(self.model)

    def run(self, state: OpenAIState, node: Node) -> Iterable[ContentChunk]:
        if isinstance(node, str):
            yield LiteralInput(value=node)
            return
        elif isinstance(node, ImageBlob):
            yield node
            return

        oai_state = state.get_state()
        if oai_state["prefill"] is not None:
            raise ValueError("Prefill not supported for OpenAI")
        if oai_state["active_role"] != "assistant":
            raise ValueError("Active role must be assistant for OpenAI")

        messages = oai_state["messages"]
        if isinstance(node, Join) and len(node.values) == 1:
            # TODO: just a hack for the moment
            node = node.values[0]

        if isinstance(node, Gen):
            if node.capture_name:
                raise NotImplementedError("Captures not yet supported for OpenAI")
            if node.body_regex != "(?s:.*)":
                raise ValueError("Body regex not supported for OpenAI")
            if node.stop_regex:
                raise ValueError("Stop regex not supported for OpenAI")
            if node.save_stop_text:
                raise ValueError("Save stop text not supported for OpenAI")

            responses = self.client.chat.completions.create(
                model=self.model,
                messages=messages,  # type: ignore[arg-type]
                max_tokens=node.max_tokens,
                temperature=node.temperature,
                logprobs=True,
                stream=True,
            )
            for response in cast(Iterable[OpenAIChatCompletionChunk], responses):
                choice = response.choices[0]
                delta = choice.delta
                if delta.content is not None:
                    content = delta.content
                    if len(content) == 0:
                        continue
                    yield TextOutput(
                        value=delta.content,
                        is_generated=True,
                        # TODO: actually get tokens from this and be less lazy
                        prob=2.718 ** choice.logprobs.content[0].logprob,  # type: ignore[union-attr,index]
                    )
                    continue
                if choice.finish_reason is not None:
                    # TODO: handle finish_reason elegantly
                    break
                raise NotImplementedError(f"Unknown delta: {delta}")
        else:
            raise NotImplementedError(f"Unknown node: {node}")


class GuidanceClient(Client[S], ABC):
    def __init__(self, engine: Engine):
        self.engine = engine

    @abstractmethod
    def build_prompt(self, state: S) -> str:
        pass

    def run(self, state: S, node: Node) -> Iterable[ContentChunk]:
        if isinstance(node, str):
            yield LiteralInput(value=node)
        elif isinstance(node, ImageBlob):
            yield node

        elif isinstance(node, Function):
            prompt = self.build_prompt(state)
            engine_gen = self.engine(
                prompt,
                node,
                ensure_bos_token=False,
                echo=False,
            )

            def partial_decode(data: bytes) -> tuple[str, bytes]:
                try:
                    return (data.decode("utf-8"), b"")
                except UnicodeDecodeError as e:
                    valid_part = data[: e.start].decode("utf-8")
                    delayed_part = data[e.start :]
                    return (valid_part, delayed_part)

            delayed_bytes = b""
            for chunk in engine_gen:
                generated_bytes = delayed_bytes + chunk.generated_bytes
                generated_text, delayed_bytes = partial_decode(generated_bytes)
                ff_bytes = delayed_bytes + chunk.force_forwarded_bytes
                ff_text, delayed_bytes = partial_decode(ff_bytes)

                if generated_bytes:
                    yield TextOutput(
                        value=generated_text,
                        is_generated=True,
                        prob=chunk.new_bytes_prob,
                        token_count=len(chunk.generated_tokens),
                        tokens=chunk.generated_tokens,
                    )
                if ff_bytes:
                    yield TextOutput(
                        value=ff_text,
                        is_generated=False,
                        prob=chunk.new_bytes_prob,
                        token_count=len(chunk.force_forwarded_tokens),
                        tokens=chunk.force_forwarded_tokens,
                    )

                for name in chunk.capture_groups.keys():
                    values = chunk.capture_groups[name]
                    log_probs = chunk.capture_group_log_probs[name]
                    if isinstance(values, list):
                        assert isinstance(log_probs, list) and len(log_probs) == len(values)
                        list_append = True
                    else:
                        values = [values]
                        log_probs = [log_probs]
                        list_append = False

                    for value, log_prob in zip(values, log_probs):
                        yield CaptureOutput(
                            name=name,
                            value=value,
                            is_append=list_append,
                            # TODO: let this be Optional?
                            log_probs=log_prob,
                        )

            if delayed_bytes:
                raise RuntimeError("Shouldn't have any delayed bytes left...")

        else:
            raise NotImplementedError(f"Unknown node: {node}")


class TransformersClient(GuidanceClient[Union[CompletionState, BaseTransformersChatState]]):
    def __init__(self, model_id: str = "microsoft/Phi-3-mini-4k-instruct", **model_kwargs):
        self.model_id = model_id
        guidance_model = Transformers(model_id, **model_kwargs)
        super().__init__(guidance_model.engine)

    def initial_state(self) -> Union[CompletionState, BaseTransformersChatState]:
        return BaseTransformersChatState.from_model_id(self.model_id)

    def build_prompt(self, state: Union[CompletionState, BaseTransformersChatState]) -> str:
        if isinstance(state, ChatState):
            chat_state = state.get_state()
            prefill = chat_state["prefill"]
            if prefill is None:
                role = chat_state["active_role"]
                if role is None:
                    raise ValueError("Can't generate with no active role")
                prefill = {"role": "user", "content": ""}
            prompt = apply_chat_template(
                list(chat_state["messages"]),
                chat_state["prefill"],
                None,
                None,
                self.engine.tokenizer._orig_tokenizer,  # type: ignore[attr-defined]
            )
        elif isinstance(state, CompletionState):
            completion_state = state.get_state()
            prompt = completion_state["prompt"]
        else:
            if TYPE_CHECKING:
                assert_never(state)
            raise TypeError(f"Expected ChatState or CompletionState, got {type(state)}")

        return prompt


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
