import base64
from io import BytesIO
from typing import Iterable, Optional, TypedDict, Union

from .._grammar import Gen, Join
from ..experimental.ast import ContentChunk, ImageBlob, Node
from ..trace import LiteralInput, TextOutput
from .base import ChatState, CompletionState, Model


class OpenAIContentMessage(TypedDict):
    role: str
    content: list[dict]


class OpenAIAudioMessage(TypedDict):
    role: str
    audio: dict


OpenAIMessage = Union[OpenAIContentMessage, OpenAIAudioMessage]


class OpenAIState(ChatState[OpenAIMessage]):
    def __init__(self) -> None:
        super().__init__()
        self.content: list[dict] = []
        self.audio: Optional[dict] = None

    @classmethod
    def from_openai_model(cls, model: str) -> Union["OpenAIState", CompletionState]:
        if model in {
            "gpt-35-turbo-instruct",
            "gpt-3.5-turbo-instruct",
            "babbage-002",
            "davinci-002",
        }:
            return CompletionState()

        if "audio-preview" in model:
            return OpenAIAudioState()
        if model.startswith("gpt-4o") or model.startswith("o1"):
            return OpenAIImageState()
        else:
            return OpenAIState()

    def get_active_message(self) -> Optional[OpenAIMessage]:
        if self.active_role is None:
            return None

        if self.audio:
            if self.content:
                raise ValueError("Expected either content or audio in OpenAI message, not both")
            return OpenAIAudioMessage({"role": self.active_role.role, "audio": self.audio})

        if self.content:
            return OpenAIContentMessage({"role": self.active_role.role, "content": self.content})

        return None

    def reset_active_message(self) -> None:
        super().reset_active_message()
        self.content = []
        self.audio = None

    def apply_text(self, text: str) -> None:
        self.content.append({"type": "text", "text": text})


class OpenAIImageState(OpenAIState):
    def apply_image(self, image: ImageBlob) -> None:
        format = image.image.format
        if format is None:
            raise ValueError(f"Cannot upload image with unknown format: {image.image}")

        with BytesIO() as buffer:
            image.image.save(buffer, format=format)
            b64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")

        mime_type = f"image/{format.lower()}"
        self.content.append(
            {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{b64_image}"}}
        )


class OpenAIAudioState(OpenAIState):
    def __init__(self) -> None:
        raise NotImplementedError("OpenAI audio not yet implemented")


class OpenAI(Model[Union[CompletionState, OpenAIState]]):
    def __init__(
        self,
        model: str,
        echo: bool = True,
        api_key: Optional[str] = None,
        **kwargs,
    ):
        """Build a new OpenAI model object that represents a model in a given state.

        Parameters
        ----------
        model : str
            The name of the OpenAI model to use (e.g. gpt-4o-mini).
        echo : bool
            If true the final result of creating this model state will be displayed (as HTML in a notebook).
        api_key : None or str
            The OpenAI API key to use for remote requests, passed directly to the `openai.OpenAI` constructor.

        **kwargs :
            All extra keyword arguments are passed directly to the `openai.OpenAI` constructor. Commonly used argument
            names include `base_url` and `organization`
        """

        try:
            import openai
        except ImportError:
            raise Exception(
                "Please install the openai package version >= 1 using `pip install openai -U` in order to use guidance.models.OpenAI!"
            )
        self.client = openai.OpenAI(api_key=api_key, **kwargs)
        self.model = model
        super().__init__(echo=echo)

    def initial_state(self) -> Union[CompletionState, OpenAIState]:
        return OpenAIState.from_openai_model(self.model)

    def run(
        self, state: Union[CompletionState, OpenAIState], node: Node
    ) -> Iterable[ContentChunk]:
        if isinstance(node, str):
            yield LiteralInput(value=node)
            return

        elif isinstance(node, ImageBlob):
            yield node
            return

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

            if isinstance(state, CompletionState):
                prompt = state.get_state()["prompt"]
                responses = self.client.completions.create(
                    model=self.model,
                    prompt=prompt,
                    max_tokens=node.max_tokens,
                    temperature=node.temperature,
                    stream=True,
                )
            else:
                oai_state = state.get_state()
                if oai_state["prefill"] is not None:
                    raise ValueError("Prefill not supported for OpenAI")
                if oai_state["active_role"] != "assistant":
                    raise ValueError("Active role must be assistant for OpenAI")
                messages = oai_state["messages"]
                responses = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,  # type: ignore[arg-type]
                    max_tokens=node.max_tokens,
                    temperature=node.temperature,
                    logprobs=True,
                    stream=True,
                )
            for response in responses:
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
            raise ValueError(
                "OpenAI model currently only supports unconstrained generation with `gen()`"
            )
