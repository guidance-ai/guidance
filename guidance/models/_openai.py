import base64
from io import BytesIO
from typing import TYPE_CHECKING, Iterator, Literal, Optional, Union

from pydantic import BaseModel, Discriminator, Field, TypeAdapter
from pydantic.types import Base64Str
from typing_extensions import Annotated, assert_never

from .._ast import (
    ASTNode,
    ImageBlob,
    ImageUrl,
    JsonNode,
    LiteralNode,
    RegexNode,
    RoleEnd,
    RoleStart,
    RuleNode,
)
from .._utils import bytes_from
from ..trace import ImageOutput, OutputAttr, TextOutput
from ._base import Client, Model, State


def get_role_start(role: str) -> str:
    # ChatML is as good as anything
    return "<|im_start|>" + role + "\n"


def get_role_end(role: str) -> str:
    # ChatML is as good as anything
    return "\n<|im_end|>\n"


class AssistantAudio(BaseModel):
    id: str
    expires_at: int = Field(exclude=True)
    data: Base64Str = Field(exclude=True)
    transcript: str = Field(exclude=True)


class AssistantAudioMessage(BaseModel):
    role: Literal["assistant"]
    audio: AssistantAudio


class TextContent(BaseModel):
    type: Literal["text"]
    text: str


class InputAudio(BaseModel):
    data: Base64Str
    format: str


class AudioContent(BaseModel):
    type: Literal["input_audio"]
    audio: InputAudio


class ImageUrlContentInner(BaseModel):
    url: str


class ImageUrlContent(BaseModel):
    type: Literal["image_url"]
    image_url: ImageUrlContentInner


Content = Annotated[Union[TextContent, AudioContent, ImageUrlContent], Discriminator("type")]


class ContentMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: list[Content]


Message = Union[ContentMessage, AssistantAudioMessage]


class OpenAIState(State):
    def __init__(self) -> None:
        super().__init__()
        self.messages: list[Message] = []
        self.content: list[Content] = []
        self.audio: Optional[AssistantAudio] = None

    def apply_text(self, text: str) -> None:
        if len(self.content) > 0 and isinstance(self.content[-1], TextContent):
            self.content[-1].text += text
        else:
            self.content.append(TextContent(type="text", text=text))

    def get_active_message(state) -> Optional[Message]:
        if state.active_role is None:
            return None
        if state.content and state.audio:
            raise ValueError("Cannot have both content and audio")
        if state.audio:
            return AssistantAudioMessage(
                role=state.active_role,
                audio=state.audio,
            )
        elif state.content:
            return ContentMessage(
                role=state.active_role,
                content=state.content,
            )
        else:
            return None

    def __str__(self) -> str:
        messages = self.messages
        active_message = self.get_active_message()
        if active_message is not None:
            messages = messages + [active_message]
        s = ""
        for i, message in enumerate(messages):
            s += get_role_start(message.role)
            if isinstance(message, AssistantAudioMessage):
                s += "[AUDIO]"
            elif isinstance(message, ContentMessage):
                for content in message.content:
                    if isinstance(content, TextContent):
                        s += content.text
                    elif isinstance(content, ImageUrlContent):
                        s += "[IMAGE]"  # Arbitrary stringification
                    elif isinstance(content, AudioContent):
                        s += "[AUDIO]"  # transcript?
                    else:
                        if TYPE_CHECKING:
                            assert_never(content)
                        raise TypeError(f"Unknown content type: {content}")
            else:
                if TYPE_CHECKING:
                    assert_never(message)
                raise TypeError(f"Unknown message type: {message}")
            if active_message is None or i != len(messages) - 1:
                # For the sake of consistency, don't add role end for the active message
                s += get_role_end(message.role)
        return s


class OpenAIClient(Client[OpenAIState]):
    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        **kwargs,
    ):
        try:
            import openai
        except ImportError:
            raise Exception(
                "Please install the openai package version >= 1 using `pip install openai -U` in order to use guidance.models.OpenAI!"
            )
        self.model = model
        self.client = openai.OpenAI(api_key=api_key, **kwargs)

    def run(self, state: OpenAIState, node: ASTNode, **kwargs) -> Iterator[OutputAttr]:
        if not isinstance(node, RoleStart) and state.active_role is None:
            raise ValueError(
                "OpenAI models require an active role (e.g. use `with assistant(): ...`)"
            )
        return super().run(state, node, **kwargs)

    def role_start(self, state: OpenAIState, node: RoleStart, **kwargs) -> Iterator[OutputAttr]:
        state.active_role = node.role
        # TODO: drop this and yield nothing. We need to add this for now as a workaround for the
        # fact that current vis code assumes that there is actually a role start message
        yield TextOutput(value=get_role_start(node.role), is_input=True)

    def role_end(self, state: OpenAIState, node: RoleEnd, **kwargs) -> Iterator[OutputAttr]:
        state.messages.append(state.get_active_message())
        state.audio = None
        state.content = []
        state.active_role = None
        yield from ()

    def text(self, state: OpenAIState, node: LiteralNode, **kwargs) -> Iterator[OutputAttr]:
        state.apply_text(node.value)
        yield TextOutput(value=node.value, input=True)

    def rule(self, state: OpenAIState, node: RuleNode, **kwargs) -> Iterator[OutputAttr]:
        if node.stop:
            raise ValueError("Stop condition not yet supported for OpenAI")
        if node.suffix:
            raise ValueError("Suffix not yet supported for OpenAI")
        if node.stop_capture:
            raise ValueError("Save stop text not yet supported for OpenAI")

        kwargs = kwargs.copy()
        if node.temperature:
            kwargs["temperature"] = node.temperature
        if node.max_tokens:
            kwargs["max_tokens"] = node.max_tokens

        chunks = self.run(state, node.value, **kwargs)
        if node.capture:
            buffered_text = ""
            for chunk in chunks:
                # TODO: this isinstance check is pretty darn fragile.
                # ~there must be a better way~
                if isinstance(chunk, TextOutput):
                    buffered_text += chunk.value
                yield chunk
            yield state.apply_capture(
                name=node.capture,
                value=buffered_text,
                log_prob=1,  # TODO
                is_append=node.list_append,
            )
        else:
            return chunks

    def regex(self, state: OpenAIState, node: RegexNode, **kwargs) -> Iterator[OutputAttr]:
        if node.regex is not None:
            raise ValueError("Regex not yet supported for OpenAI")
        # We're in unconstrained mode now.
        return self._run(state, **kwargs)

    def json(self, state: OpenAIState, node: JsonNode, **kwargs) -> Iterator[OutputAttr]:
        return self._run(
            state,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "json_schema",  # TODO?
                    "schema": node.schema,
                    "strict": True,
                },
            },
            **kwargs,
        )

    def _run(self, state: OpenAIState, **kwargs) -> Iterator[OutputAttr]:
        if state.active_role is None:
            # Should never happen?
            raise ValueError(
                "OpenAI models require chat blocks (e.g. use `with assistant(): ...`)"
            )
        if state.active_role != "assistant":
            raise ValueError(
                "OpenAI models can only generate as the assistant (i.e. inside of `with assistant(): ...`)"
            )
        if state.content:
            raise ValueError(
                f"OpenAI models do not support pre-filled assistant messages: got data {state.content}."
            )

        responses = self.client.chat.completions.create(
            model=self.model,
            messages=TypeAdapter(list[Message]).dump_python(state.messages),  # type: ignore[arg-type]
            logprobs=True,
            stream=True,
            **kwargs,
        )
        for response in responses:
            choice = response.choices[0]
            delta = choice.delta
            if delta.content is not None:
                content = delta.content
                if len(content) == 0:
                    continue
                state.apply_text(content)
                yield TextOutput(
                    value=delta.content,
                    is_generated=True,
                    # TODO: actually get tokens from this and be less lazy
                    prob=2.718 ** choice.logprobs.content[0].logprob,  # type: ignore[union-attr,index]
                )
            elif choice.finish_reason is not None:
                # TODO: handle finish_reason elegantly
                break
            else:
                NotImplementedError(f"Unknown delta: {delta}")


class OpenAIImageClient(OpenAIClient):
    def image_blob(self, state: OpenAIState, node: ImageBlob, **kwargs) -> Iterator[OutputAttr]:
        try:
            import PIL.Image
        except ImportError:
            raise Exception(
                "Please install the Pillow package `pip install Pillow` in order to use images with OpenAI!"
            )

        image_bytes = base64.b64decode(node.data)
        with PIL.Image.open(BytesIO(image_bytes)) as pil_image:
            # Use PIL to infer file format
            # TODO: just store format on ImageOutput type
            format = pil_image.format
            if format is None:
                raise ValueError(f"Cannot upload image with unknown format")

        mime_type = f"image/{format.lower()}"
        return self.image_url(state, ImageUrl(url=f"data:{mime_type};base64,{node.data}"))

    def image_url(self, state: OpenAIState, node: ImageUrl, **kwargs) -> Iterator[OutputAttr]:
        state.content.append({"type": "image_url", "image_url": {"url": node.url}})
        image_bytes = bytes_from(node.url, allow_local=False)
        base64_string = base64.b64encode(image_bytes).decode("utf-8")
        yield ImageOutput(value=base64_string, input=True)


class OpenAI(Model):
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

        if model.startswith("gpt-4o") or model.startswith("o1"):
            client_cls = OpenAIImageClient
        elif "audio-preview" in model:
            client_cls = OpenAIAudioClient
        else:
            client_cls = OpenAIClient

        super().__init__(
            client=client_cls(model, api_key=api_key, **kwargs), state=OpenAIState(), echo=echo
        )
