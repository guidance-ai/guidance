import base64
import wave
from io import BytesIO
from typing import TYPE_CHECKING, Iterator, Literal, Optional, Union

from pydantic import BaseModel, Discriminator, Field, TypeAdapter
from typing_extensions import Annotated, assert_never

from .._ast import (
    ASTNode,
    GenAudio,
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
from ..trace._trace import AudioOutput
from ._base import Client, Model, State

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletionChunk


def get_role_start(role: str) -> str:
    # ChatML is as good as anything
    return "<|im_start|>" + role + "\n"


def get_role_end(role: str) -> str:
    # ChatML is as good as anything
    return "\n<|im_end|>\n"


class AssistantAudio(BaseModel):
    id: str
    expires_at: int = Field(exclude=True)
    data: str = Field(exclude=True)
    transcript: str = Field(exclude=True)


class AssistantAudioMessage(BaseModel):
    role: Literal["assistant"]
    audio: AssistantAudio


class TextContent(BaseModel):
    type: Literal["text"]
    text: str


class InputAudio(BaseModel):
    data: str
    format: str


class AudioContent(BaseModel):
    type: Literal["input_audio"]
    input_audio: InputAudio


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
    log_probs: bool = True

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
        yield TextOutput(value=node.value, is_input=True)

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
            yield from chunks

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

        with self.client.chat.completions.create(
            model=self.model,
            messages=TypeAdapter(list[Message]).dump_python(state.messages),  # type: ignore[arg-type]
            logprobs=self.log_probs,
            stream=True,
            **kwargs,
        ) as chunks:
            yield from self._handle_stream(state, chunks)

    def _handle_stream(
        self, state: OpenAIState, chunks: Iterator["ChatCompletionChunk"]
    ) -> Iterator[OutputAttr]:
        audio: Optional[AssistantAudio] = None
        for chunk in chunks:
            choice = chunk.choices[0]
            delta = choice.delta
            if delta.content is not None:
                assert audio is None
                content = delta.content
                if len(content) == 0:
                    continue
                state.apply_text(content)
                if choice.logprobs is not None:
                    # TODO: actually get tokens from this and be less lazy
                    prob = 2.718 ** choice.logprobs.content[0].logprob
                else:
                    prob = float("nan")
                yield TextOutput(value=delta.content, is_generated=True, prob=prob)
            elif getattr(delta, "audio", None) is not None:
                transcript_chunk: Optional[str] = None
                if audio is None:
                    assert delta.audio.get("id") is not None
                    audio = AssistantAudio(
                        id=delta.audio["id"],
                        expires_at=delta.audio.get("expires_at", 0),  # ?
                        transcript=delta.audio.get("transcript", ""),
                        data=delta.audio.get("data", ""),
                    )
                    transcript_chunk = delta.audio.get("transcript")
                else:
                    assert delta.audio.get("id") is None or delta.audio["id"] == audio.id
                    if delta.audio.get("data") is not None:
                        audio.data += delta.audio["data"]
                    if delta.audio.get("transcript") is not None:
                        audio.transcript += delta.audio["transcript"]
                        transcript_chunk = delta.audio["transcript"]
                    if delta.audio.get("expires_at") is not None:
                        assert audio.expires_at == 0
                        audio.expires_at = delta.audio["expires_at"]
                if transcript_chunk is not None:
                    # Why not give the users some transcript? :)
                    yield TextOutput(
                        value=delta.audio["transcript"],
                        is_generated=True,
                    )
            elif delta.function_call is not None:
                raise NotImplementedError("Function calling not yet supported for OpenAI")
            elif delta.tool_calls is not None:
                raise NotImplementedError("Tool calling not yet supported for OpenAI")
            elif delta.refusal is not None:
                raise ValueError(f"OpenAI refused the request: {delta.refusal}")

            if choice.finish_reason is not None:
                break

        if audio is not None:
            assert state.audio is None
            state.audio = audio
            # Create an in-memory WAV file
            wav_buffer = BytesIO()
            with wave.open(wav_buffer, "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)  # PCM16 = 2 bytes per sample
                wav_file.setframerate(22050)  # A guess
                wav_file.writeframes(base64.b64decode(audio.data))

            # Get WAV bytes
            wav_bytes = wav_buffer.getvalue()
            yield AudioOutput(value=base64.b64encode(wav_bytes).decode(), is_input=False)


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
        state.content.append(
            {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{node.data}"}}
        )
        yield ImageOutput(value=node.data, input=True)

    def image_url(self, state: OpenAIState, node: ImageUrl, **kwargs) -> Iterator[OutputAttr]:
        state.content.append({"type": "image_url", "image_url": {"url": node.url}})
        image_bytes = bytes_from(node.url, allow_local=False)
        base64_string = base64.b64encode(image_bytes).decode("utf-8")
        yield ImageOutput(value=base64_string, input=True)


class OpenAIAudioClient(OpenAIClient):
    log_probs: bool = False

    def audio_blob(self, state: OpenAIState, node: ImageBlob, **kwargs) -> Iterator[OutputAttr]:
        format = "wav"  # TODO: infer from node
        state.content.append(
            AudioContent(
                type="input_audio",
                input_audio=InputAudio(
                    data=node.data,
                    format=format,
                ),
            )
        )
        yield AudioOutput(value=node.data, format=format, input=True)

    def gen_audio(self, state: OpenAIState, node: GenAudio, **kwargs) -> Iterator[OutputAttr]:
        yield from self._run(
            state,
            modalities=["text", "audio"],  # Has to be both?
            audio={
                "voice": node.kwargs.get("voice", "alloy"),
                "format": "pcm16",  # Has to be pcm16 for streaming
            },
        )


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

        if "audio-preview" in model:
            client_cls = OpenAIAudioClient
        elif model.startswith("gpt-4o") or model.startswith("o1"):
            client_cls = OpenAIImageClient
        else:
            client_cls = OpenAIClient

        super().__init__(
            client=client_cls(model, api_key=api_key, **kwargs), state=OpenAIState(), echo=echo
        )
