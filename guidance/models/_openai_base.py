import base64
import wave
from copy import deepcopy
from io import BytesIO
from typing import TYPE_CHECKING, Iterator, Literal, Optional, Union, cast, ContextManager
from abc import ABC, abstractmethod
import json
from pydantic import BaseModel, Discriminator, Field, TypeAdapter
from typing_extensions import Annotated, assert_never

from .._ast import (
    ASTNode,
    GenAudio,
    AudioBlob,
    ImageBlob,
    ImageUrl,
    JsonNode,
    LiteralNode,
    RegexNode,
    RoleEnd,
    RoleStart,
    RuleNode,
    ToolCallNode,
    ToolDefinition,
)
from .._utils import bytes_from
from ..trace import ImageOutput, OutputAttr, TextOutput
from ..trace._trace import AudioOutput
from ._base import Interpreter, State

if TYPE_CHECKING:
    import openai
    from openai.types.chat import ChatCompletionChunk
    from openai.types.chat.chat_completion_chunk import ChoiceLogprobs


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

class Function(BaseModel):
    name: str
    arguments: str


class ToolCall(BaseModel):
    id: str
    type: Literal["function"] = "function"
    function: Function


class ToolCallMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    tool_calls: list[ToolCall]


class ToolCallResult(BaseModel):
    role: Literal["tool"] = "tool"
    tool_call_id: str
    content: str

Message = Union[ContentMessage, AssistantAudioMessage, ToolCallMessage, ToolCallResult]

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

    def get_active_message(self) -> Optional[Message]:
        if self.active_role is None:
            return None
        if self.active_role not in ["system", "user", "assistant"]:
            raise ValueError(f"Invalid active role: {self.active_role}")
        active_role = cast(Literal["system", "user", "assistant"], self.active_role)

        if self.content and self.audio:
            raise ValueError("Cannot have both content and audio")
        if self.audio:
            if active_role != "assistant":
                raise ValueError("Audio messages can only be sent by the assistant")
            return AssistantAudioMessage(
                role=active_role,
                audio=self.audio,
            )
        elif self.content:
            return ContentMessage(
                role=active_role,
                content=self.content,
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
            elif isinstance(message, ToolCallMessage):
                for tool_call in message.tool_calls:
                    s += f"<function={tool_call.function.name}>"
                    s += tool_call.function.arguments
                    s += "</function>"
            elif isinstance(message, ToolCallResult):
                s += f"<function_result={message.tool_call_id}>"
                s += message.content
                s += "</function_result>"
            else:
                if TYPE_CHECKING:
                    assert_never(message)
                raise TypeError(f"Unknown message type: {message}")
            if active_message is None or i != len(messages) - 1:
                # For the sake of consistency, don't add role end for the active message
                s += get_role_end(message.role)
        return s


class BaseOpenAIClientWrapper(ABC):
    @abstractmethod
    def streaming_chat_completions(
        self,
        model: str,
        messages: list[Message],
        log_probs: bool,
        **kwargs,
    ) -> ContextManager[Iterator["ChatCompletionChunk"]]:
        """Streaming chat completions."""
        raise NotImplementedError("This method should be implemented by subclasses.")


class OpenAIClientWrapper(BaseOpenAIClientWrapper):
    def __init__(self, client: "openai.OpenAI"):
        self.client = client

    def streaming_chat_completions(
        self,
        model: str,
        messages: list[Message],
        log_probs: bool,
        **kwargs,
    ) -> ContextManager[Iterator["ChatCompletionChunk"]]:
        """Streaming chat completions."""
        return self.client.chat.completions.create(
            model=model,
            messages=TypeAdapter(list[Message]).dump_python(messages),  # type: ignore[arg-type]
            logprobs=log_probs,
            stream=True,
            **kwargs,
        )

class BaseOpenAIInterpreter(Interpreter[OpenAIState]):
    """Base class for interacting with OpenAI models."""

    log_probs: bool = True

    def __init__(
        self,
        model: str,
        client: BaseOpenAIClientWrapper,
    ):
        self.state = OpenAIState()
        self.model = model
        self.client = client

    def run(self, node: ASTNode, **kwargs) -> Iterator[OutputAttr]:
        if not isinstance(node, RoleStart) and self.state.active_role is None:
            raise ValueError(
                "OpenAI models require an active role (e.g. use `with assistant(): ...`)"
            )
        return super().run(node, **kwargs)

    def role_start(self, node: RoleStart, **kwargs) -> Iterator[OutputAttr]:
        if node.role not in ["system", "user", "assistant"]:
            raise ValueError(
                f"OpenAI models only support roles 'system', 'user', and 'assistant', got {node.role}"
            )
        self.state.active_role = cast(Literal["system", "user", "assistant"], node.role)
        # TODO: drop this and yield nothing. We need to add this for now as a workaround for the
        # fact that current vis code assumes that there is actually a role start message
        yield TextOutput(value=get_role_start(node.role), is_input=True)

    def role_end(self, node: RoleEnd, **kwargs) -> Iterator[OutputAttr]:
        active_message = self.state.get_active_message()
        if active_message is not None:
            self.state.messages.append(active_message)
        self.state.audio = None
        self.state.content = []
        self.state.active_role = None
        yield from ()

    def text(self, node: LiteralNode, **kwargs) -> Iterator[OutputAttr]:
        self.state.apply_text(node.value)
        yield TextOutput(value=node.value, is_input=True)

    def _run(self, tools: Optional[dict[str, ToolDefinition]] = None, **kwargs) -> Iterator[OutputAttr]:
        if self.state.active_role is None:
            # Should never happen?
            raise ValueError(
                "OpenAI models require chat blocks (e.g. use `with assistant(): ...`)"
            )
        if self.state.active_role != "assistant":
            raise ValueError(
                "OpenAI models can only generate as the assistant (i.e. inside of `with assistant(): ...`)"
            )
        if self.state.content:
            raise ValueError(
                f"OpenAI models do not support pre-filled assistant messages: got data {self.state.content}."
            )

        with self.client.streaming_chat_completions(
            messages=self.state.messages,
            log_probs=self.log_probs,
            tools=(
                [
                    {
                        "type": "function",
                        "function": {
                            "name": name,
                            "description": tool.description,
                            "parameters": tool.args.model_json_schema(),
                            "strict": True,
                        },
                    }
                    for name, tool in tools.items()
                ]
                if tools is not None
                else None
            ),
            **kwargs,
        ) as chunks:
            yield from self._handle_stream(chunks, tools)

    def _handle_stream(
        self,
        chunks: Iterator["ChatCompletionChunk"],
        tools: Optional[dict[str, ToolDefinition]],
    ) -> Iterator[OutputAttr]:
        audio: Optional[AssistantAudio] = None
        final_tool_calls: dict[int, ToolCall] = {}
        for chunk in chunks:
            try:
                choice = chunk.choices[0]
            except IndexError:
                # TODO: azure seems to return empty choices sometimes (on first chunk?)
                # Need to make this more robust
                continue
            delta = choice.delta
            if delta.content is not None:
                assert audio is None
                content = delta.content
                if len(content) == 0:
                    continue
                self.state.apply_text(content)
                if (
                    hasattr(choice, "logprobs")
                    and choice.logprobs is not None
                    and choice.logprobs.content is not None
                    and len(choice.logprobs.content) > 0
                ):
                    prob = 2.718 ** choice.logprobs.content[0].logprob
                else:
                    prob = float("nan")
                yield TextOutput(value=delta.content, is_generated=True, prob=prob)
            elif (delta_audio:=cast(Optional[dict], getattr(delta, "audio", None))) is not None:
                transcript_chunk: Optional[str] = None
                if audio is None:
                    assert delta_audio.get("id") is not None
                    audio = AssistantAudio(
                        id=delta_audio["id"],
                        expires_at=delta_audio.get("expires_at", 0),  # ?
                        transcript=delta_audio.get("transcript", ""),
                        data=delta_audio.get("data", ""),
                    )
                    transcript_chunk = delta_audio.get("transcript")
                else:
                    assert delta_audio.get("id") is None or delta_audio["id"] == audio.id
                    if delta_audio.get("data") is not None:
                        audio.data += delta_audio["data"]
                    if delta_audio.get("transcript") is not None:
                        audio.transcript += delta_audio["transcript"]
                        transcript_chunk = delta_audio["transcript"]
                    if delta_audio.get("expires_at") is not None:
                        assert audio.expires_at == 0
                        audio.expires_at = delta_audio["expires_at"]
                if transcript_chunk is not None:
                    # Why not give the users some transcript? :)
                    yield TextOutput(
                        value=delta_audio["transcript"],
                        is_generated=True,
                    )
            elif (tool_calls := delta.tool_calls) is not None:
                for tool_call in tool_calls:
                    index = tool_call.index
                    if index not in final_tool_calls:
                        if final_tool_calls:
                            # Close previous one
                            yield TextOutput(
                                value=f"</function>",
                            )
                        final_tool_calls[index] = ToolCall.model_validate(
                            tool_call, from_attributes=True
                        )
                        yield TextOutput(
                            value=f"<function={tool_call.function.name}>",
                        )
                    else:
                        yield TextOutput(value=tool_call.function.arguments)
                        final_tool_calls[index].function.arguments += tool_call.function.arguments
            elif delta.function_call is not None:
                # Deprecated?
                raise NotImplementedError("Function calling not yet supported for OpenAI")
            elif delta.refusal is not None:
                raise ValueError(f"OpenAI refused the request: {delta.refusal}")

            if choice.finish_reason is not None:
                break

        if audio is not None:
            assert self.state.audio is None
            self.state.audio = audio
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

        if final_tool_calls:
            # Close last one
            yield TextOutput(
                value=f"</function>",
            )
            self.state.messages.append(
                ToolCallMessage(
                    tool_calls=[
                        ToolCall.model_validate(tc, from_attributes=True)
                        for tc in final_tool_calls.values()
                    ]
                )
            )
            for tool_call in final_tool_calls.values():
                tool = tools[tool_call.function.name]
                args = tool.args.model_validate_json(tool_call.function.arguments)
                result = tool.callable(**args.model_dump())
                result_str = json.dumps(result)
                self.state.messages.append(
                    ToolCallResult(
                        tool_call_id=tool_call.id,
                        content=result_str,
                    )
                )
                yield TextOutput(
                    value=f"<function_result={tool_call.function.name}>{result_str}</function_result>",
                )

    def tool_call(self, node: ToolCallNode, **kwargs) -> Iterator[OutputAttr]:
        yield from self._run(
            tools=node.tools,
            tool_choice=node.tool_choice,
            parallel_tool_calls=node.parallel_tool_calls,
            **kwargs,
        )

    def __deepcopy__(self, memo):
        """Custom deepcopy to ensure client is not copied."""
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k == "client":
                # Don't copy the client
                setattr(result, k, v)
            else:
                setattr(result, k, deepcopy(v, memo))
        return result


class OpenAIRuleMixin(BaseOpenAIInterpreter):
    def rule(self, node: RuleNode, **kwargs) -> Iterator[OutputAttr]:
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

        chunks = self.run(node.value, **kwargs)
        if node.capture:
            buffered_text = ""
            for chunk in chunks:
                # TODO: this isinstance check is pretty darn fragile.
                # ~there must be a better way~
                if isinstance(chunk, TextOutput):
                    buffered_text += chunk.value
                yield chunk
            yield self.state.apply_capture(
                name=node.capture,
                value=buffered_text,
                log_prob=1,  # TODO
                is_append=node.list_append,
            )
        else:
            yield from chunks


class OpenAIRegexMixin(BaseOpenAIInterpreter):
    def regex(self, node: RegexNode, **kwargs) -> Iterator[OutputAttr]:
        if node.regex is not None:
            raise ValueError("Regex not yet supported for OpenAI")
        # We're in unconstrained mode now.
        return self._run(**kwargs)


class OpenAIJSONMixin(BaseOpenAIInterpreter):
    def json(self, node: JsonNode, **kwargs) -> Iterator[OutputAttr]:
        return self._run(
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "json_schema",  # TODO?
                    "schema": {k: v for k,v in node.schema.items() if k != "x-guidance"},
                    "strict": True,
                },
            },
            **kwargs,
        )


class OpenAIImageMixin(BaseOpenAIInterpreter):
    def image_blob(self, node: ImageBlob, **kwargs) -> Iterator[OutputAttr]:
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
        self.state.content.append(
            ImageUrlContent(
                type="image_url",
                image_url=ImageUrlContentInner(url=f"data:{mime_type};base64,{node.data}"),
            )
        )
        yield ImageOutput(value=node.data, is_input=True)

    def image_url(self, node: ImageUrl, **kwargs) -> Iterator[OutputAttr]:
        self.state.content.append(
            ImageUrlContent(
                type="image_url",
                image_url=ImageUrlContentInner(url=node.url)
            )
        )
        image_bytes = bytes_from(node.url, allow_local=False)
        base64_string = base64.b64encode(image_bytes).decode("utf-8")
        yield ImageOutput(value=base64_string, is_input=True)


class OpenAIAudioMixin(BaseOpenAIInterpreter):
    # Audio models don't support logprobs
    log_probs: bool = False

    def audio_blob(self, node: AudioBlob, **kwargs) -> Iterator[OutputAttr]:
        format = "wav"  # TODO: infer from node
        self.state.content.append(
            AudioContent(
                type="input_audio",
                input_audio=InputAudio(
                    data=node.data,
                    format=format,
                ),
            )
        )
        yield AudioOutput(value=node.data, format=format, is_input=True)

    def gen_audio(self, node: GenAudio, **kwargs) -> Iterator[OutputAttr]:
        yield from self._run(
            modalities=["text", "audio"],  # Has to be both?
            audio={
                "voice": node.kwargs.get("voice", "alloy"),
                "format": "pcm16",  # Has to be pcm16 for streaming
            },
        )
