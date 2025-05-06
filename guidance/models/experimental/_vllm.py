from typing import Iterator, Optional, TYPE_CHECKING
import wave
import base64
from io import BytesIO
from copy import deepcopy
from pydantic import TypeAdapter
if TYPE_CHECKING:
    from openai.types.chat import ChatCompletionChunk

from ..._ast import GrammarNode, RoleStart, RoleEnd, ASTNode, LiteralNode
from ...trace import OutputAttr, TextOutput
from ...trace._trace import AudioOutput
from .._openai_base import OpenAIState, AssistantAudio, Message, get_role_start
from .._base import Model, Interpreter

class BaseOpenAIInterpreterForVLLM(Interpreter[OpenAIState]):
    log_probs: bool = True

    def __init__(
        self,
        model: str,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs,
    ):
        try:
            import openai
        except ImportError:
            raise Exception(
                "Please install the openai package version >= 1 using `pip install openai -U` in order to use guidance.models.OpenAI!"
            )
        self.state = OpenAIState()
        self.model = model
        self.client = openai.OpenAI(base_url=base_url, api_key=api_key, **kwargs) 

    def role_start(self, node: RoleStart, **kwargs) -> Iterator[OutputAttr]:
        self.state.active_role = node.role
        # TODO: drop this and yield nothing. We need to add this for now as a workaround for the
        # fact that current vis code assumes that there is actually a role start message
        yield TextOutput(value=get_role_start(node.role), is_input=True)

    def role_end(self, node: RoleEnd, **kwargs) -> Iterator[OutputAttr]:
        self.state.messages.append(self.state.get_active_message())
        self.state.audio = None
        self.state.content = []
        self.state.active_role = None
        yield from ()

    def text(self, node: LiteralNode, **kwargs) -> Iterator[OutputAttr]:
        self.state.apply_text(node.value)
        yield TextOutput(value=node.value, is_input=True)

    def run(self, node: ASTNode, **kwargs) -> Iterator[OutputAttr]:
        if not isinstance(node, RoleStart) and self.state.active_role is None:
            raise ValueError(
                "OpenAI models require an active role (e.g. use `with assistant(): ...`)"
            )
        return super().run(node, **kwargs)

    def _run(self, **kwargs) -> Iterator[OutputAttr]:
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

        with self.client.chat.completions.create(
            model=self.model,
            messages=TypeAdapter(list[Message]).dump_python(self.state.messages),  # type: ignore[arg-type]
            logprobs=self.log_probs,
            stream=True,
            **kwargs,
        ) as chunks:
            yield from self._handle_stream(chunks)

    def _handle_stream(
        self, chunks: Iterator["ChatCompletionChunk"]
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
                self.state.apply_text(content)
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

class VLLMInterpreter(BaseOpenAIInterpreterForVLLM):
    def grammar(self, node: GrammarNode, **kwargs) -> Iterator[OutputAttr]:
        buffer: str = ""
        for attr in self._run(
            extra_body = dict(
                guided_decoding_backend="guidance",
                guided_grammar=node.ll_grammar(),
            )
        ):
            if isinstance(attr, TextOutput):
                buffer += attr.value
            yield attr
        matches = node.match(
            buffer,
            raise_exceptions=False,
            # Turn of max_tokens since we don't have access to the tokenizer
            enforce_max_tokens=False,
        )
        if matches is None:
            # TODO: should probably raise...
            # raise ValueError("vLLM failed to constrain the grammar")
            pass
        else:
            for name, value in matches.captures.items():
                log_probs = matches.log_probs[name]
                if isinstance(value, list):
                    assert isinstance(log_probs, list)
                    assert len(value) == len(log_probs)
                    for v, l in zip(value, log_probs):
                        yield self.state.apply_capture(name=name, value=v, log_prob=l, is_append=True)
                else:
                    yield self.state.apply_capture(name=name, value=value, log_prob=log_probs, is_append=False)

class VLLMModel(Model):
    def __init__(self, model: str, echo=True, **kwargs):
        super().__init__(
            interpreter=VLLMInterpreter(model=model, **kwargs),
            echo=echo,
        )
