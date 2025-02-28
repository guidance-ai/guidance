import base64
from io import BytesIO
from typing import Any, Iterator, Optional, TypedDict

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
from ..trace import (
    ImageOutput,
    OutputAttr,
    RoleCloserInput,
    RoleOpenerInput,
    TextOutput,
)
from ._base import Client, Model, State


def get_role_start(role: str) -> str:
    # ChatML is as good as anything
    return "<|im_start|>" + role + "\n"


def get_role_end(role: str) -> str:
    # ChatML is as good as anything
    return "\n<|im_end|>\n"


class Message(TypedDict):
    role: str
    content: list[dict[str, Any]]


class OpenAIState(State):
    def __init__(self) -> None:
        super().__init__()
        self.messages: list[Message] = []
        self.content: list[dict[str, Any]] = []

    def apply_text(self, text: str) -> None:
        if len(self.content) > 0 and self.content[-1].get("type") == "text":
            # Reduce verbosity by combining adjacent text nodes
            self.content[-1].setdefault("text", "")
            self.content[-1]["text"] += text
        else:
            self.content.append({"type": "text", "text": text})

    def __str__(self) -> str:
        s = ""
        for message in self.messages:
            s += get_role_start(message["role"])
            for content in message["content"]:
                s += self._fmt_content(content)
            s += get_role_end(message["role"])
        if self.active_role is not None:
            s += get_role_start(self.active_role)
            for content in self.content:
                s += self._fmt_content(content)
        return s

    def _fmt_content(self, content: dict[str, Any]) -> str:
        if content["type"] == "text":
            return content["text"]
        elif content["type"] == "image_url":
            return "[IMAGE]"  # arbitrary stringification
        else:
            raise ValueError(f"Unknown content type: {content['type']}")


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
        # ChatML is as good as anything
        opener_text = "<|im_start|>" + node.role + "\n"
        state.active_role = node.role

        yield RoleOpenerInput(
            name=node.role,
            text=opener_text,
        )

    def role_end(self, state: OpenAIState, node: RoleEnd, **kwargs) -> Iterator[OutputAttr]:
        # ChatML is as good as anything
        closer_text = "\n<|im_end|>\n"
        state.messages.append(
            Message(
                role=node.role,
                content=state.content,
            )
        )
        state.content = []
        state.active_role = None

        yield RoleCloserInput(
            name=node.role,
            text=closer_text,
        )

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
            messages=state.messages,  # type: ignore[arg-type]
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


class OpenAIAudioClient(OpenAIClient):
    # Stub
    pass


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
