import base64
from io import BytesIO
from typing import Iterator, Optional

from .._ast import ASTNode

from .._grammar import Gen, Join, RoleEnd, RoleStart
from ..trace import LiteralInput, TextOutput, RoleOpenerInput, RoleCloserInput, ImageOutput
from ._base import Model, State, Client, ContentChunk

class OpenAIState(State):
    def apply_content_chunk(self, chunk: ContentChunk) -> None:
        if self.active_message["role"] is None:
            raise ValueError("OpenAI models require chat blocks (e.g. use `with assistant(): ...`)")
        super().apply_content_chunk(chunk)

    def apply_text(self, text: str) -> None:
        content = self.active_message["data"].setdefault("content", [])
        if len(content) > 0 and content[-1]["type"] == "text":
            # No need to add a new text block; we can be less verbose
            content[-1]["text"] += text
        else:
            content.append({"type": "text", "text": text})
        self.text += text

class OpenAIImageState(OpenAIState):
    def apply_image(self, image: ImageOutput) -> None:
        try:
            import PIL.Image
        except ImportError:
            raise Exception(
                "Please install the Pillow package `pip install Pillow` in order to use images with OpenAI!"
            )

        image_bytes = base64.b64decode(image.value)
        with PIL.Image.open(BytesIO(image_bytes)) as pil_image:
            # Use PIL to infer file format
            # TODO: just store format on ImageOutput type
            format = pil_image.format
            if format is None:
                raise ValueError(f"Cannot upload image with unknown format")

        mime_type = f"image/{format.lower()}"
        content = self.active_message["data"].setdefault("content", [])
        content.append(
            {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{image.value}"}}
        )
        self.text += "<|image|>" # Arbitrary stringification of image


class OpenAIAudioState(OpenAIState):
    def __init__(self) -> None:
        raise NotImplementedError("OpenAI audio not yet implemented")


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

    def run(
        self, state: OpenAIState, node: ASTNode
    ) -> Iterator[ContentChunk]:
        def inner(node):
            if isinstance(node, Join):
                for inner_node in node.values:
                    yield from inner(inner_node)

            elif isinstance(node, str):
                yield LiteralInput(value=node)

            elif isinstance(node, RoleStart):
                # ChatML is as good as anything!
                yield RoleOpenerInput(
                    name=node.role,
                    text="<|im_start|>" + node.role + "\n",
                )

            elif isinstance(node, RoleEnd):
                # ChatML is as good as anything!
                yield RoleCloserInput(
                    name=node.role,
                    text="\n<|im_end|>\n",
                )

            elif isinstance(node, ImageOutput):
                yield node

            elif isinstance(node, Gen):
                if node.capture_name:
                    raise NotImplementedError("Captures not yet supported for OpenAI")
                if node.body_regex != "(?s:.*)":
                    raise ValueError("Body regex not supported for OpenAI")
                if node.stop_regex:
                    raise ValueError("Stop regex not supported for OpenAI")
                if node.save_stop_text:
                    raise ValueError("Save stop text not supported for OpenAI")

                messages = []
                for message in state.messages:
                    if message["role"] is None:
                        # Should never happen?
                        raise ValueError("OpenAI models require chat blocks (e.g. use `with assistant(): ...`)")
                    messages.append(
                        {
                            "role": message["role"],
                            "content": message["data"].get("content", []),
                        }
                    )
                active_message = state.active_message
                if active_message["role"] is None:
                    # Should never happen?
                    raise ValueError("OpenAI models require chat blocks (e.g. use `with assistant(): ...`)")
                if active_message["role"] != "assistant":
                    raise ValueError("OpenAI models can only generate as the assistant (i.e. inside of `with assistant(): ...`)")
                if active_message["data"]:
                    raise ValueError(f"OpenAI models do not support pre-filled assistant messages: got data {active_message['data']}.")
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

        yield from inner(node)

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
            state = OpenAIImageState()
        elif "audio-preview" in model:
            state = OpenAIAudioState()
        else:
            state = OpenAIState()

        super().__init__(
            client = OpenAIClient(model, api_key=api_key, **kwargs),
            state = state,
            echo=echo
        )
