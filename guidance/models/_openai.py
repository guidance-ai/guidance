import base64
import wave
from io import BytesIO
from copy import deepcopy
from typing import TYPE_CHECKING, Callable, Iterator, Optional, Union

from pydantic import TypeAdapter


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
from ._base import Interpreter, Model, State

from ._openai_base import BaseOpenAIInterpreter, AudioContent, OpenAIState, Message

if TYPE_CHECKING:
    import openai
    from openai.types.chat import ChatCompletionChunk
    from azure.core.credentials import AzureKeyCredential, TokenCredential


class OpenAIInterpreter(BaseOpenAIInterpreter):
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
        client = openai.OpenAI(api_key, **kwargs)
        super().__init__(model=model, client=client)


class OpenAIImageMixin:
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
            {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{node.data}"}}
        )
        yield ImageOutput(value=node.data, input=True)

    def image_url(self, node: ImageUrl, **kwargs) -> Iterator[OutputAttr]:
        self.state.content.append({"type": "image_url", "image_url": {"url": node.url}})
        image_bytes = bytes_from(node.url, allow_local=False)
        base64_string = base64.b64encode(image_bytes).decode("utf-8")
        yield ImageOutput(value=base64_string, input=True)


class OpenAIAudioInterpreter(OpenAIInterpreter):
    log_probs: bool = False

    def audio_blob(self, node: ImageBlob, **kwargs) -> Iterator[OutputAttr]:
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
        yield AudioOutput(value=node.data, format=format, input=True)

    def gen_audio(self, node: GenAudio, **kwargs) -> Iterator[OutputAttr]:
        yield from self._run(
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
        *,
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
            interpreter_cls = OpenAIAudioInterpreter
        elif model.startswith("gpt-4o") or model.startswith("o1"):
            interpreter_cls = type(
                "OpenAIImageInterpreter", (OpenAIInterpreter, OpenAIImageMixin), {}
            )
        else:
            interpreter_cls = OpenAIInterpreter

        super().__init__(interpreter=interpreter_cls(model, api_key=api_key, **kwargs), echo=echo)


class AzureOpenAIInterpreter(BaseOpenAIInterpreter):
    def __init__(
        self,
        *,
        model: str,
        azure_deployment: Optional[str] = None,
        api_version: Optional[str] = None,
        api_key: Optional[str] = None,
        azure_ad_token: Optional[str] = None,
        azure_ad_token_provider: Optional[Callable[[], str]] = None,
        organization: Optional[str] = None,
        **kwargs,
    ):
        try:
            import openai
        except ImportError:
            raise Exception(
                "Please install the openai package version >= 1 using `pip install openai -U` in order to use guidance.models.OpenAI!"
            )
        client = openai.AzureOpenAI(
            azure_deployment=azure_deployment,
            api_version=api_version,
            api_key=api_key,
            azure_ad_token=azure_ad_token,
            azure_ad_token_provider=azure_ad_token_provider,
            organization=organization,
            **kwargs,
        )
        super().__init__(model, client)


class AzureOpenAI(Model):
    def __init__(
        self,
        model: str,
        echo: bool = True,
        *,
        azure_deployment: Optional[str] = None,
        api_version: Optional[str] = None,
        api_key: Optional[str] = None,
        azure_ad_token: Optional[str] = None,
        azure_ad_token_provider: Optional[Callable[[], str]] = None,
        organization: Optional[str] = None,
        **kwargs,
    ):
        """Build a new Azure OpenAI model object that represents a model in a given state.

        Parameters
        ----------
        model : str
            The name of the Azure OpenAI model to use (e.g. gpt-4o-mini).
        azure_deployment : str | None
            The Azure deployment name to use for the model.
        api_version : str | None
            The API version to use for the Azure OpenAI service.
        api_key : str | None
            The API key to use for the Azure OpenAI service.
        azure_ad_token : str | None
            The Azure AD token to use for authentication.
        azure_ad_token_provider : Callable[[], str] | None
            A callable that returns an Azure AD token for authentication.
        organization : str | None
            The organization ID to use for the Azure OpenAI service.
        echo : bool
            If true the final result of creating this model state will be displayed (as HTML in a notebook).
        **kwargs :
            All extra keyword arguments are passed directly to the `openai.OpenAI` constructor. Commonly used argument
            names include `base_url` and `organization`
        """
        if "audio-preview" in model:
            interpreter_cls = type(
                "AzureOpenAIAudioInterpreter", (AzureOpenAIInterpreter, OpenAIAudioInterpreter), {}
            )
        elif model.startswith("gpt-4o") or model.startswith("o1"):
            interpreter_cls = type(
                "AzureOpenAIImageInterpreter", (AzureOpenAIInterpreter, OpenAIImageInterpreter), {}
            )
        else:
            interpreter_cls = AzureOpenAIInterpreter

        super().__init__(
            interpreter=interpreter_cls(
                model=model,
                azure_deployment=azure_deployment,
                api_version=api_version,
                api_key=api_key,
                azure_ad_token=azure_ad_token,
                azure_ad_token_provider=azure_ad_token_provider,
                organization=organization,
                **kwargs,
            ),
            echo=echo,
        )


class AzureInferenceInterpreter(OpenAIInterpreter):
    def __init__(
        self,
        *,
        endpoint: str,
        credential: Union["AzureKeyCredential", "TokenCredential"],
    ):
        try:
            import azure.ai.inference
        except ImportError:
            raise Exception(
                "Please install the azure-ai-inference package  using `pip install azure-ai-inference` in order to use guidance.models.AzureInference!"
            )
        self.state = OpenAIState()
        self.client = azure.ai.inference.ChatCompletionsClient(
            endpoint=endpoint,
            credential=credential,
        )

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

        with self.client.complete(
            body={
                "messages": TypeAdapter(list[Message]).dump_python(self.state.messages),
                "log_probs": self.log_probs,
                "stream": True,
                **kwargs,
            },
            headers={
                "extra-parameters": "pass-through",
            },
        ) as chunks:
            yield from self._handle_stream(chunks)

    def rule(self, node: RuleNode, **kwargs) -> Iterator[OutputAttr]:
        raise ValueError("Rule nodes are not supported for Azure Inference")

    def json(self, node: JsonNode, **kwargs) -> Iterator[OutputAttr]:
        return self._run(
            json_schema={
                "name": "json_schema",  # TODO?
                "schema": node.schema,
                "strict": True,
            },
            **kwargs,
        )


class AzureInference(Model):
    def __init__(
        self,
        *,
        endpoint: str,
        credential: Union["AzureKeyCredential", "TokenCredential"],
        echo: bool = True,
    ):
        """Build a new Azure Inference model object that represents a model in a given state.

        Parameters
        ----------
        endpoint : str
            The endpoint of the Azure Inference service.
        credential : AzureKeyCredential | TokenCredential
            The credential to use for authentication with the Azure Inference service.
        echo : bool
            If true the final result of creating this model state will be displayed (as HTML in a notebook).
        """
        super().__init__(
            interpreter=AzureInferenceInterpreter(
                endpoint=endpoint,
                credential=credential,
            ),
            echo=echo,
        )
