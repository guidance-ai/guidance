from typing import TYPE_CHECKING, Callable, Iterator, Optional, Union

from pydantic import TypeAdapter


from .._ast import (
    JsonNode,
    RuleNode,
)
from ..trace import OutputAttr
from ._base import Model

from ._openai_base import (
    BaseOpenAIInterpreter,
    AudioContent,
    OpenAIState,
    Message,
    OpenAIImageMixin,
    OpenAIAudioMixin,
)

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
            interpreter_cls = type(
                "OpenAIAudioInterpreter", (OpenAIInterpreter, OpenAIAudioMixin), {}
            )
        elif model.startswith("gpt-4o") or model.startswith("o1"):
            interpreter_cls = type(
                "OpenAIImageInterpreter", (OpenAIInterpreter, OpenAIImageMixin), {}
            )
        else:
            interpreter_cls = OpenAIInterpreter

        super().__init__(interpreter=interpreter_cls(model, api_key=api_key, **kwargs), echo=echo)



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
