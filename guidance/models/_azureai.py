from typing import Callable, Iterator, Optional, Union

from pydantic import TypeAdapter

from .._ast import (
    JsonNode,
    RuleNode,
)
from ._base import Model
from ._openai_base import (
    BaseOpenAIInterpreter,
    OpenAIAudioMixin,
    OpenAIImageMixin,
    Message,
)
from ..trace import OutputAttr


class AzureOpenAIInterpreter(BaseOpenAIInterpreter):
    def __init__(
        self,
        *,
        model_name: str,
        azure_deployment: str,
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
        super().__init__(model_name, client)


def create_azure_openai_model(
    model_name: str,
    azure_deployment: str,
    echo: bool = True,
    *,
    api_version: Optional[str] = None,
    api_key: Optional[str] = None,
    azure_ad_token: Optional[str] = None,
    azure_ad_token_provider: Optional[Callable[[], str]] = None,
    organization: Optional[str] = None,
    **kwargs,
) -> Model:
    """Create a Model capable of interacting with an Azure AI OpenAI deployment

    Parameters
    ----------
    model_name : str
        The name of the Azure OpenAI model to use (e.g. gpt-4o-mini).
    azure_deployment : str
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
    if "audio-preview" in model_name:
        interpreter_cls = type(
            "AzureOpenAIAudioInterpreter", (AzureOpenAIInterpreter, OpenAIAudioMixin), {}
        )
    elif model_name.startswith("gpt-4o") or model_name.startswith("o1"):
        interpreter_cls = type(
            "AzureOpenAIImageInterpreter", (AzureOpenAIInterpreter, OpenAIImageMixin), {}
        )
    else:
        interpreter_cls = AzureOpenAIInterpreter

    interpreter = interpreter_cls(
        model_name=model_name,
        azure_deployment=azure_deployment,
        api_version=api_version,
        api_key=api_key,
        azure_ad_token=azure_ad_token,
        azure_ad_token_provider=azure_ad_token_provider,
        organization=organization,
        **kwargs,
    )

    model = Model(interpreter=interpreter, echo=echo)

    return model


class AzureInferenceInterpreter(BaseOpenAIInterpreter):
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
        client = azure.ai.inference.ChatCompletionsClient(
            endpoint=endpoint,
            credential=credential,
        )
        super().__init__(client=client)

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
