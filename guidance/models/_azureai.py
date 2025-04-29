import logging

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

logger = logging.getLogger(__name__)


class AzureOpenAIInterpreter(BaseOpenAIInterpreter):
    def __init__(
        self,
        *,
        azure_endpoint: str,
        azure_deployment: str,
        model_name: str,
        api_version: Optional[str] = None,
        api_key: Optional[str] = None,
        azure_ad_token: Optional[str] = None,
        azure_ad_token_provider: Optional[Callable[[], str]] = None,
        **kwargs,
    ):
        try:
            import openai
        except ImportError:
            raise Exception(
                "Please install the openai package version >= 1 using `pip install openai -U` in order to use guidance.models.OpenAI!"
            )
        client = openai.AzureOpenAI(
            azure_endpoint=azure_endpoint,
            azure_deployment=azure_deployment,
            api_version=api_version,
            api_key=api_key,
            azure_ad_token=azure_ad_token,
            azure_ad_token_provider=azure_ad_token_provider,
            **kwargs,
        )
        super().__init__(model_name, client)


def create_azure_openai_model(
    azure_endpoint: str,
    azure_deployment: str,
    echo: bool = True,
    *,
    model_name: Optional[str] = None,
    api_version: Optional[str] = None,
    api_key: Optional[str] = None,
    azure_ad_token: Optional[str] = None,
    azure_ad_token_provider: Optional[Callable[[], str]] = None,
    has_audio_support: bool = False,
    has_image_support: bool = False,
    **kwargs,
) -> Model:
    """Create a Model capable of interacting with an Azure AI OpenAI deployment

    Parameters
    ----------
    azure_deployment : str
        The Azure deployment name to use for the model. The Azure AI portal will
        default this to being the model_name, but it can be different
    model_name : str
        The name of the Azure OpenAI model to use (e.g. gpt-4o-mini).
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
    if has_audio_support and has_image_support:
        raise ValueError(f"No known models have both audio and image support")

    if (model_name and "audio-preview" in model_name) or has_audio_support:
        interpreter_cls = type(
            "AzureOpenAIAudioInterpreter", (AzureOpenAIInterpreter, OpenAIAudioMixin), {}
        )
    elif (
        model_name and (model_name.startswith("gpt-4o") or model_name.startswith("o1"))
    ) or has_image_support:
        interpreter_cls = type(
            "AzureOpenAIImageInterpreter", (AzureOpenAIInterpreter, OpenAIImageMixin), {}
        )
    else:
        interpreter_cls = AzureOpenAIInterpreter

    interpreter = interpreter_cls(
        azure_endpoint=azure_endpoint,
        model_name=model_name,
        azure_deployment=azure_deployment,
        api_version=api_version,
        api_key=api_key,
        azure_ad_token=azure_ad_token,
        azure_ad_token_provider=azure_ad_token_provider,
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
        model_name: str,
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
        super().__init__(client=client, model=model_name)

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
                "model": self.model,
                **kwargs,
            },
            headers={
                "extra-parameters": "pass-through",
            },
        ) as chunks:
            yield from self._handle_stream(chunks)

    #    def rule(self, node: RuleNode, **kwargs) -> Iterator[OutputAttr]:
    #        raise ValueError("Rule nodes are not supported for Azure Inference")

    def json(self, node: JsonNode, **kwargs) -> Iterator[OutputAttr]:
        return self._run(
            json_schema={
                "name": "json_schema",  # TODO?
                "schema": node.schema,
                "strict": True,
            },
            **kwargs,
        )


def create_azure_aifoundry_model(
    azure_endpoint: str,
    echo: bool = True,
    *,
    model_name: Optional[str] = None,
    api_key: Optional[str] = None,
    token_credential: Optional["TokenCredential"] = None,
    has_audio_support: bool = False,
    has_image_support: bool = False,
) -> Model:
    if has_audio_support:
        raise NotImplementedError("No support yet for Audio in Azure AI Foundry")
    if has_image_support:
        raise NotImplementedError("No support yet for Images in Azure AI Foundry")

    try:
        from azure.core.credentials import AzureKeyCredential, TokenCredential
    except ImportError:
        raise Exception(
            "Please install the azure-core package using `pip install -U azure-core` in order to use guidance.models.AzureAI!"
        )

    credential: AzureKeyCredential | TokenCredential | None = None
    if api_key and token_credential:
        raise ValueError("Specify either api_key or token_credential")
    elif api_key:
        credential = AzureKeyCredential(api_key)
    elif token_credential:
        credential = token_credential
    else:
        raise ValueError("Must specify either api_key or token_credential")

    interpreter = AzureInferenceInterpreter(
        endpoint=azure_endpoint,
        credential=credential,
        model_name=model_name,
    )

    result = Model(interpreter=interpreter, echo=echo)
    return result


def create_azure_model(
    is_openai: bool,
    azure_endpoint: str,
    azure_deployment: str | None = None,
    model_name: Optional[str] = None,
    api_key: Optional[str] = None,
    api_version: Optional[str] = None,
    azure_ad_token: Optional[str] = None,
    azure_credential=None,
    echo: bool = True,
    has_audio_support: bool = False,
    has_image_support: bool = False,
    **kwargs,
) -> Model:
    # Dispatches to appropriate factory above...
    if is_openai:
        if azure_credential:
            from azure.identity import get_bearer_token_provider

            azure_ad_token_provider = get_bearer_token_provider(
                azure_credential, "https://cognitiveservices.azure.com/.default"
            )
        else:
            azure_ad_token_provider = None
        result = create_azure_openai_model(
            azure_endpoint=azure_endpoint,
            azure_deployment=azure_deployment,
            echo=echo,
            model_name=model_name,
            api_version=api_version,
            api_key=api_key,
            azure_ad_token_provider=azure_ad_token_provider,
            azure_ad_token=azure_ad_token,
            has_audio_support=has_audio_support,
            has_image_support=has_image_support,
            **kwargs,
        )
    else:
        if azure_deployment:
            logger.info("Ignoring azure_deployment")
        if api_version:
            logger.info("Ignoring api_version")
        if azure_ad_token:
            logger.info("Ignoring azure_ad_token")
        if kwargs:
            logger.info("Ignoring kwargs")
        result = create_azure_aifoundry_model(
            azure_endpoint=azure_endpoint,
            echo=echo,
            model_name=model_name,
            api_key=api_key,
            token_credential=azure_credential,
            has_audio_support=has_audio_support,
            has_image_support=has_image_support,
        )

    return result
