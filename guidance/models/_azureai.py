import logging
from typing import TYPE_CHECKING, Any, Callable, ContextManager, Iterator, Optional, Union, cast

from guidance._schema import SamplingParams

from .._ast import (
    JsonNode,
)
from ..trace import OutputAttr
from ._base import Model
from ._openai_base import (
    BaseOpenAIClientWrapper,
    BaseOpenAIInterpreter,
    OpenAIAudioMixin,
    OpenAIClientWrapper,
    OpenAIImageMixin,
    OpenAIJSONMixin,
    OpenAIRegexMixin,
    OpenAIRuleMixin,
)

if TYPE_CHECKING:
    import azure.ai.inference
    from azure.core.credentials import AzureKeyCredential, TokenCredential
    from openai.types.chat import ChatCompletionChunk

logger = logging.getLogger(__name__)


class AzureOpenAIInterpreter(OpenAIRuleMixin, OpenAIJSONMixin, OpenAIRegexMixin, BaseOpenAIInterpreter):
    """A basic class for interacting with Azure OpenAI."""

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
        except ImportError as ie:
            raise Exception(
                "Please install the openai package version >= 1 using `pip install openai -U` in order to use guidance.models.OpenAI!"
            ) from ie
        client = openai.AzureOpenAI(
            azure_endpoint=azure_endpoint,
            azure_deployment=azure_deployment,
            api_version=api_version,
            api_key=api_key,
            azure_ad_token=azure_ad_token,
            azure_ad_token_provider=azure_ad_token_provider,
            **kwargs,
        )
        super().__init__(model=model_name, client=OpenAIClientWrapper(client))


class AzureOpenAIAudioInterpreter(OpenAIAudioMixin, AzureOpenAIInterpreter):
    """Class to add audio capabilities to an Azure OpenAI model"""

    pass


class AzureOpenAIImageInterpreter(OpenAIImageMixin, AzureOpenAIInterpreter):
    """Class to add image capabilities to an Azure OpenAI model"""

    pass


def create_azure_openai_model(
    azure_endpoint: str,
    azure_deployment: str,
    echo: bool = True,
    *,
    model_name: str,
    api_version: Optional[str] = None,
    api_key: Optional[str] = None,
    azure_ad_token: Optional[str] = None,
    azure_ad_token_provider: Optional[Callable[[], str]] = None,
    has_audio_support: bool = False,
    has_image_support: bool = False,
    sampling_params: Optional[SamplingParams] = None,
    **kwargs,
) -> Model:
    """Create a Model capable of interacting with an Azure AI OpenAI deployment

    Parameters
    ----------
    azure_endpoint : str
        The endpoint which holds the OpenAI model deployment. It will probably be
        https://<AZURE OPENAI RESOURCE NAME>.openai.azure.com/"
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
    echo : bool
        If true the final result of creating this model state will be displayed (as HTML in a notebook).
    has_audio_support : bool
        Indicates if the deployed model has support for audio. This factory attempts
        to work this out from the model_name, but this can be used to force the addition
        of audio support to the returned Model.
    has_image_support : bool
        Indicates if the deployed model has support for images. This factory attempts
        to work this out from the model_name, but this can be used to force the addition
        of image support to the returned Model.
    **kwargs :
        All extra keyword arguments are passed directly to the `openai.OpenAI` constructor. Commonly used argument
        names include `base_url` and `organization`
    """
    if has_audio_support and has_image_support:
        raise ValueError("No known models have both audio and image support")

    interpreter_cls: type[AzureOpenAIInterpreter]
    if (model_name and "audio-preview" in model_name) or has_audio_support:
        interpreter_cls = AzureOpenAIAudioInterpreter
    elif (model_name and (model_name.startswith("gpt-4o") or model_name.startswith("o1"))) or has_image_support:
        interpreter_cls = AzureOpenAIImageInterpreter
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

    model = Model(
        interpreter=interpreter,
        echo=echo,
        sampling_params=SamplingParams() if sampling_params is None else sampling_params,
    )

    return model


class AzureAIClientWrapper(BaseOpenAIClientWrapper):
    def __init__(self, client: "azure.ai.inference.ChatCompletionsClient"):
        self.client = client

    def streaming_chat_completions(
        self,
        model: str,
        messages: list[dict[str, Any]],
        log_probs: Optional[int] = None,
        **kwargs,
    ) -> ContextManager[Iterator["ChatCompletionChunk"]]:
        request = self.client.complete(
            body={
                "model": model,
                "messages": messages,
                "logprobs": log_probs,
                "stream": True,
                **kwargs,
            },
            headers={
                "extra-parameters": "pass-through",
            },
        )
        # It's at least... "mostly" compliant with the OpenAI API?
        return cast(
            ContextManager[Iterator["ChatCompletionChunk"]],
            request,
        )


class AzureInferenceInterpreter(OpenAIRuleMixin, OpenAIJSONMixin, OpenAIRegexMixin, BaseOpenAIInterpreter):
    def __init__(
        self,
        *,
        endpoint: str,
        credential: Union["AzureKeyCredential", "TokenCredential"],
        model_name: str,
    ):
        try:
            import azure.ai.inference
        except ImportError as ie:
            raise Exception(
                "Please install the azure-ai-inference package  using `pip install azure-ai-inference` in order to use guidance.models.AzureInference!"
            ) from ie
        client = azure.ai.inference.ChatCompletionsClient(
            endpoint=endpoint,
            credential=credential,
        )
        super().__init__(model=model_name, client=AzureAIClientWrapper(client))

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
    model_name: str,
    api_key: Optional[str] = None,
    token_credential: Optional["TokenCredential"] = None,
    sampling_params: Optional[SamplingParams] = None,
) -> Model:
    """Create a Model capable of interacting with an Azure AI OpenAI deployment

    Parameters
    ----------
    azure_endpoint : str
        The endpoint which holds the OpenAI model deployment. It can be obtained
        from the AI Foundry portal, and will look something like
        "https://<DEPLOYMENT_NAME>.<AZURE REGION>.models.ai.azure.com"
    echo : bool
        If true the final result of creating this model state will be displayed (as HTML in a notebook).
    model_name : str
        The actual name of the deployed model
    api_key : str
        One of the authentication options. This can be obtained from the AI Foundry portal
    token_credential :
        The other authentication option. An Azure Token Credential
    """
    try:
        from azure.core.credentials import AzureKeyCredential, TokenCredential
    except ImportError as ie:
        raise Exception(
            "Please install the azure-core package using `pip install -U azure-core` in order to use guidance.models.AzureAI!"
        ) from ie

    credential: Union[AzureKeyCredential, TokenCredential, None] = None
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

    result = Model(
        interpreter=interpreter,
        echo=echo,
        sampling_params=SamplingParams() if sampling_params is None else sampling_params,
    )
    return result
