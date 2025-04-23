from typing import Callable, Optional

from ._base import Model
from ._openai_base import BaseOpenAIInterpreter, OpenAIAudioMixin, OpenAIImageMixin


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
