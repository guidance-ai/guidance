from typing import Callable

from . import experimental
from ._azureai import create_azure_aifoundry_model, create_azure_openai_model
from ._base import Model
from ._llama_cpp import LlamaCpp
from ._mock import Mock
from ._onnxruntime import OnnxRuntimeGenAI
from ._openai import OpenAI
from ._transformers import Transformers


def AzureOpenAI(
    model: str,
    azure_endpoint: str,
    azure_deployment: str,
    *,
    version: str | None = None,
    api_key: str | None = None,
    azure_ad_token: str | None = None,
    azure_ad_token_provider: Callable[[], str] | None = None,
    echo: bool = True,
    **kwargs,
) -> Model:
    """Create an Azure OpenAI model.

    This is a convenience wrapper around create_azure_openai_model that provides
    a more intuitive API matching the OpenAI SDK naming conventions.

    Parameters
    ----------
    model : str
        The name of the Azure OpenAI model to use (e.g. gpt-4o-mini).
    azure_endpoint : str
        The endpoint URL for the Azure OpenAI service.
    azure_deployment : str
        The Azure deployment name to use for the model.
    version : str | None
        The API version to use for the Azure OpenAI service.
    api_key : str | None
        The API key to use for the Azure OpenAI service.
    azure_ad_token : str | None
        The Azure AD token to use for authentication.
    azure_ad_token_provider : Callable[[], str] | None
        A callable that returns an Azure AD token for authentication.
    echo : bool
        If true the final result will be displayed (as HTML in a notebook).
    **kwargs :
        Additional arguments passed to the underlying client.
    """
    return create_azure_openai_model(
        azure_endpoint=azure_endpoint,
        azure_deployment=azure_deployment,
        model_name=model,
        api_version=version,
        api_key=api_key,
        azure_ad_token=azure_ad_token,
        azure_ad_token_provider=azure_ad_token_provider,
        echo=echo,
        **kwargs,
    )


__all__ = [
    "AzureOpenAI",
    "LlamaCpp",
    "Mock",
    "Model",
    "OnnxRuntimeGenAI",
    "OpenAI",
    "Transformers",
    "create_azure_aifoundry_model",
    "create_azure_openai_model",
    "experimental",
]
