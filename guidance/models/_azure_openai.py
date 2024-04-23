import pathlib
import re

from typing import Type
from urllib.parse import parse_qs, urlparse

import tiktoken

from ._grammarless import Grammarless
from ._model import Chat, Instruct
from ._openai import (
    OpenAIChatEngine,
    OpenAICompletionEngine,
    OpenAIInstructEngine,
    chat_model_pattern,
)

try:
    import openai as openai_package

    is_openai = True
except ModuleNotFoundError:
    is_openai = False


class AzureOpenAI(Grammarless):
    """Represents an Azure OpenAI model as exposed through their remote API.

    Note that because this uses a remote API endpoint without built-in guidance support
    there are some things we cannot do, like force the model to follow a pattern inside
    a chat role block.

    Authentication can be provided via an `api_key` or through Entra/Azure Active
    Directory.
    """

    def __init__(
        self,
        model: str,
        azure_endpoint: str,
        azure_deployment: str = None,
        azure_ad_token_provider=None,
        api_key: str = None,
        tokenizer=None,
        echo=True,
        version: str = None,
        max_streaming_tokens=1000,
        timeout=0.5,
        compute_log_probs=False,
        **kwargs,
    ):
        """Build a new AzureOpenAI model object that represents a model in a given state.

        Parameters
        ----------
        model : str
            The name of the OpenAI model to use (e.g. gpt-3.5-turbo).
        azure_endpoint: str
            The endpoint of the deployed model (e.g. https://my_azureai_instance.openai.azure.com)
        azure_deployment: str
            The deployed name of the model (given when the deployment was created)
        api_key: str
            The API key for calling the model
        azure_ad_token_provider:
            Alternative to the api_key, allows for use of Azure Entra authentication
        """
        if not is_openai or not hasattr(openai_package, "OpenAI"):
            raise ImportError(
                "Please install the openai package version >= 1 using `pip install openai -U` "
                "in order to use guidance.models.OpenAI!"
            )

        if api_key is None and azure_ad_token_provider is None:
            raise ValueError("Please provide either api_key or azure_ad_token_provider")

        parsed_url = urlparse(azure_endpoint)

        # if we are called directly (as opposed to through super()) then we convert ourselves to
        # a more specific subclass if possible
        if self.__class__ is AzureOpenAI:
            # Default to a completion model
            found_subclass: Type[AzureOpenAI] = AzureOpenAICompletion
            # Now see if we should be using a chat model
            if parsed_url.path.endswith("/chat/completions"):
                found_subclass = AzureOpenAIChat
            elif re.match(chat_model_pattern, model):
                found_subclass = AzureOpenAIChat

            # convert to any found subclass
            self.__class__ = found_subclass
            found_subclass.__init__(
                self,
                model=model,
                azure_endpoint=azure_endpoint,
                api_key=api_key,
                azure_ad_token_provider=azure_ad_token_provider,
                azure_deployment=azure_deployment,
                tokenizer=tokenizer,
                echo=echo,
                version=version,
                **kwargs,
            )
            return

        if azure_deployment is None:
            azure_deployment = pathlib.Path(parsed_url.path).parts[3]
        parsed_query = parse_qs(parsed_url.query)
        api_version = (
            version
            if "api-version" not in parsed_query
            else parsed_query["api-version"]
        )
        engine_map = {
            AzureOpenAICompletion: OpenAICompletionEngine,
            AzureOpenAIChat: OpenAIChatEngine,
            AzureOpenAIInstruct: OpenAIInstructEngine,
        }
        engine_class = engine_map[self.__class__]

        if tokenizer is None:
            tokenizer = tiktoken.encoding_for_model(model)

        engine_instance = engine_class(
            tokenizer=tokenizer,
            max_streaming_tokens=max_streaming_tokens,
            timeout=timeout,
            compute_log_probs=compute_log_probs,
            model=model,
            azure_endpoint=azure_endpoint,
            api_key=api_key,
            azure_ad_token_provider=azure_ad_token_provider,
            api_version=api_version,
            azure_deployment=azure_deployment,
            client_class=openai_package.AzureOpenAI,
            **kwargs,
        )

        super().__init__(
            engine_instance,
            echo=echo,
        )


class AzureOpenAIChat(AzureOpenAI, Chat):
    pass


class AzureOpenAICompletion(AzureOpenAI):
    pass


class AzureOpenAIInstruct(AzureOpenAI, Instruct):
    def get_role_start(self, name):
        return ""

    def get_role_end(self, name):
        if name == "instruction":
            return "<|endofprompt|>"
        else:
            raise ValueError(
                f"The OpenAIInstruct model does not know about the {name} role type!"
            )
