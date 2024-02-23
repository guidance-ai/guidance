from urllib.parse import parse_qs, urlparse

from ._grammarless import Grammarless
from ._model import Chat, Instruct
from ._openai import (
    OpenAIChatEngine,
    OpenAICompletionEngine,
    OpenAIInstructEngine,
)

try:
    # TODO: can we eliminate the torch requirement for llama.cpp by using numpy in the caller instead?
    import openai as openai_package

    is_openai = True
except ImportError:
    is_openai = False


class AzureOpenAI(Grammarless):
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
        """Build a new AzureOpenAI model object that represents a model in a given state."""
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
            # chat
            if parsed_url.path.endswith("/chat/completions"):
                found_subclass = AzureOpenAIChat
            # regular completion
            else:
                found_subclass = AzureOpenAICompletion

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
