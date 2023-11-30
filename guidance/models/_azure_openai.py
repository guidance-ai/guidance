import re
from urllib.parse import urlparse, parse_qs

import tiktoken

from ._model import Chat
from ._model import Instruct
from ._openai import (
    OAIChatMixin,
    OAICompletionMixin,
    OAIInstructMixin,
    chat_model_pattern,
)
from ._remote import Remote

try:
    # TODO: can we eliminate the torch requirement for llama.cpp by using numpy in the caller instead?
    import openai as openai_package

    is_openai = True
except ImportError:
    is_openai = False


class AzureOpenAI(Remote):
    def __init__(
        self,
        model: str,
        azure_endpoint: str,
        api_key: str,
        tokenizer=None,
        echo=True,
        caching=True,
        temperature=0.0,
        max_streaming_tokens=500,
        version="2023-10-01-preview",
        **kwargs,
    ):
        if not is_openai or not hasattr(openai_package, "OpenAI"):
            raise Exception(
                "Please install the openai package version >= 1 using `pip install openai -U` "
                "in order to use guidance.models.OpenAI!"
            )

        # if we are called directly (as opposed to through super()) then we convert ourselves to
        # a more specific subclass if possible
        if self.__class__ is AzureOpenAI:
            # if isinstance(model, str):
            #     model_name = model
            # else:
            #     model_name = self.model_obj._model_id

            # chat
            if re.match(chat_model_pattern, model):
                found_subclass = AzureOpenAIChat

            # instruct
            elif "instruct" in model:
                found_subclass = AzureOpenAIInstruct

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
                tokenizer=tokenizer,
                echo=echo,
                caching=caching,
                temperature=temperature,
                max_streaming_tokens=max_streaming_tokens,
                version=version,
                **kwargs,
            )
            return

        parsed_url = urlparse(azure_endpoint)
        parsed_query = parse_qs(parsed_url.query)
        api_version = (
            version
            if "api-version" not in parsed_query
            else parsed_query["api-version"]
        )

        self.client = openai_package.AzureOpenAI(
            azure_endpoint=azure_endpoint, api_key=api_key, api_version=api_version
        )
        self.model_name = model

        super().__init__(
            model=azure_endpoint,
            tokenizer=tiktoken.encoding_for_model(model),
            echo=echo,
            caching=caching,
            temperature=temperature,
            max_streaming_tokens=max_streaming_tokens,
            **kwargs,
        )


class AzureOpenAIChat(AzureOpenAI, OAIChatMixin, Chat):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class AzureOpenAIInstruct(AzureOpenAI, OAIInstructMixin, Instruct):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class AzureOpenAICompletion(AzureOpenAI, OAICompletionMixin, Instruct):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
