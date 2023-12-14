import os
from pathlib import Path
import multiprocessing
from itertools import takewhile
import operator
import threading
import numpy as np
import queue
import time
import tiktoken
import re

from ._model import Chat, Instruct
from ._remote import Remote

class Anthropic(Remote):
    def __init__(self, model, tokenizer=None, echo=True, caching=True, api_base=None, api_key=None, custom_llm_provider=None, temperature=0.0, max_streaming_tokens=1000, **kwargs):
        try:
            from anthropic import Anthropic
        except ImportError:
            raise Exception("Please install the anthropic package version >= 0.7 using `pip install anthropic -U` in order to use guidance.models.Anthropic!")
        
        # if we are called directly (as opposed to through super()) then we convert ourselves to a more specific subclass if possible
        if self.__class__ is Anthropic:
            raise Exception("The Anthropic class is not meant to be used directly! Please use AnthropicChat depending on the model you are using.")

        if api_key is None:
            api_key = os.environ.get("ANTHROPIC_API_KEY")

        self.anthropic = Anthropic(api_key=api_key)

        # self.client = openai_package.OpenAI(api_key=api_key, organization=organization, base_url=base_url)
        self.model_name = model

        # we pretend it tokenizes like gpt2 if tiktoken does not know about it... TODO: make this better
        if tokenizer is None:
            try:
                tokenizer = tiktoken.encoding_for_model(model)
            except:
                tokenizer = tiktoken.get_encoding("gpt2")

        super().__init__(
            model, tokenizer=tokenizer, echo=echo,
            caching=caching, temperature=temperature,
            max_streaming_tokens=max_streaming_tokens, **kwargs
        )

class AnthropicChat(Anthropic, Chat):
    def get_role_start(self, role_name, **kwargs):
        if role_name == "user":
            return "\n\nHuman:"
        if role_name == "assistant":
            return "\n\nAssistant:"
        if role_name == "system":
            return ""
    
    def get_role_end(self, role_name=None):
        return ""
    
    def _generator(self, prompt, temperature):

        # update our shared data state
        self._reset_shared_data(prompt, temperature)

        try:
            generator = self.anthropic.completions.create(
                model=self.model_name,
                prompt=prompt.decode("utf8"),
                max_tokens_to_sample=self.max_streaming_tokens,
                stream=True,
                temperature=temperature
            )
        except Exception as e: # TODO: add retry logic
            raise e
        
        for part in generator:
            chunk = part.completion or ""
            # print(chunk)
            yield chunk.encode("utf8")
