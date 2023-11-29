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


class LiteLLM(Remote):
    def __init__(self, model, tokenizer=None, echo=True, caching=True, api_base=None, api_key=None, custom_llm_provider=None, temperature=0.0, max_streaming_tokens=1000, **kwargs):
        try:
            import litellm
        except ImportError:
            raise Exception("Please install the litellm package version >= 1.7 using `pip install litellm -U` in order to use guidance.models.LiteLLM!")
        
        # if we are called directly (as opposed to through super()) then we convert ourselves to a more specific subclass if possible
        if self.__class__ is LiteLLM:
            raise Exception("The LightLLM class is not meant to be used directly! Please use LiteLLMChat, LiteLLMInstruct, or LiteLLMCompletion depending on the model you are using.")


        self.litellm = litellm

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


class LiteLLMCompletion(LiteLLM, Instruct):

    def _generator(self, prompt):
        self._shared_state["not_running_stream"].clear() # so we know we are running
        self._shared_state["data"] = prompt # we start with this data

        try:
            generator = self.litellm.completion(
                model=self.model_name,
                messages=[{"content": prompt.decode("utf8"), "role": "system"}], # note that role=system is just ignored by litellm but used by them to match chat syntax
                max_tokens=self.max_streaming_tokens,
                n=1,
                top_p=1,
                temperature=0,
                stream=True
            )
        except Exception as e: # TODO: add retry logic
            raise e
        
        for part in generator:
            chunk = part.choices[0].delta.content or ""
            yield chunk.encode("utf8")

class LiteLLMInstruct(LiteLLM, Instruct):

    def get_role_start(self, name):
        return ""
    
    def get_role_end(self, name):
        if name == "instruction":
            return "<|endofprompt|>"
        else:
            raise Exception(f"The LiteLLMInstruct model does not know about the {name} role type!")

    def _generator(self, prompt):
        # start the new stream
        prompt_end = prompt.find(b'<|endofprompt|>')
        if prompt_end >= 0:
            stripped_prompt = prompt[:prompt_end]
        else:
            raise Exception("This model cannot handle prompts that don't match the instruct format!")
        
        # make sure you don't try and instruct the same model twice
        if b'<|endofprompt|>' in prompt[prompt_end + len(b'<|endofprompt|>'):]:
            raise Exception("This model has been given two separate instruct blocks, but this is not allowed!")
        
        self._shared_state["not_running_stream"].clear() # so we know we are running
        self._shared_state["data"] = stripped_prompt + b'<|endofprompt|>'# we start with this data

        try:
            generator = self.litellm.completion(
                model=self.model_name,
                messages=[{"content": self._shared_state["data"].decode("utf8"), "role": "system"}], # note that role=system is just ignored by litellm but used by them to match chat syntax
                prompt=self._shared_state["data"].decode("utf8"), 
                max_tokens=self.max_streaming_tokens, 
                n=1, 
                top_p=1, 
                temperature=0, 
                stream=True
            )
        except Exception as e: # TODO: add retry logic
            raise e
        
        for part in generator:
            chunk = part.choices[0].delta.content or ""
            yield chunk.encode("utf8")

class LiteLLMChat(LiteLLM, Chat):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _generator(self, prompt):
        
        # find the system text
        pos = 0
        role_end = b'<|im_end|>'

        # find the user/assistant pairs
        messages = []
        found = True
        while found:

            # find the user text
            found = False
            for role_name,start_bytes in (("system", b'<|im_start|>system\n'), ("user", b'<|im_start|>user\n'), ("assistant", b'<|im_start|>assistant\n')):
                if prompt[pos:].startswith(start_bytes):
                    pos += len(start_bytes)
                    end_pos = prompt[pos:].find(role_end)
                    if end_pos < 0:
                        assert role_name == "assistant", "Bad chat format! Last role before gen needs to be assistant!"
                        break
                    btext = prompt[pos:pos+end_pos]
                    pos += end_pos + len(role_end)
                    messages.append({"role": role_name, "content": btext.decode("utf8")})
                    found = True
                    break
        
        self._shared_state["data"] = prompt[:pos]

        try:
            generator = self.litellm.completion(
                model=self.model_name,
                messages=messages,
                max_tokens=self.max_streaming_tokens, 
                n=1, 
                top_p=1, 
                temperature=0, 
                stream=True
            )
        except Exception as e: # TODO: add retry logic
            raise e

        for part in generator:
            chunk = part.choices[0].delta.content or ""
            yield chunk.encode("utf8")