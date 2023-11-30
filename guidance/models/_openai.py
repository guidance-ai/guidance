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

try:
    # TODO: can we eliminate the torch requirement for llama.cpp by using numpy in the caller instead?
    import openai as openai_package
    is_openai = True
except ImportError:
    is_openai = False

chat_model_pattern = r'^(ft:)?(gpt-3\.5-turbo|gpt-4)((-\w+)+)?(:[\w-]+(?:[:\w-]+)*)?(::\w+)?$'

class OpenAI(Remote):
    def __init__(self, model, tokenizer=None, echo=True, caching=True, api_key=None, organization=None, base_url=r"https://api.openai.com/v1", temperature=0.0, top_p=1.0, max_streaming_tokens=1000, **kwargs):
        if not is_openai or not hasattr(openai_package, "OpenAI"):
            raise Exception("Please install the openai package version >= 1 using `pip install openai -U` in order to use guidance.models.OpenAI!")
        
        # if we are called directly (as opposed to through super()) then we convert ourselves to a more specific subclass if possible
        if self.__class__ is OpenAI:
            found_subclass = None

            # chat
            if re.match(chat_model_pattern, model):
                found_subclass = OpenAIChat

            # instruct
            elif "instruct" in model:
                found_subclass = OpenAIInstruct

            # regular completion
            else:
                found_subclass = OpenAICompletion
            
            # convert to any found subclass
            self.__class__ = found_subclass
            found_subclass.__init__(self, model, tokenizer=tokenizer, echo=echo, caching=caching, api_key=api_key, organization=organization, base_url=base_url, temperature=temperature, max_streaming_tokens=max_streaming_tokens, **kwargs)
            return # we return since we just ran init above and don't need to run again

        # Configure an AsyncOpenAI Client with user params.
        if api_key is None:
            api_key = os.environ.get("OPENAI_API_KEY")

        if organization is None:
            organization = os.environ.get("OPENAI_ORG_ID")

        self.client = openai_package.OpenAI(api_key=api_key, organization=organization, base_url=base_url)
        self.model_name = model
        self.top_p = top_p

        super().__init__(
            model, tokenizer=tiktoken.encoding_for_model(model), echo=echo,
            caching=caching, temperature=temperature,
            max_streaming_tokens=max_streaming_tokens, **kwargs
        )
        

class OAIChatMixin:
    def _generator(self, prompt):
        # find the role tags
        pos = 0
        role_end = b'<|im_end|>'
        messages = []
        found = True
        while found:

            # find the role text blocks
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
            generator = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=self.max_streaming_tokens,
                n=1,
                top_p=self.top_p,
                temperature=self.temperature,
                stream=True
            )
        except Exception as e: # TODO: add retry logic
            raise e

        for part in generator:
            if len(part.choices) > 0:
                chunk = part.choices[0].delta.content or ""
            else:
                chunk = ""
            yield chunk.encode("utf8")


class OAICompletionMixin:
    def _generator(self, prompt):
        self._shared_state["not_running_stream"].clear() # so we know we are running
        self._shared_state["data"] = prompt # we start with this data

        try:
            generator = self.client.completions.create(
                model=self.model_name,
                prompt=prompt.decode("utf8"), 
                max_tokens=self.max_streaming_tokens, 
                n=1, 
                top_p=self.top_p, 
                temperature=self.temperature, 
                stream=True
            )
        except Exception as e: # TODO: add retry logic
            raise e
        
        for part in generator:
            if len(part.choices) > 0:
                chunk = part.choices[0].text or ""
            else:
                chunk = ""
            yield chunk.encode("utf8")


class OAIInstructMixin:
    def get_role_start(self, name):
        return ""
    
    def get_role_end(self, name):
        if name == "instruction":
            return "<|endofprompt|>"
        else:
            raise Exception(f"The OpenAIInstruct model does not know about the {name} role type!")

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
            generator = self.client.completions.create(
                model=self.model_name,
                prompt=self._shared_state["data"].decode("utf8"), 
                max_tokens=self.max_streaming_tokens, 
                n=1, 
                top_p=self.top_p, 
                temperature=self.temperature, 
                stream=True
            )
        except Exception as e: # TODO: add retry logic
            raise e
        
        for part in generator:
            if len(part.choices) > 0:
                chunk = part.choices[0].text or ""
            else:
                chunk = ""
            yield chunk.encode("utf8")

class OpenAICompletion(OpenAI, Instruct, OAICompletionMixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class OpenAIInstruct(OpenAI, Instruct, OAIInstructMixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class OpenAIChat(OpenAI, Chat, OAIChatMixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
