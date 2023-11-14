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

from ._vertexai import VertexAIInstruct, VertexAIChat

# try:
#     # TODO: can we eliminate the torch requirement for llama.cpp by using numpy in the caller instead?
#     import torch
#     is_torch = True
# except ImportError:
#     is_torch = False

try:
    # TODO: can we eliminate the torch requirement for llama.cpp by using numpy in the caller instead?
    from vertexai.preview.language_models import TextGenerationModel
    from vertexai.language_models import ChatModel, InputOutputTextPair
    is_vertexai = True
except ImportError:
    is_vertexai = False

class PaLM2Instruct(VertexAIInstruct):
    def __init__(self, model, tokenizer=None, echo=True, caching=True, temperature=0.0, max_streaming_tokens=500, **kwargs):\
    
        if isinstance(model, str):
            self.model_name = model
            self.model_obj = TextGenerationModel.from_pretrained(self.model_name)
        
        # PaLM2 does not have a public tokenizer, so we pretend it tokenizes like gpt2...
        if tokenizer is None:
            tokenizer = tiktoken.get_encoding("gpt2")
        
        # the superclass does all the work
        super().__init__(
            model,
            tokenizer=tokenizer,
            echo=echo,
            caching=caching,
            temperature=temperature,
            max_streaming_tokens=max_streaming_tokens,
            **kwargs
        )

class PaLM2Chat(VertexAIChat):
    def __init__(self, model, tokenizer=None, echo=True, caching=True, temperature=0.0, max_streaming_tokens=500, **kwargs):\
    
        if isinstance(model, str):
            self.model_name = model
            self.model_obj = ChatModel.from_pretrained(self.model_name)
        
        # PaLM2 does not have a public tokenizer, so we pretend it tokenizes like gpt2...
        if tokenizer is None:
            tokenizer = tiktoken.get_encoding("gpt2")
        
        # the superclass does all the work
        super().__init__(
            model,
            tokenizer=tokenizer,
            echo=echo,
            caching=caching,
            temperature=temperature,
            max_streaming_tokens=max_streaming_tokens,
            **kwargs
        )

    # def get_role_start(self, role_name, **kwargs):
    #     if role_name == "user":

    #         # if we follow an auto-nested system role then we are done
    #         if self._current_prompt().endswith("\n<</SYS>>\n\n"):
    #             return ""
    #         else:
    #             return "[INST] "
        
    #     elif role_name == "assistant":
    #         return " "
        
    #     elif role_name == "system":
            
    #         # check if we are already embedded at the top of a user role
    #         if self._current_prompt().endswith("[INST] "):
    #             return "<<SYS>>\n"

    #         # if not then we auto nest ourselves
    #         else:
    #             return "[INST] <<SYS>>\n"
    
    # def get_role_end(self, role_name=None):
    #     if role_name == "user":
    #         return " [/INST]"
    #     elif role_name == "assistant":
    #         return " "
    #     elif role_name == "system":
    #         return "\n<</SYS>>\n\n"