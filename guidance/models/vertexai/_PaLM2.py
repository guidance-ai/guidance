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

from ._vertexai import VertexAICompletion, VertexAIInstruct, VertexAIChat

try:
    from vertexai.language_models import TextGenerationModel, ChatModel, InputOutputTextPair
    is_vertexai = True
except ImportError:
    is_vertexai = False

class PaLM2Completion(VertexAICompletion):
    def __init__(self, model, tokenizer=None, echo=True, caching=True, temperature=0.0, max_streaming_tokens=None, **kwargs):
    
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

class PaLM2Instruct(VertexAIInstruct):
    def __init__(self, model, tokenizer=None, echo=True, caching=True, temperature=0.0, max_streaming_tokens=None, **kwargs):
    
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
    def __init__(self, model, tokenizer=None, echo=True, caching=True, temperature=0.0, max_streaming_tokens=None, **kwargs):
    
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