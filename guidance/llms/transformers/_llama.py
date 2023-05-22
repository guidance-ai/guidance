import os
import time
import collections
import regex
import pygtrie
import queue
import threading
import logging
from .._llm import LLM, LLMSession, SyncSession
from .._transformers import Transformers, TransformersSession

class LLaMA(Transformers):
    """ A HuggingFace transformers version of the LLaMA language model with Guidance support.
    """

    cache = LLM._open_cache("_llama.diskcache")

    def __init__(self, model, tokenizer=None, caching=True, token_healing=True, \
                 acceleration=True, temperature=0.0, device=None, device_map=None, **kwargs):
        """ Create a new LLaMA model.
        """

        # load the LLaMA specific tokenizer and model
        import transformers
        if isinstance(model, str):
            if tokenizer is None:
                tokenizer = transformers.LlamaTokenizer.from_pretrained(model, device_map=device_map, **kwargs)
            model = transformers.LlamaForCausalLM.from_pretrained(model, device_map=device_map, **kwargs)

        super().__init__(model, tokenizer=tokenizer, caching=caching, token_healing=token_healing, \
                         acceleration=acceleration, temperature=temperature, device=device, device_map=device_map, **kwargs)