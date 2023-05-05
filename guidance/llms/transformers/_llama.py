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

    def __init__(self, model, tokenizer=None, device_map=None, **kwargs):
        """ Create a new LLaMA model.
        """

        # load the LLaMA specific tokenizer and model
        import transformers
        if isinstance(model, str):
            if tokenizer is None:
                tokenizer = transformers.LlamaTokenizer.from_pretrained(model, device_map=device_map)
            model = transformers.LlamaForCausalLM.from_pretrained(model, device_map=device_map)

        super().__init__(model, tokenizer=tokenizer, device_map=device_map, **kwargs)