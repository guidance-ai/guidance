
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

class StableLMChat(Transformers):
    """ A HuggingFace transformers version of the StableLM language model with Guidance support.
    """

    cache = LLM._open_cache("_stablelm.diskcache")

    @staticmethod
    def role_start(role):
        return "<|"+role.upper()+"|>"
    
    @staticmethod
    def role_end(role):
        return '' 