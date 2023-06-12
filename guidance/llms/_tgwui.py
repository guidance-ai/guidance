import os
import time
import collections
import regex
import pygtrie
import torch
import queue
import threading
import logging
import collections.abc
import asyncio
import requests
from typing import Any, Dict, Optional, Callable
from ._llm import LLM, LLMSession, SyncSession


class TGWUI(LLM):
    def __init__(self, base_url, chat_mode=False):
        self.chat_mode = False  # by default models are not in role-based chat mode
        self.base_url = base_url
        self.model_name = "unknown"


    def __getitem__(self, key):
        """Gets an attribute from the LLM."""
        return getattr(self, key)

    def session(self, asynchronous=False):
        """Creates a session for the LLM.

        This implementation is meant to be overridden by subclasses.
        """
        return TWGUISession(self)

    def encode(self, string, **kwargs):
        args={"text": string, "kwargs": kwargs}
        try:
            response = requests.post(self.base_url+'/api/v1/encode',json=args)
            response.raise_for_status()  
        except requests.exceptions.RequestException as e:
            print(f'Encode request failed with error {e}')
            return None

        resp = response.json()
        if 'results' in resp and len(resp['results']) > 0 and 'tokens' in resp['results'][0]:
            return resp['results'][0]['tokens'][0]
        else:
            print('Unexpected response format')
            return None


    def decode(self, tokens, **kwargs):
        args={"tokens": tokens, "kwargs": kwargs}
        try:
            response = requests.post(self.base_url+'/api/v1/decode',json=args)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f'Decode request failed with error {e}')
            return None

        resp = response.json()
        if 'results' in resp and len(resp['results']) > 0 and 'ids' in resp['results'][0]:
            return resp['results'][0]['ids']
        else:
            print('Unexpected response format')
            return None


    

class TWGUISession(LLMSession):
    def __init__(self, llm):
        self.llm = llm
        self._call_counts = {} # tracks the number of repeated identical calls to the LLM with non-zero temperature

    def __enter__(self):
        return self

    async def __call__(
            self, prompt, stop=None, stop_regex=None, temperature=None, n=1, max_tokens=1000, logprobs=None,                
            top_p=1.0, echo=False, logit_bias=None, token_healing=None, pattern=None, stream=None,
            cache_seed=0, caching=None, **completion_kwargs
        ):
        args={
            "prompt":prompt, "stop": stop, "stop_regex":stop_regex, "temperature": temperature, "n":n, 
            "max_tokens":max_tokens, "logprobs":logprobs, "top_p":top_p, "echo":echo, "logit_bias":logit_bias, 
            "token_healing":token_healing, "pattern":pattern, "stream":stream, "cache_seed":cache_seed, 
            "completion_kwargs":completion_kwargs
        }

        try:
            response = requests.post(self.llm.base_url+'/api/v1/call',json=args)
            response.raise_for_status()  # This will raise a HTTPError if the response was an error
        except requests.exceptions.RequestException as e:  # This will catch any kind of request exception
            print(f'Request failed with error {e}')
            return None

        resp = response.json()
        if 'choices' in resp and len(resp['choices']) > 0:
            return resp['choices'][0]['text']
        else:
            print('Unexpected response format')
            return None

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def _gen_key(self, args_dict):
        del args_dict["self"]  # skip the "self" arg
        return "_---_".join([str(v) for v in ([args_dict[k] for k in args_dict] + [self.llm.model_name, self.llm.__class__.__name__, self.llm.cache_version])])

    def _cache_params(self, args_dict) -> Dict[str, Any]:
        """get the parameters for generating the cache key"""
        key = self._gen_key(args_dict)
        # if we have non-zero temperature we include the call count in the cache key
        if args_dict.get("temperature", 0) > 0:
            args_dict["call_count"] = self._call_counts.get(key, 0)

            # increment the call count
            self._call_counts[key] = args_dict["call_count"] + 1
        args_dict["model_name"] = self.llm.model_name
        args_dict["cache_version"] = self.llm.cache_version
        args_dict["class_name"] = self.llm.__class__.__name__

        return args_dict