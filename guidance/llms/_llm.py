import os
import pathlib
import diskcache
import asyncio
import platformdirs

class LLM():

    cache_version = 1
    default_system_prompt = "You are a helpful assistant."

    def __init__(self):
        self.chat_mode = False # by default models are not in role-based chat mode
        self.model_name = "unknown"
    
    def __call__(self, *args, asynchronous=False, **kwargs):
        """ Creates a session and calls the LLM with the given arguments.

        Note that this is a convenience wrapper so you don't have to call session(),
        for higher performance across multiple calls, use a session directly.
        """
        with self.session(asynchronous=asynchronous) as s:
            out = s(*args, **kwargs)
        return out
    
    def __getitem__(self, key):
        """ Gets an attribute from the LLM.
        """
        return getattr(self, key)
    
    def session(self, asynchronous=False):
        """ Creates a session for the LLM.

        This implementation is meant to be overridden by subclasses.
        """
        
        if asynchronous:
            return LLMSession(self)
        else:
            return SyncSession(LLMSession(self))
    
    def encode(self, string, **kwargs):
        return self._tokenizer.encode(string, **kwargs)
    
    def decode(self, tokens, **kwargs):
        return self._tokenizer.decode(tokens, **kwargs)
    
    def id_to_token(self, id):
        return self.decode([id])
    
    def token_to_id(self, token):
        return self.encode(token)[0]
    
    @staticmethod
    def _open_cache(file_name):
        return diskcache.Cache(os.path.join(platformdirs.user_cache_dir("guidance"), file_name))
    

class LLMSession():
    def __init__(self, llm):
        self.llm = llm
        self._call_counts = {} # used to track the number of repeated identical calls to the LLM with non-zero temperature
    
    def __enter__(self):
        return self
    
    async def __call__(self, *args, **kwargs):
        return self.llm(*args, **kwargs)

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def _gen_key(self, args_dict):
        var_names = list(args_dict.keys())[1:] # skip the "self" arg
        return "_---_".join([str(v) for v in ([args_dict[k] for k in var_names] + [self.llm.model_name, self.llm.__class__.__name__, self.llm.cache_version])])
    
    def _cache_key(self, args_dict):
        """ Get a cache key for the given args.
        """

        # generate the key without the call count included
        key = self._gen_key(args_dict)

        # if we have non-zero temperature we include the call count in the cache key
        if args_dict.get("temperature", 0) > 0:
            args_dict["call_count"] = self._call_counts.get(key, 0)

            # increment the call count
            self._call_counts[key] = args_dict["call_count"] + 1

            # regenerate the key with the call count in it
            key = self._gen_key(args_dict)

        return key

class SyncSession():
    def __init__(self, session):
        self._session = session
    
    def __enter__(self):
        self._session.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return self._session.__exit__(exc_type, exc_value, traceback)
    
    def __call__(self, *args, **kwargs):
        return asyncio.get_event_loop().run_until_complete(self._session.__call__(*args, **kwargs))