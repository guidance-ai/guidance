import os
import pathlib
import diskcache
import platformdirs

class LLM():

    cache_version = 1

    def __init__(self):
        self.chat_mode = False # by default models are not in role-based chat mode
        self.model_name = "unknown"
    
    def __call__(self, *args, **kwargs):
        """ Creates a session and calls the LLM with the given arguments.

        Note that this is a convenience wrapper so you don't have to call session(),
        for higher performance across multiple calls, use a session directly.
        """
        with self.session() as s:
            out = s(*args, **kwargs)
        return out
    
    def session(self):
        return LLMSession(self) # meant to be overridden
    
    def encode(self, string, **kwargs):
        return self._tokenizer.encode(string, **kwargs)
    
    def decode(self, tokens, **kwargs):
        return self._tokenizer.decode(tokens, **kwargs)
    
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