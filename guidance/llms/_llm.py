import os
import pathlib
import diskcache
import platformdirs

class LLM():

    cache_version = 1

    def __init__(self):

        # by default models are not in role-based chat mode
        self.chat_mode = False
    
    def __call__(self, *args, **kwargs):
        """ Creates a session and calls the LLM with the given arguments.

        Note that this is a convenience wrapper so you don't have to call session(),
        for higher performance across multiple calls, use a session directly.
        """
        with self.session() as s:
            out = s(*args, **kwargs)
        return out
    
    def _cache_key(self, args):
        """ Get a cache key for the given args.
        """
        var_names = list(args.keys())[1:] # skip the "self" arg
        key = "_---_".join([str(v) for v in ([args[k] for k in var_names] + [self.model_name, self.__class__.__name__, self.cache_version])])
        return key
    
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
    
    def __enter__(self):
        return self
    
    def __call__(self, *args, **kwargs):
        return self.llm(*args, **kwargs)

    def __exit__(self, exc_type, exc_value, traceback):
        pass