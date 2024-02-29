import os
from pathlib import Path
from itertools import takewhile
import operator
import sys
import logging
import numpy as np

from .._model import Tokenizer, Engine, Model, Chat
from .._remote import RemoteEngine
from ..._utils import normalize_notebook_stdout_stderr

try:
    import llama_cpp
    is_llama_cpp = True
except ImportError:
    is_llama_cpp = False

logger = logging.getLogger(__name__)

class _LlamaBatchContext:
    def __init__(self, n_batch, n_ctx):
        self._llama_batch_free = llama_cpp.llama_batch_free
        self.batch = llama_cpp.llama_batch_init(n_batch, 0, n_ctx)
        if self.batch is None:
            raise Exception("call to llama_cpp.llama_batch_init returned NULL.")

    def __del__(self):
        llama_batch_free = getattr(self, "_llama_batch_free", None)
        batch = getattr(self, "batch", None)
        if batch is not None and llama_batch_free is not None:
            self._llama_batch_free = None
            self.batch = None
            llama_batch_free(batch)

class LlamaCppTokenizer(Tokenizer):
    def __init__(self, model_obj):
        self._model_obj = model_obj

        tokenizer = llama_cpp.LlamaTokenizer(model_obj)
        if not hasattr(tokenizer, 'llama'):
            tokenizer.llama = tokenizer._model

        # get the bytes strings for all the tokens
        tokens = []
        for i in range(tokenizer.llama.n_vocab()):
            tok = tokenizer.llama.detokenize([i]) # note that detokenize returns bytes directly
            if tok == b'':
                tok = llama_cpp.llama_token_get_text(model_obj.model, i) # get text rep of special tokens
            tokens.append(tok)

        super().__init__(
            tokens,
            tokenizer.llama.token_bos(),
            tokenizer.llama.token_eos()
        )

    def __call__(self, byte_string):
        return self._model_obj.tokenize(byte_string, add_bos=False, special=True)

class LlamaCppEngine(Engine):
    '''The core class that runs inference using llama.cpp.'''

    def __init__(self, model, compute_log_probs, **kwargs):
        if not is_llama_cpp:
            raise Exception("Please install llama-cpp-python with `pip install llama-cpp-python` in order to use guidance.models.LlamaCpp!")

        if isinstance(model, Path):
            model = str(model)
        if model is None or isinstance(model, str) and len(model.strip()) == 0:
            model = os.environ.get("LLAMA_CPP_MODEL", "")
            if len(model.strip()) == 0:
                try:
                    with open(os.path.expanduser('~/.llama_cpp_model'), 'r') as file:
                        model = file.read().replace('\n', '')
                except:
                    pass
                if len(model.strip()) == 0:
                    raise ValueError("If model is None then a model file must be specified in either the LLAMA_CPP_MODEL environment variable or in the ~/.llama_cpp_model file.")

        if isinstance(model, str):
            self.model = model
            if "verbose" not in kwargs:
                kwargs["verbose"] = False

            # patch over https://github.com/abetlen/llama-cpp-python/issues/729
            try:
                sys.stdout.fileno()
            except:
                logger.warn("Cannot use verbose=True in this context (probably CoLab). See https://github.com/abetlen/llama-cpp-python/issues/729")
                kwargs["verbose"] = True # llama-cpp-python can't hide output in this case

            with normalize_notebook_stdout_stderr():
                self.model_obj = llama_cpp.Llama(model_path=model, **kwargs)
        elif isinstance(model, llama_cpp.Llama):
            self.model = model.__class__.__name__
            self.model_obj = model
        else:
            raise TypeError("model must be None, a file path string, or a llama_cpp.Llama object.")

        self._context = _LlamaBatchContext(self.model_obj.n_batch, self.model_obj.n_ctx())
        self._cache_token_ids = []

        super().__init__(
            LlamaCppTokenizer(self.model_obj),
            compute_log_probs=compute_log_probs
        )

        self._n_vocab = len(self.tokenizer.tokens)

    def _joint_tokenize(self, token_ids):
        '''What a full joint tokenizer would give for a given byte string'''
        byte_string = b"".join([self.tokenizer.tokens[t] for t in token_ids])
        return self.tokenizer(byte_string)

    def get_logits(self, token_ids, forced_bytes, current_temp):
        '''Computes the logits for the given token state.
        
        This overrides a method from the LocalEngine class that is used to get
        inference results from the model.
        '''

        if len(token_ids) == 0:
            raise ValueError("token_ids must contain some tokens.")

        # check what we have already cached
        cache_token_ids = self._cache_token_ids
        num_cached = sum(takewhile(operator.truth, map(operator.eq, token_ids, cache_token_ids)))
        if num_cached == len(token_ids):
            if num_cached == len(cache_token_ids):
                return self._cached_logits
            num_cached = num_cached - 1 # llama_cpp doesn't like it when we pass in 0 new tokens, so re-input one
        
        # make sure we don't run off the end of the model's context
        if self.model_obj.n_ctx() <= len(token_ids):
            raise Exception(f"Attempted to use a context length of {len(token_ids)} tokens, but this LlamaCpp model is only configured to support up to {self.model_obj.n_ctx()}!")

        self._cache_token_ids = token_ids.copy()

        # clear obsolete parts of kv cache
        llama_cpp.llama_kv_cache_seq_rm(self.model_obj.ctx, -1, num_cached, -1)

        # eval the model
        n_batch = self.model_obj.n_batch
        batch = self._context.batch
        for i in range(num_cached, len(token_ids), n_batch):
            n_tokens = min(i + n_batch, len(token_ids)) - i
            batch.n_tokens = n_tokens
            for j in range(n_tokens):
                batch.token[j] = token_ids[i + j]
                batch.pos[j] = i + j
                batch.seq_id[j][0] = 0
                batch.n_seq_id[j] = 1
                batch.logits[j] = False

            if i + n_tokens == len(token_ids):
                batch.logits[n_tokens - 1] = True

            ret = llama_cpp.llama_decode(self.model_obj.ctx, batch)
            if ret != 0:
                raise Exception(f"Call to llama_cpp.llama_decode returned {ret}.")

        # get the logits
        logits = llama_cpp.llama_get_logits(self.model_obj.ctx)
        logits = logits[(n_tokens - 1) * self._n_vocab : n_tokens * self._n_vocab]
        logits = np.ctypeslib.as_array(logits, shape=(self._n_vocab,)).copy()

        self._cached_logits = logits

        return logits

class LlamaCpp(Model):
    def __init__(self, model=None, echo=True, compute_log_probs=False, api_key=None, **llama_cpp_kwargs):
        '''Build a new LlamaCpp model object that represents a model in a given state.'''

        if isinstance(model, str) and model.startswith("http"):
            engine = RemoteEngine(model, api_key=api_key, **llama_cpp_kwargs)
        else:
            engine = LlamaCppEngine(model, compute_log_probs=compute_log_probs, **llama_cpp_kwargs)

        super().__init__(
            engine,
            echo=echo
        )


class LlamaCppChat(LlamaCpp, Chat):
    def get_role_start(self, role_name, **kwargs):
        if role_name == "user":

            # if we follow an auto-nested system role then we are done
            if self._current_prompt().endswith("\n<</SYS>>\n\n"):
                return ""
            else:
                return "[INST] "
        
        elif role_name == "assistant":
            return " "
        
        elif role_name == "system":
            
            # check if we are already embedded at the top of a user role
            if self._current_prompt().endswith("[INST] "):
                return "<<SYS>>\n"

            # if not then we auto nest ourselves
            else:
                return "[INST] <<SYS>>\n"
    
    def get_role_end(self, role_name=None):
        if role_name == "user":
            return " [/INST]"
        elif role_name == "assistant":
            return " "
        elif role_name == "system":
            return "\n<</SYS>>\n\n"
