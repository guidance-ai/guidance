import os
from pathlib import Path
from itertools import takewhile
import operator

import numpy as np

from ._model import Chat
from ._local import Local
from .._utils import normalize_notebook_stdout_stderr

try:
    import llama_cpp
    is_llama_cpp = True
except ImportError:
    is_llama_cpp = False

class LlamaCpp(Local):
    def __init__(self, model=None, tokenizer=None, echo=True, caching=True, temperature=0.0, **kwargs):

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

            with normalize_notebook_stdout_stderr():
                self.model_obj = llama_cpp.Llama(model_path=model, **kwargs)
        elif isinstance(model, llama_cpp.Llama):
            self.model = model.__class__.__name__
            self.model_obj = model
        else:
            raise TypeError("model must be None, a file path string, or a llama_cpp.Llama object.")

        if tokenizer is None:
            tokenizer = llama_cpp.LlamaTokenizer(self.model_obj)
        elif not isinstance(tokenizer, llama_cpp.LlamaTokenizer):
            raise TypeError("tokenizer must be None or a llama_cpp.LlamaTokenizer object.")
        self._orig_tokenizer = tokenizer

        self._n_vocab = tokenizer.llama.n_vocab()
        self.caching = caching
        self.temperature = temperature

        tokens = [tokenizer.llama.detokenize([i]) for i in range(self._n_vocab)] # note that detokenize returns bytes directly
        tokens[1] = b"<s>" # these are not decoded correctly by llama_cpp
        tokens[2] = b"</s>"
        super().__init__(
            tokens,
            tokenizer.llama.token_bos(),
            tokenizer.llama.token_eos(),
            echo=echo
        )

        self._cache_state["cache_token_ids"] = []

    def _joint_tokenize(self, token_ids):
        byte_string = b"".join([self.tokens[t] for t in token_ids])
        return self.model_obj.tokenize(byte_string, add_bos=False, special=True)

    def _get_logits(self, token_ids, forced_bytes):
        '''Computes the logits for the given token state.
        
        This overrides a method from the LocalEngine class that is used to get
        inference results from the model.
        '''

        if len(token_ids) == 0:
            raise ValueError("token_ids must contain some tokens.")

        # check what we have already cached
        cache_token_ids = self._cache_state["cache_token_ids"]
        num_cached = sum(takewhile(operator.truth, map(operator.eq, token_ids, cache_token_ids)))
        if num_cached == len(token_ids):
            if num_cached == len(cache_token_ids):
                return self._cache_state["logits"]
            num_cached = num_cached - 1 # llama_cpp doesn't like it when we pass in 0 new tokens, so re-input one
        
        # make sure we don't run off the end of the model's context
        if self.model_obj.n_ctx() <= len(token_ids):
            raise Exception(f"Attempted to use a context length of {len(token_ids)} tokens, but this LlamaCpp model is only configured to support up to {self.model_obj.n_ctx()}!")

        self._cache_state["cache_token_ids"] = token_ids.copy()

        # clear obsolete parts of kv cache
        llama_cpp.llama_kv_cache_seq_rm(self.model_obj.ctx, -1, num_cached, -1)

        # eval the model
        n_batch = self.model_obj.n_batch
        batch = llama_cpp.llama_batch_init(
            n_tokens=n_batch, 
            embd=0, 
            n_seq_max=self.model_obj.n_ctx()
        )
        if batch is None:
            raise Exception("call to llama_cpp.llama_batch_init returned NULL.")
        try:
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
        finally:
            llama_cpp.llama_batch_free(batch)

        self._cache_state["logits"] = logits

        return logits
    
class LlamaCppChat(LlamaCpp, Chat):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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
