import atexit
import logging
import operator
import os
import sys
from itertools import takewhile
from pathlib import Path
from typing import TYPE_CHECKING, Union
import ctypes

import numpy as np

from .._schema import GenToken, GenTokenExtra
from .._utils import normalize_notebook_stdout_stderr, softmax
from ..chat import ChatTemplate
from ._base import Model
from ._engine import Engine, EngineInterpreter, Tokenizer

try:
    import llama_cpp

    is_llama_cpp = True
except ModuleNotFoundError:
    is_llama_cpp = False
else:
    import llguidance.llamacpp

if TYPE_CHECKING:
    from llama_cpp.llama_tokenizer import LlamaTokenizer
    from llama_cpp.llama import Llama

logger = logging.getLogger(__name__)

shutdown_none = True


def set_shutdown_flag():
    global shutdown_none
    # python can set anything to None at shutdown, so use None
    shutdown_none = None


atexit.register(set_shutdown_flag)


class _LlamaBatchContext:
    def __init__(self, n_batch, n_ctx):
        self._llama_batch_free = llama_cpp.llama_batch_free
        self.batch = llama_cpp.llama_batch_init(n_batch, 0, n_ctx)
        if self.batch is None:
            raise Exception("call to llama_cpp.llama_batch_init returned NULL.")

    def __del__(self):
        if shutdown_none is not None:
            llama_batch_free = getattr(self, "_llama_batch_free", None)
            batch = getattr(self, "batch", None)
            if batch is not None and llama_batch_free is not None:
                self._llama_batch_free = None
                self.batch = None
                llama_batch_free(batch)

def detokenize(tokenizer: "LlamaTokenizer", tokens: list[int], special: bool, size: int) -> bytes:
    """Re-implementation of llama_cpp.LLamaTokenizer.detokenize that ditches the hard-coded size=32"""
    output = b""
    buffer = ctypes.create_string_buffer(size)
    for token in tokens:
        n = llama_cpp.llama_token_to_piece(
            tokenizer._model.vocab,
            llama_cpp.llama_token(token),
            buffer,
            size,
            0,
            special
        )
        if n < 0:
            raise ValueError(f"Error writing token {token} to buffer of size {size}. Error: {n}")
        assert n <= size
        output += bytes(buffer[:n])
    # NOTE: Llama1 models automatically added a space at the start of the prompt
    # this line removes a leading space if the first token is a beginning of sentence token
    return (
        output[1:]
        if len(tokens) > 0 and tokens[0] == tokenizer._model.token_bos() and output[0:1] == b" "
        else output
        )

class LlamaCppTokenizer(Tokenizer):
    def __init__(self, model_obj: "Llama", chat_template: Union[str, ChatTemplate, None] = None):
        self._model_obj = model_obj

        vocab = llama_cpp.llama_model_get_vocab(model_obj.model)
        if vocab is None:
            raise Exception("call to llama_cpp.llama_model_get_vocab returned NULL.")
        ll_tokenizer = llguidance.llamacpp.lltokenizer_from_vocab(vocab)

        # Chat Template logic
        if chat_template is None:
            if (
                hasattr(self._model_obj, "metadata")
                and "tokenizer.chat_template" in self._model_obj.metadata
            ):
                chat_template = self._model_obj.metadata["tokenizer.chat_template"]

        super().__init__(
            ll_tokenizer=ll_tokenizer,
            chat_template=chat_template,
            bos_token_id=model_obj.token_bos()
        )


class LlamaCppEngine(Engine):
    """The core class that runs inference using llama.cpp."""

    def __init__(
        self,
        model,
        compute_log_probs,
        chat_template=None,
        enable_backtrack=True,
        enable_ff_tokens=True,
        enable_monitoring=True,
        **kwargs,
    ):
        if not is_llama_cpp:
            raise Exception(
                "Please install llama-cpp-python with `pip install llama-cpp-python` in order to use guidance.models.LlamaCpp!"
            )

        if isinstance(model, Path):
            model = str(model)
        if model is None or isinstance(model, str) and len(model.strip()) == 0:
            model = os.environ.get("LLAMA_CPP_MODEL", "")
            if len(model.strip()) == 0:
                try:
                    with open(os.path.expanduser("~/.llama_cpp_model"), "r") as file:
                        model = file.read().replace("\n", "")
                except:
                    pass
                if len(model.strip()) == 0:
                    raise ValueError(
                        "If model is None then a model file must be specified in either the LLAMA_CPP_MODEL environment variable or in the ~/.llama_cpp_model file."
                    )

        if isinstance(model, str):
            self.model = model
            if "verbose" not in kwargs:
                kwargs["verbose"] = False

            # patch over https://github.com/abetlen/llama-cpp-python/issues/729
            try:
                sys.stdout.fileno()
            except:
                logger.warning(
                    "Cannot use verbose=True in this context (probably CoLab). See https://github.com/abetlen/llama-cpp-python/issues/729"
                )
                kwargs["verbose"] = True  # llama-cpp-python can't hide output in this case

            with normalize_notebook_stdout_stderr():
                self.model_obj = llama_cpp.Llama(model_path=model, logits_all=True, **kwargs)
        elif isinstance(model, llama_cpp.Llama):
            self.model = model.__class__.__name__
            self.model_obj = model
        else:
            raise TypeError("model must be None, a file path string, or a llama_cpp.Llama object.")

        self._context = _LlamaBatchContext(self.model_obj.n_batch, self.model_obj.n_ctx())
        self._cache_token_ids = []
        self._n_vocab = self.model_obj.n_vocab()
        self._cached_logits = None

        super().__init__(LlamaCppTokenizer(self.model_obj, chat_template=chat_template),
                         compute_log_probs=compute_log_probs, enable_backtrack=enable_backtrack,
                         enable_ff_tokens=enable_ff_tokens, enable_monitoring=enable_monitoring, **kwargs)

    def get_logits(self, token_ids, full_sequence=False):
        """Computes the logits for the given token state.

        This overrides a method from the LocalEngine class that is used to get
        inference results from the model.
        """

        if len(token_ids) == 0:
            raise ValueError("token_ids must contain some tokens.")

        # check what we have already cached
        cache_token_ids = self._cache_token_ids
        num_cached = sum(takewhile(operator.truth, map(operator.eq, token_ids, cache_token_ids)))
        if num_cached == len(token_ids):
            # TODO: do something similar for Transformers
            if full_sequence:
                return self._cached_logits[:num_cached, :]
            else:
                return self._cached_logits[num_cached - 1, :]

        # make sure we don't run off the end of the model's context
        if self.model_obj.n_ctx() <= len(token_ids):
            raise Exception(
                f"Attempted to use a context length of {len(token_ids)} tokens, but this LlamaCpp model is only configured to support up to {self.model_obj.n_ctx()}!"
            )

        self._cache_token_ids = token_ids.copy()
        if self._cached_logits is not None and len(self._cached_logits) > num_cached:
            self._cached_logits = self._cached_logits[:num_cached, :]

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
                batch.logits[j] = True

            ret = llama_cpp.llama_decode(self.model_obj.ctx, batch)
            self.metrics.engine_input_tokens += n_tokens
            if ret != 0:
                raise Exception(f"Call to llama_cpp.llama_decode returned {ret}.")

        self.metrics.engine_output_tokens += 1

        # get the logits
        llama_logits = llama_cpp.llama_get_logits(self.model_obj.ctx)
        logits = np.ctypeslib.as_array(
            llama_logits,
            shape=(
                len(token_ids) - num_cached,
                self._n_vocab,
            ),
        )

        if self._cached_logits is None:
            self._cached_logits = logits
        else:
            self._cached_logits = np.concatenate(
                (self._cached_logits, logits),
                axis=0
            )

        if full_sequence:
            return self._cached_logits
        else:
            return self._cached_logits[-1, :]

class LlamaCpp(Model):
    def __init__(
        self,
        model=None,
        echo=True,
        compute_log_probs=False,
        api_key=None,
        chat_template=None,
        enable_backtrack=True,
        enable_ff_tokens=True,
        enable_monitoring=True,
        **llama_cpp_kwargs,
    ):
        """Build a new LlamaCpp model object that represents a model in a given state."""

        engine = LlamaCppEngine(
            model,
            compute_log_probs=compute_log_probs,
            chat_template=chat_template,
            enable_backtrack=enable_backtrack,
            enable_ff_tokens=enable_ff_tokens,
            enable_monitoring=enable_monitoring,
            **llama_cpp_kwargs,
        )
        interpreter = EngineInterpreter(engine)
        super().__init__(interpreter=interpreter, echo=echo)
