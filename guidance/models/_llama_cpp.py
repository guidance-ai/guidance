import atexit
import logging
import operator
import os
import sys
from itertools import takewhile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Sequence
import ctypes

import numpy as np

from .._schema import GenToken, GenTokenExtra
from .._utils import normalize_notebook_stdout_stderr, softmax
from ._base import Model
from ._engine import Engine, EngineClient, EngineState, Tokenizer

try:
    import llama_cpp

    is_llama_cpp = True
except ModuleNotFoundError:
    is_llama_cpp = False

if TYPE_CHECKING:
    from llama_cpp.llama_tokenizer import LlamaTokenizer
    from llama_cpp.llama import Llama
    from llama_cpp.llama_chat_format import Jinja2ChatFormatter

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
    def __init__(self, model_obj, chat_template=None):
        self._model_obj = model_obj

        self._sentinel_bytes = "\x02".encode()
        self._sentinel_tokens = self._model_obj.tokenize(
            self._sentinel_bytes, add_bos=False, special=True
        )

        tokenizer = llama_cpp.LlamaTokenizer(model_obj)
        vocab = llama_cpp.llama_model_get_vocab(model_obj.model)
        if vocab is None:
            raise Exception("call to llama_cpp.llama_model_get_vocab returned NULL.")

        if not hasattr(tokenizer, "llama"):
            tokenizer.llama = tokenizer._model

        # get the bytes strings for all the tokens
        tokens = []
        special_token_ids = []
        for i in range(tokenizer.llama.n_vocab()):
            tok_attrs = tokenizer.llama.token_get_attr(i)
            if tok_attrs & llama_cpp.LLAMA_TOKEN_ATTR_CONTROL:
                special_token_ids.append(i)
            tok = detokenize(tokenizer, [i], special=True, size=256)
            tokens.append(tok)

        # Chat Template logic
        if chat_template is None:
            if (
                hasattr(self._model_obj, "metadata")
                and "tokenizer.chat_template" in self._model_obj.metadata
            ):
                chat_template = self._model_obj.metadata["tokenizer.chat_template"]

        super().__init__(
            tokens, chat_template, tokenizer.llama.token_bos(), tokenizer.llama.token_eos(), special_token_ids
        )

    def encode(self, byte_string: bytes) -> list[int]:
        # Workaround for the LlamaCpp prepending spaces on encoding
        raw_tokens = self._model_obj.tokenize(
            self._sentinel_bytes + byte_string, add_bos=False, special=True
        )
        assert raw_tokens[: len(self._sentinel_tokens)] == self._sentinel_tokens
        return raw_tokens[len(self._sentinel_tokens) :]


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
                logger.warn(
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

        super().__init__(LlamaCppTokenizer(self.model_obj, chat_template=chat_template),
                         compute_log_probs=compute_log_probs, enable_backtrack=enable_backtrack,
                         enable_ff_tokens=enable_ff_tokens, enable_monitoring=enable_monitoring, **kwargs)

        self._n_vocab = len(self.tokenizer.tokens)

    def get_logits(self, token_ids):
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
            if num_cached == len(cache_token_ids):
                return self._cached_logits
            num_cached = (
                num_cached - 1
            )  # llama_cpp doesn't like it when we pass in 0 new tokens, so re-input one

        # make sure we don't run off the end of the model's context
        if self.model_obj.n_ctx() <= len(token_ids):
            raise Exception(
                f"Attempted to use a context length of {len(token_ids)} tokens, but this LlamaCpp model is only configured to support up to {self.model_obj.n_ctx()}!"
            )

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
            self.metrics.engine_input_tokens += n_tokens
            if ret != 0:
                raise Exception(f"Call to llama_cpp.llama_decode returned {ret}.")

        self.metrics.engine_output_tokens += 1

        # get the logits
        logits = llama_cpp.llama_get_logits(self.model_obj.ctx)
        if llama_cpp.__version__ < "0.2.58":
            logits = logits[(n_tokens - 1) * self._n_vocab : n_tokens * self._n_vocab]
        logits = np.ctypeslib.as_array(logits, shape=(self._n_vocab,)).copy()

        self._cached_logits = logits

        return logits

    def get_per_token_topk_probs(
        self, token_ids: list[int], top_k: int = 5
    ) -> list[GenTokenExtra]:
        if len(token_ids) == 0:
            return []

        # clear kv-cache
        llama_cpp.llama_kv_cache_seq_rm(self.model_obj.ctx, -1, 0, -1)

        # create new context
        context = self._context
        n_batch = self.model_obj.n_batch
        batch = context.batch

        logits_batch = []
        for i in range(0, len(token_ids), n_batch):
            n_tokens = min(i + n_batch, len(token_ids)) - i
            batch.n_tokens = n_tokens

            for j in range(n_tokens):
                batch.token[j] = token_ids[i + j]
                batch.pos[j] = i + j
                batch.seq_id[j][0] = 0
                batch.n_seq_id[j] = 1
                batch.logits[j] = True

            ret = llama_cpp.llama_decode(self.model_obj.ctx, batch)
            if ret != 0:
                raise Exception(f"Call to llama_cpp.llama_decode returned {ret}.")

            # get all the logits
            if llama_cpp.__version__ < "0.2.58":
                logits = llama_cpp.llama_get_logits(self.model_obj.ctx)
                logits = logits.reshape((-1, self._n_vocab))
                logits_batch.extend(logits)
            else:
                for j in range(n_tokens):
                    logits = llama_cpp.llama_get_logits_ith(self.model_obj.ctx, j)
                    logits = np.ctypeslib.as_array(logits, shape=(self._n_vocab,)).copy()
                    logits_batch.append(logits)

        # process each token's logits
        text_sequence = []

        # add 1st token
        _bytes = self.tokenizer.decode([token_ids[0]])
        try:
            _text = _bytes.decode("utf-8")
        except Exception as e:
            _text = str(_bytes)
            print(f"Failed to decode token: {token_ids[0]}, error: {e}, _bytes: {str(_bytes)}")
        text_sequence.append(
            GenTokenExtra(
                token_id=token_ids[0],
                prob=1.0,
                text=_text,
                top_k=[GenToken(token_id=token_ids[0], prob=1.0, text=_text)],
            )
        )

        for token_id, logits in zip(token_ids[1:], logits_batch[:-1]):
            _probs = softmax(logits)

            # get the top k indices
            top_k_ids, top_k_probs = self._get_top_k(_probs.copy(), top_k, ascending=False)
            if token_id not in top_k_ids:
                top_k_ids = np.append(top_k_ids, token_id)
                top_k_probs = np.append(top_k_probs, _probs[token_id])

            top_k_list = []
            for _token_id, _prob in zip(top_k_ids, top_k_probs):
                _text = ""
                try:
                    _text = self.tokenizer.decode([_token_id]).decode("utf-8")
                except Exception as e:
                    _bytes = self.tokenizer.decode([_token_id])
                    _text = str(_bytes)
                    print(
                        f"Failed to decode token: {_token_id}, error: {e}, _bytes: {str(_bytes)}"
                    )
                top_k_list.append(GenToken(token_id=_token_id, prob=_prob, text=_text))

            text_sequence.append(
                GenTokenExtra(
                    token_id=token_id,
                    prob=_probs[token_id],
                    text=self.tokenizer.decode([token_id]).decode("utf-8"),
                    top_k=top_k_list,
                )
            )

        return text_sequence

    def _get_top_k(self, probs: np.ndarray, k: int, axis: int = None, ascending: bool = True):
        if not ascending:
            probs *= -1
        ind = np.argpartition(probs, k, axis=axis)
        ind = np.take(ind, np.arange(k), axis=axis)  # k non-sorted indices
        probs = np.take_along_axis(probs, ind, axis=axis)  # k non-sorted values

        # sort within k elements
        ind_part = np.argsort(probs, axis=axis)
        ind = np.take_along_axis(ind, ind_part, axis=axis)
        if not ascending:
            probs *= -1
        val = np.take_along_axis(probs, ind_part, axis=axis)
        return ind, val


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
        state = EngineState()
        client = EngineClient(engine)
        super().__init__(client=client, state=state, echo=echo)
