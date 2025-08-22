import ast
import asyncio
import functools
import http
import inspect
import json
import logging
import pathlib
import re
import sys
import textwrap
import types
import urllib
import weakref
from typing import TYPE_CHECKING, Optional, Union, cast

import numpy as np
import pydantic

if TYPE_CHECKING:
    from ._ast import GrammarNode, JoinNode
    from .models._engine import Tokenizer
    from .types import SamplingParams

LLG_SPECIAL_TOKEN_PAT = re.compile(rb"\xff\[([0-9]+)\]")


logger = logging.getLogger(__name__)


def bytes_from(src: Union[str, pathlib.Path, bytes], allow_local: bool) -> bytes:
    if isinstance(src, str) and re.match(r"[^:/]+://", src):
        with urllib.request.urlopen(src) as response:
            response = cast(http.client.HTTPResponse, response)
            bytes_data = response.read()

    # ...from a local path
    elif allow_local and (isinstance(src, str) or isinstance(src, pathlib.Path)):
        with open(src, "rb") as f:
            bytes_data = f.read()

    # ...from audio file bytes
    elif isinstance(src, bytes):
        bytes_data = src

    else:
        raise Exception(f"Unable to load bytes from {src}!")

    return bytes_data


class _Rewrite(ast.NodeTransformer):
    def __init__(self, source_lines):
        self.source_lines = source_lines
        self.indentation = [None for _ in source_lines]

    def visit_JoinedStr(self, node):
        for value in node.values:
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                self._dedent_constant(value, node.lineno)
        return node

    def visit_Constant(self, node):
        if isinstance(node.value, str) and "\n" in node.value:
            self._dedent_constant(node, node.lineno)
        return node

    def _dedent_constant(self, node, lineno):
        start_lineno = lineno - 1
        start_line = self.source_lines[start_lineno]
        indent = len(start_line) - len(start_line.lstrip())

        if indent > 0:
            new_lines = []
            for line in node.value.split("\n"):
                if line.startswith(" " * indent):
                    new_lines.append(line[indent:])
                else:
                    new_lines.append(line)
            node.value = "\n".join(new_lines)


class normalize_notebook_stdout_stderr:
    """Remaps stdout and stderr back to their normal selves from what ipykernel did to them.

    Based on: https://github.com/ipython/ipykernel/issues/795
    """

    def __enter__(self):
        normal_stdout = sys.__stdout__.fileno()
        self.restore_stdout = None
        if getattr(sys.stdout, "_original_stdstream_copy", normal_stdout) != normal_stdout:
            self.restore_stdout = sys.stdout._original_stdstream_copy
            sys.stdout._original_stdstream_copy = normal_stdout

        normal_stderr = sys.__stderr__.fileno()
        self.restore_stderr = None
        if getattr(sys.stderr, "_original_stdstream_copy", normal_stderr) != normal_stderr:
            self.restore_stderr = sys.stderr._original_stdstream_copy
            sys.stderr._original_stdstream_copy = normal_stderr

    def __exit__(self, exc_type, exc_value, traceback):
        if self.restore_stdout is not None:
            sys.stderr._original_stdstream_copy = self.restore_stdout
        if self.restore_stderr is not None:
            sys.stderr._original_stdstream_copy = self.restore_stderr


def strip_multiline_string_indents(f):
    source = textwrap.dedent(inspect.getsource(f))
    blanks = "\n" * f.__code__.co_firstlineno  # pad the source so the lines in the file line up for the debugger
    source = blanks + "\n".join(source.splitlines()[1:])  # remove the decorator first line.

    # define the external closure variables so f.__closure__ will match our recompiled version
    if len(f.__code__.co_freevars) > 0:
        raise Exception(
            "You currently must use @guidance(dedent=False) for closure functions (function nested within other functions that reference the outer functions variables)!"
        )
    #       lines = source.split("\n")
    #       lines[0] = "def __outer__closure_wrap():"
    #       lines[1] = (
    #           "    "
    #           + ",".join(f.__code__.co_freevars)
    #           + " = "
    #           + ",".join("None" for _ in f.__code__.co_freevars)
    #       )
    #       source = "    \n".join(
    #           lines
    #       )  # TODO: this does not quite work because new_code_obj is now the __outer__closure_wrap() function...could be fixed with work...

    old_code_obj = f.__code__
    old_ast = ast.parse(source)
    r = _Rewrite(source.split("\n"))
    new_ast = r.visit(old_ast)
    new_code_obj = compile(new_ast, old_code_obj.co_filename, "exec")

    # find the code block
    for i in range(len(new_code_obj.co_consts)):
        if str(type(new_code_obj.co_consts[i])) == "<class 'code'>":
            break

    # create a new function based on the modified code
    new_f = types.FunctionType(
        new_code_obj.co_consts[i],
        f.__globals__,
        name=f.__name__,
        argdefs=f.__defaults__,
        closure=f.__closure__,
    )
    new_f.__kwdefaults__ = f.__kwdefaults__
    new_f.__qualname__ = f.__qualname__
    new_f.__annotations__ = f.__annotations__
    new_f.__doc__ = f.__doc__
    new_f.__module__ = f.__module__
    return new_f


def make_weak_bound_method(f, instance):
    instance_ref = weakref.ref(instance)
    instance_repr = repr(instance)

    @functools.wraps(f)  # ish
    def weak_bound_f(*args, **kwargs):
        instance = instance_ref()
        if instance is None:
            raise ReferenceError(f"Lost reference to {instance_repr} and cannot bind {f} to it.")
        method = types.MethodType(f, instance)
        return method(*args, **kwargs)

    # remove the first argument from the wrapped function since it is now bound
    weak_bound_f.__signature__ = signature_pop(inspect.signature(f), 0)
    return weak_bound_f


def signature_pop(signature, index):
    params = list(signature.parameters.values())
    params.pop(index)
    return signature.replace(parameters=params)


class JupyterComm:
    def __init__(self, target_id, ipython_handle, callback=None, on_open=None, mode="register"):
        from ipykernel.comm import Comm

        self.target_name = "guidance_interface_target_" + target_id
        # print("TARGET NAME", self.target_name)
        self.callback = callback
        self.jcomm = None
        self.ipython_handle = ipython_handle
        self.addd = 1
        self.send_queue = asyncio.Queue()
        self.open_event = asyncio.Event()
        self.is_open = False
        asyncio.get_event_loop().create_task(self._send_loop())
        if mode == "register":
            # log("REGISTERING", self.target_name)
            # asyncio.get_event_loop().create_task(self._register())
            def comm_opened(comm, open_msg):
                # log("OPENED")
                self.addd = 2
                self.jcomm = comm
                self.is_open = True
                self.jcomm.on_msg(self._fire_callback)
                self.open_event.set()
                self._fire_callback({"content": {"data": {"event": "opened"}}})

            self.ipython_handle.kernel.comm_manager.register_target(self.target_name, comm_opened)
            # get_ipython().kernel.comm_manager.register_target(self.target_name, comm_opened) # noqa: F821
        elif mode == "open":
            # log("OPENING", self.target_name)
            self.jcomm = Comm(target_name=self.target_name)
            self.jcomm.on_msg(self._fire_callback)
            # self._fire_callback({"content": {"data": "opened"}})
        else:
            raise Exception("Passed mode must be either 'open' or 'register'!")

    def clear_send_queue(self):
        while not self.send_queue.empty():
            self.send_queue.get_nowait()
            self.send_queue.task_done()

    def _fire_callback(self, msg):
        self.callback(msg["content"]["data"])

    def send(self, data):
        self.send_queue.put_nowait(data)

    async def _send_loop(self):
        while True:
            # log("SENDING_LOOP")
            if self.jcomm is None:
                self.open_event.clear()
                await self.open_event.wait()
            data = await self.send_queue.get()
            # log("SENDING_LOOP got one!")
            self.jcomm.send({"data": json.dumps(data)})

    # async def _waiting_send(self, data):
    #     #log("SENDING", self.jcomm, data)

    #     # await the open event if needed
    #     if self.jcomm is None:
    #         self.open_event.clear()
    #         await self.open_event.wait()
    #     #log("SENDING_now", self.jcomm, data)
    #     self.jcomm.send({"data": json.dumps(data)}) # we encode the JSON so iPython doesn't mess it up


# https://stackoverflow.com/questions/15411967/how-can-i-check-if-code-is-executed-in-the-ipython-notebook
def is_interactive():
    import __main__ as main

    return not hasattr(main, "__file__")


def log_softmax(array: np.ndarray, axis: int = -1) -> np.ndarray:
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.log_softmax.html
    array_maxs: np.ndarray = np.amax(array, axis=axis, keepdims=True)
    if array_maxs.ndim > 0:
        array_maxs[~np.isfinite(array_maxs)] = 0
    elif not np.isfinite(array_maxs):
        array_maxs = np.zeros(array_maxs.shape)
    subtract_maxs = array - array_maxs
    exp = np.exp(subtract_maxs)
    # suppress warnings about log of zero
    with np.errstate(divide="ignore"):
        summed = np.sum(exp, axis=axis, keepdims=True)
        out = np.log(summed)
    return subtract_maxs - out


def softmax(array: np.ndarray, axis: int = -1) -> np.ndarray:
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.softmax.html
    array_maxs = np.amax(array, axis=axis, keepdims=True)
    exp_x_shifted = np.exp(array - array_maxs)
    return exp_x_shifted / np.sum(exp_x_shifted, axis=axis, keepdims=True)


def pydantic_no_default_repr(obj: pydantic.BaseModel, target_fields=None):
    if target_fields is None:
        records = (
            f"{getattr(obj, name)!r}"
            for name, field in obj.__class__.model_fields.items()
            if getattr(obj, name) != field.default and not field.exclude
        )
    else:
        records = (
            f"{getattr(obj, name)!r}"
            for name, field in obj.__class__.model_fields.items()
            if getattr(obj, name) != field.default and not field.exclude and name in target_fields
        )
    out = f"{type(obj).__name__}:{':'.join(records)}"
    return out


def pydantic_no_default_str(obj: pydantic.BaseModel, target_fields=None):
    if target_fields is None:
        records = (
            f"{getattr(obj, name)!s}"
            for name, field in obj.__class__.model_fields.items()
            if getattr(obj, name) != field.default and not field.exclude
        )
    else:
        records = (
            f"{getattr(obj, name)!s}"
            for name, field in obj.__class__.model_fields.items()
            if getattr(obj, name) != field.default and not field.exclude and name in target_fields
        )
    out = "\n".join(records)
    return out


def log_init(s: str):
    logger.debug(f"INIT:{s}")
    pass


def log_copy(s: str):
    logger.debug(f"COPY:{s}")
    pass


def log_cleanup(s: str):
    logger.debug(f"CLEANUP:{s}")
    pass


def to_utf8_or_bytes_string(_bytes: bytes) -> str:
    """
    Converts a byte sequence to a UTF-8 string if possible. If the byte sequence
    cannot be decoded as UTF-8, it returns the string representation of the byte sequence.

    Parameters
    ----------
    _bytes : bytes
        The byte sequence to be converted.

    Returns
    -------
    str
        The decoded UTF-8 string or the string representation of the byte sequence
        if UTF-8 decoding fails.
    """
    try:
        return _bytes.decode("utf-8")
    except UnicodeDecodeError:
        return str(_bytes)


def apply_repetition_penalty(input_ids: list[int], logits: np.ndarray, sampling_params: Optional["SamplingParams"]):
    if sampling_params is None:
        return logits

    penalty = sampling_params.get("repetition_penalty", None)
    if penalty is None or penalty <= 0:
        return logits

    # Gather the logits for the input_ids
    # input_ids: shape (1, seq_len), scores: shape (1, vocab_size)
    # We want to get the logits for each token in input_ids from scores
    # For each token in input_ids, get its logit from scores
    # This is equivalent to: score = scores[0, input_ids[0, :]]
    input_ids = np.asarray(input_ids)

    score = np.take_along_axis(logits, input_ids, axis=0)
    # Apply repetition penalty
    # If score < 0, multiply by penalty; else, divide by penalty
    score_processed = np.where(score < 0, score * penalty, score / penalty)
    # Scatter the processed scores back into the scores array
    # For each position in input_ids, set scores[0, input_ids[0, i]] = score_processed[0, i]
    scores_processed = logits.copy()
    np.put_along_axis(scores_processed, input_ids, score_processed, axis=0)
    return scores_processed


def apply_top_k_only(logits: np.ndarray, k: int) -> np.ndarray:
    if k <= 0:
        return logits

    indices_to_remove = logits.argpartition(-k)[:-k]
    logits[indices_to_remove] = -float("inf")
    return logits


def apply_min_p_filter(logits: np.ndarray, sampling_params: Optional["SamplingParams"]) -> np.ndarray:
    if sampling_params is None:
        return logits

    min_p = sampling_params.get("min_p", None)
    if min_p is None:
        return logits

    probs = softmax(logits, axis=-1)
    top_probs = np.max(probs, axis=-1)
    scaled_min_p = min_p * top_probs

    indices_to_remove = probs < scaled_min_p
    logits[indices_to_remove] = -np.inf
    return logits


def apply_top_k_and_top_p_filter(logits: np.ndarray, sampling_params: Optional["SamplingParams"]) -> np.ndarray:
    if sampling_params is None:
        return logits

    top_p = sampling_params.get("top_p", None)
    top_k = sampling_params.get("top_k", None)

    if top_k is None and top_p is None:
        # No filtering, return logits as is
        return logits

    if top_k is not None and top_p is None:
        return apply_top_k_only(logits, top_k)

    # try our best to sort logits one time only
    sorted_logits = logits.argsort()
    if top_k is not None and top_k > 0:
        indices_to_remove = sorted_logits[:-top_k]
        logits[indices_to_remove] = -float("inf")

    if top_p is not None and top_p >= 0:
        sorted_indices = sorted_logits[::-1]
        sorted_logits = logits[sorted_indices]
        probs = softmax(sorted_logits)
        cumulative_probs = np.cumsum(probs)
        indices_to_remove = cumulative_probs > top_p
        if np.any(indices_to_remove):
            # +1 to keep the first token that exceeds the threshold
            first_to_remove = np.argmax(indices_to_remove) + 1
            # make sure we always keep at least one token
            sorted_indices_to_remove = sorted_indices[max(1, first_to_remove) :]
            logits[sorted_indices_to_remove] = -float("inf")

    return logits


def partial_decode(data: bytes) -> tuple[str, bytes]:
    """Decode a byte string into a valid UTF-8 string, handling partial characters.
    If the byte string contains a partial character, it will return the valid part and the remaining bytes."""
    try:
        return (data.decode("utf-8"), b"")
    except UnicodeDecodeError as e:
        valid_part = data[: e.start].decode("utf-8")
        delayed_part = data[e.start :]
    return (valid_part, delayed_part)


def recode_special_tokens(tokenizer: "Tokenizer", data: bytes) -> bytes:
    """Recode a byte string with special tokens in llguidance format to their actual byte representation."""
    return LLG_SPECIAL_TOKEN_PAT.sub(lambda m: tokenizer.decode([int(m.group(1).decode("utf-8"))]), data)


def text_to_grammar(tokenizer: "Tokenizer", text: str) -> "JoinNode":
    """
    Convert a text string into a GrammarNode that can be used in the grammar.
    This is useful for converting static text into a grammar node that can be processed by the engine.
    """
    from guidance._ast import JoinNode, LiteralNode, SpecialToken

    grammar_bits: list["GrammarNode"] = []
    delayed_bytes = b""
    for token_id in tokenizer.encode(text.encode("utf-8"), parse_special=True):
        if tokenizer.is_special_token(token_id):
            assert not delayed_bytes, "Should not have any delayed bytes when encountering a special token"
            grammar_bits.append(SpecialToken(id=token_id))
        else:
            new_bytes = tokenizer.decode([token_id])
            new_text, delayed_bytes = partial_decode(delayed_bytes + new_bytes)
            if new_text:
                grammar_bits.append(LiteralNode(new_text))
    assert not delayed_bytes, "Should not have any delayed bytes left after processing the text"
    if len(grammar_bits) == 1:
        return grammar_bits[0]
    return JoinNode(tuple(grammar_bits))
