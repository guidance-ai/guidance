import base64
import copy
import html
import logging
import queue
import re
import textwrap
import threading
import time
import warnings
from typing import Union


from pprint import pprint
from typing import Dict, TYPE_CHECKING


import numpy as np

try:
    from IPython.display import clear_output, display, HTML

    ipython_is_imported = True
except ImportError:
    ipython_is_imported = False
try:
    import torch

    torch_is_imported = True
except ImportError:
    torch_is_imported = False


logger = logging.getLogger(__name__)
try:
    from .. import cpp  # type: ignore[attr-defined]
except ImportError:
    logger.warn(
        "Failed to load guidance.cpp, falling back to Python mirror implementations..."
    )
    from .. import _cpp as cpp
from ._guidance_engine_metrics import GuidanceEngineMetrics
from .._utils import softmax, CaptureEvents
from .._parser import EarleyCommitParser, Parser
from .._grammar import (
    GrammarFunction,
    string,
    _call_pool,
    _tag_pattern,
    Null,
    replace_model_variables,
    unreplace_model_variables,
    select,
)

from .. import _serialization_pb2
from ..chat import load_template_class

from ._tokenizer import Tokenizer

if TYPE_CHECKING:
    from ..library._block import ContextBlock

# define some constants we will reuse many times
_null_grammar = string("")
format_pattern = re.compile(r"<\|\|_.*?_\|\|>", flags=re.DOTALL)
nodisp_pattern = re.compile(
    r"&lt;\|\|_#NODISP_\|\|&gt;.*?&lt;\|\|_/NODISP_\|\|&gt;", flags=re.DOTALL
)
html_pattern = re.compile(r"&lt;\|\|_html:(.*?)_\|\|&gt;", flags=re.DOTALL)
image_pattern = re.compile(r"&lt;\|_image:(.*?)\|&gt;")




class EngineCallResponse:
    new_bytes: bytes
    is_generated: bool
    new_bytes_prob: float
    capture_groups: dict
    capture_group_log_probs: dict
    new_token_count: int

    def __init__(
        self,
        new_bytes,
        is_generated,
        new_bytes_prob,
        capture_groups,
        capture_group_log_probs,
        new_token_count,
    ):
        self.new_bytes = new_bytes
        self.is_generated = is_generated
        self.new_bytes_prob = new_bytes_prob
        self.capture_groups = capture_groups
        self.capture_group_log_probs = capture_group_log_probs
        self.new_token_count = new_token_count

    def _to_proto(self):
        """Converts an EngineCallResponse object to its Protobuf representation.

        Returns:
            engine_response_pb2.EngineCallResponse: The Protobuf equivalent of this object.
        """
        groups = {}
        group_log_probs = {}

        def to_protobuf_value(v: Union[str, bytes, float, list]) -> _serialization_pb2.Value:
            """Convert Python values to Protobuf Value messages."""
            value = _serialization_pb2.Value()
            if isinstance(v, str):
                value.string_value = v
            elif isinstance(v, bytes):
                value.bytes_value = v
            elif isinstance(v, float):
                value.float_value = v
            elif isinstance(v, list):
                for item in v:
                    value.list_value.values.append(to_protobuf_value(item))
            else:
                raise TypeError(f"Unsupported type: {type(v)}")
            return value

        for k, v in self.capture_groups.items():
            groups[k] = to_protobuf_value(v)

        for k, v in self.capture_group_log_probs.items():
            group_log_probs[k] = to_protobuf_value(v)

        return _serialization_pb2.EngineCallResponse(
            new_bytes=self.new_bytes,
            is_generated=self.is_generated,
            new_bytes_prob=self.new_bytes_prob,
            capture_groups=groups,
            capture_group_log_probs=group_log_probs,
            new_token_count=self.new_token_count,
        )

    def encode(self, charset):
        """Used to support FastAPI encoding of EngineCallResponse objects."""
        return self.serialize()

    def serialize(self):
        proto = self._to_proto()
        return proto.SerializeToString()

    @staticmethod
    def deserialize(byte_data):
        proto = _serialization_pb2.EngineCallResponse()
        proto.ParseFromString(byte_data)

        def from_protobuf_value(value: _serialization_pb2.Value) -> Union[str, bytes, float, list]:
            """Convert Protobuf Value message to Python values"""
            if value.HasField("string_value"):
                return value.string_value
            elif value.HasField("bytes_value"):
                return value.bytes_value
            elif value.HasField("float_value"):
                return value.float_value
            elif value.HasField("list_value"):
                return [from_protobuf_value(item) for item in value.list_value.values]
            else:
                raise ValueError("Protobuf Value message has no recognized field set")

        groups = {}
        for k, v in proto.capture_groups.items():
            groups[k] = from_protobuf_value(v)

        group_log_probs = {}
        for k, v in proto.capture_group_log_probs.items():
            group_log_probs[k] = from_protobuf_value(v)

        return EngineCallResponse(
            new_bytes=proto.new_bytes,
            is_generated=proto.is_generated,
            new_bytes_prob=proto.new_bytes_prob,
            capture_groups=groups,
            capture_group_log_probs=group_log_probs,
            new_token_count=proto.new_token_count,
        )


class Engine:
    """The engine owns the inference computation and is used/created by the Model class.

    Engine objects represent the expensive parts of inference. While Model objects are cheap and do not
    need to know about the tokenizer or the model parameters, Engine objects know about both. Many
    Model objects can reference a single Engine object. Engine objects can also be hidden behind a
    Server so a single server can serve many clients' model objects through a single Engine object.
    """

    def __init__(self, tokenizer: Tokenizer, compute_log_probs=False):
        self.tokenizer = tokenizer
        self.compute_log_probs = compute_log_probs

        # build a prefix tree of the tokens
        self._token_trie = cpp.ByteTrie(
            self.tokenizer.tokens, np.arange(len(self.tokenizer.tokens))
        )
        self._token_trie.match = True
        self._token_trie.match_version = 0

        self.metrics = GuidanceEngineMetrics()

    def get_chat_template(self): # TODO [HN]: Add more logic here...should we instantiate class here? do we even need to?
        return self.tokenizer.chat_template() # Instantiate the class before returning to client for now
    
    def reset_metrics(self):
        self.metrics = GuidanceEngineMetrics()

    def start(self, parser, grammar, ensure_bos_token=True):
        """Start processing parser state executed through the grammar.

        Parameters
        ----------
        parser : str or Parser
            This is represents the current state of a guidance parser that will be extended
            using the passed grammar. If a string is given then we assume the previous parser
            state is just a fixed string prompt, if a full Parser is given then we extend that
            parser by appending the new grammar to the parser's current grammar and then
            inferencing the model. (TODO: implement full parser extension support)
        grammar: Grammar
            This is the grammar we are extending the parser with.
        """
        # def __call__(self, grammar, max_tokens=1000000, n=1, top_p=1, temperature=0.0, ensure_bos_token=True):
        # assert n == 1, "Still need to add support for n > 1!"

        # note we only support a fixed set of engine variables for the sake of security
        self._replacements = replace_model_variables(
            grammar, self, allowed_vars=["eos_token", "bos_token"]
        )

        # right now we only support a text/bytes prompt parser state, so we extract that
        if isinstance(parser, bytes):
            prompt = parser
        elif isinstance(parser, str):
            prompt = bytes(parser, encoding="utf8")
        elif isinstance(parser, Parser):
            raise NotImplementedError(
                "Still need to implement support for extending a full Parser state."
            )
        else:
            raise Exception("The passed parser is of an unknown type!")

        # add the beginning of sequence token if needed
        if (
            ensure_bos_token
            and self.tokenizer.bos_token is not None
            and not prompt.startswith(self.tokenizer.bos_token)
        ):
            prompt = self.tokenizer.bos_token + prompt

        # run a simple tokenizer (that does not use a grammar) on the prefix for better performance
        self._token_ids, self._token_byte_positions = self._tokenize_prefix(prompt)
        self._token_ids, self._token_byte_positions = self._cleanup_tokens(
            self._token_ids, self._token_byte_positions
        )
        if len(self._token_byte_positions) > 0:
            self._pre_parser_bytes = self._token_byte_positions[-1]
            self._trimmed_prompt_prefix = prompt[: self._token_byte_positions[-1]]
            prompt = prompt[self._token_byte_positions[-1] :]
        else:
            self._trimmed_prompt_prefix = b""
            self._pre_parser_bytes = 0

        # create a parser with a grammar that includes both our context and the passed grammar
        self._parser = EarleyCommitParser(prompt + grammar)

        # loop until we have generated a complete pattern
        self._hidden_count = len(prompt)  # we don't emit the prompt
        self._generated_pos = 0
        self._sampled_token_ind = None
        self._token_count = 0
        self._last_token_count = 0
        self._was_forced = False
        self._captured_data = {}
        self._captured_log_prob_data = {}

    def next(self, logits):
        """Move the grammar state machine processing forward to the next point where
            either get_logits is required to be called or we have a partial response
            to stream back.

        Parameters
        ----------
        logits : the logits obtained from the LLM after the last return from next(...)
        """

        logits_state = None
        response_state = None

        token_pos = 0
        is_generated = True

        is_new_token = False
        if logits is not None:
            is_new_token = True

            # if requested we compute the log probabilities so we can track the probabilities of each node
            if self.compute_log_probs:
                if torch_is_imported:
                    # note we don't adjust for temp since we consider that a sampling step, not part of the probs
                    probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1).cpu().numpy()
                else:
                     # this numpy code is slower, so we don't use it if we have torch...
                    probs = softmax(logits, axis=-1)
                self.tokenizer.clean_duplicate_tokens(probs)
                self._trie.compute_probs(probs)  # C++ impl
            else:
                probs = None

            grammar_temp = self._parser.next_byte_temperature()
            current_temp = grammar_temp if grammar_temp >= 0 else 0

            # get the sampling order
            if current_temp == 0:
                # we need numpy so the enumerate below does not get really slow...
                sampling_order = np.argsort(-logits)  
            else:
                # assert top_p == 1, "Still need to add support for top_p!"
                if torch_is_imported:
                    logits = torch.tensor(logits)
                    torch.div(logits, current_temp, out=logits)
                    probs_torch = torch.nn.functional.softmax(logits, dim=-1)
                    sampling_order = torch.multinomial(probs_torch, len(probs_torch)).cpu().numpy()
                else:
                    # this numpy version allows us to drop our dependence on pytorch...but it is way slower
                    if probs is None:
                        probs = softmax(logits / current_temp, axis=-1)
                    probs += 1e-10  # ensure we have no zero probs that mess up numpy
                    probs /= np.sum(probs)
                    sampling_order = np.random.choice(
                        len(probs), size=len(probs), p=probs, replace=False
                    )  # the 1e-10 is ensure we have no zero probs, which numpy does not like

            # loop over the tokens looking for a valid one
            for i, self._sampled_token_ind in enumerate(sampling_order):
                self._sampled_token = self.tokenizer.tokens[self._sampled_token_ind]

                # break out if we have reach impossible tokens
                if logits[self._sampled_token_ind] <= -np.inf:
                    break

                # make sure it matches any forced prefix
                used_forced_pos = min(self._forced_pos, self._start_pos + len(self._sampled_token))
                if (
                    self._start_pos < self._forced_pos
                    and not self._sampled_token.startswith(
                        self._parser.bytes[self._start_pos : used_forced_pos]
                    )
                ):
                    continue
                offset = used_forced_pos - self._start_pos

                # make sure the parse is backed up to the position we want to start checking from TODO: make this account for shared prefixes with the last token
                self._parser.pos = used_forced_pos
                self._new_bytes_prob = 1.0

                # if we have gotten to the end of the valid tokens then we stop
                # if logits[self._sampled_token_ind] == -np.inf:
                #     raise self._report_failed_match(self._trimmed_prompt_prefix + self._parser.bytes)

                # check to see if the sampled token is allowed
                token_pos = offset

                # this is the Trie node we were left at when we could force the next byte above
                node = self._trie

                while token_pos < len(self._sampled_token):
                    next_byte = self._sampled_token[token_pos : token_pos + 1]
                    next_node = node.child(next_byte)

                    # if we don't have a cached match flag compute it using the grammar
                    if next_node.match_version < self._token_trie.match_version:
                        next_byte_mask = self._parser.next_byte_mask()

                        # we update all the children since the parser knows the full mask
                        for byte in node.keys():  
                            child = node.child(byte)
                            child.match_version = self._token_trie.match_version
                            child.match = next_byte_mask[byte[0]]

                    # advance or fail according to the (now up-to-date) match cache
                    if next_node.match:

                        # get the parser to consume the next byte
                        if next_node.prob < 1e-8:
                            if node.prob < 1e-8:
                                log_prob_delta = 0
                            else:
                                log_prob_delta = -20
                        else:
                            log_prob_delta = np.log(next_node.prob) - np.log(node.prob)
                        # log_prob_delta = np.log(next_node.prob) - np.log(node.prob)
                        self._new_bytes_prob = next_node.prob
                        commit_point = self._parser.consume_byte(
                            next_byte, log_prob=log_prob_delta
                        )

                        # mark that we accepted this byte
                        node = next_node
                        token_pos += 1

                        # if we are at a hidden commit point then we need to hide the bytes that match that node
                        if commit_point is not None and commit_point.node.hidden:

                            # if we are capturing the data from this node we need to do that now since we are about to remove it
                            # TODO: build a whole parse tree under this commit_point node so we can record child node captures
                            if commit_point.node.capture_name:
                                self._captured_data[commit_point.node.capture_name] = (
                                    self._parser.bytes[commit_point.start :]
                                )
                                self._captured_log_prob_data[
                                    commit_point.node.capture_name
                                ] = commit_point.log_prob

                            # This takes the item and commits to it as part of the parse and then shrinks it to zero width
                            # in other words this hides the item
                            self._parser.commit_and_collapse_item(commit_point)

                            # keep the bytes we still need to emit
                            if self._forced_pos < commit_point.start:
                                self._parser.shadow_rewind(self._forced_pos)

                            else:
                                # pop off any tokens that overlap the hidden bytes
                                i = len(self._token_byte_positions) - 1
                                while (
                                    i >= 0
                                    and self._token_byte_positions[i]
                                    - self._pre_parser_bytes
                                    > commit_point.start
                                ):
                                    self._token_ids.pop()
                                    self._token_byte_positions.pop()
                                    self._token_count -= 1
                                    i -= 1
                                # re-add any bytes we cut too far on
                                self._parser.shadow_rewind(
                                    self._token_byte_positions[-1]
                                    - self._pre_parser_bytes
                                )
                            is_new_token = False
                            break

                        elif token_pos == len(self._sampled_token):
                            break  # this token is valid
                    else:
                        # partially valid tokens are okay if we are running off the end of a grammar, but not otherwise
                        if not self._parser.matched():
                            token_pos = -1

                        break  # this token is no longer valid

                # see if we are breaking out of the whole loop
                if not is_new_token:
                    break

                # check if this token is dominated by other longer valid tokens (and hence would never be consistent with greedy tokenization)
                # TODO: disabled for now because of sentencepeice non-local issues
                # if token_pos == len(self._sampled_token) and not self._parser.matched(): # not we don't check if we have matched, because then we can generate anything afterwards
                #     if _check_dominated(node, self._parser, self._token_trie.match_version, self._parser.next_byte_mask()):
                #         token_pos = -1

                if token_pos > 0:
                    break  # we found a valid token

                if self._parser.matched():
                    break  # if we already have a full match we don't try more tokens we just give up as soon as the model deviates from the grammar

        is_done = False
        while True:  # each iteration generates one more token (and some of the associated bytes)
            if is_new_token:
                # emit whatever we know will not be hidden
                new_bytes = self._parser.bytes[self._generated_pos : self._parser.earliest_hidden_start()]

                # if we cannot consume any more tokens then we are done
                if (
                    not self._is_forced
                    and token_pos < len(self._sampled_token)
                    and self._trie == self._token_trie
                ):

                    # which if can't consume any more tokens, but we are not yet done
                    if not self._parser.matched():
                        self._parser.matched()
                        raise self._report_failed_match(
                            self._trimmed_prompt_prefix + self._parser.bytes
                        )

                    # TODO: if we exactly match the end of the pattern then we can commit to this last token
                    # if m.span()[1] == len(generated_text):
                    #     self._cache_state["new_token_ids"].append(self._sampled_token_ind)

                    # capture the named groups from the parse tree
                    self._parser.get_captures(self._captured_data, self._captured_log_prob_data)

                    # we have no valid log prob data if we didn't compute it
                    # yield new_bytes[self._hidden_count:], self._is_generated, self._new_bytes_prob, self._captured_data, self._captured_log_prob_data, token_count - last_token_count

                    response_state = (
                        new_bytes[self._hidden_count :],
                        is_generated,
                        self._new_bytes_prob if self.compute_log_probs else 1.0,
                        self._captured_data,
                        self._captured_log_prob_data,
                        self._token_count - self._last_token_count,
                    )

                    self._last_token_count = self._token_count

                    # TODO: we only need to do this when we might re-use the grammar object...we might want to account for that
                    unreplace_model_variables(self._replacements)

                    is_done = True
                else:
                    self._generated_pos += len(new_bytes)

                    # yeild the snippet of text created by the next token
                    out = new_bytes[self._hidden_count :]
                    if len(out) > 0:
                        # capture the named groups from the (partial) parse tree, # TODO: disabled for now until we handle list_append correctly
                        # new_captured_data, new_captured_log_prob_data = self._parser.get_captures()
                        # self._captured_data.update(new_captured_data)
                        # self._captured_log_prob_data.update(new_captured_log_prob_data)
                        # yield out, self._is_generated, self._new_bytes_prob, self._captured_data, self._captured_log_prob_data, self._token_count - self._last_token_count # note that we don't capture groups until a complete parse right now...

                        response_state = (
                            out,
                            is_generated,
                            self._new_bytes_prob if self.compute_log_probs else 1.0,
                            self._captured_data,
                            self._captured_log_prob_data,
                            self._token_count - self._last_token_count,
                        )

                        self._last_token_count = self._token_count
                        self._hidden_count = 0
                        self._token_count += 1  # note we only update this for tokens that emit non-hidden content
                    else:
                        self._hidden_count -= len(new_bytes)

                    self._token_ids.append(self._sampled_token_ind)

                    # track the byte position of each token
                    if len(self._token_byte_positions) == 0:
                        self._token_byte_positions.append(len(self._sampled_token))
                    else:
                        self._token_byte_positions.append(
                            self._token_byte_positions[-1] + len(self._sampled_token)
                        )

                if response_state is not None:
                    break

            token_pos = 0
            is_generated = False

            is_new_token = True

            # note where we are starting for this token
            self._start_pos = self._parser.pos

            # let the parser know that we have advanced another token (used ofr tracking max token limits)
            self._parser.mark_new_token()

            # walk down the trie as far as possible before computing the logits
            self._trie = self._token_trie
            
            # this invalidates all the match caches from the previous token
            self._trie.match_version += 1
            # self._trie.prob = 0.0 # need to reset when we reset the match_version
            while True:
                next_byte_mask = self._parser.next_byte_mask()
                next_byte_mask_sum = next_byte_mask.sum()

                # see if we reached a dead end of the grammar
                if next_byte_mask_sum == 0:
                    break

                # if there is more than one option we cannot advance without computing the logits
                elif next_byte_mask_sum != 1:
                    break

                # we are not forced if we are at the end of the grammar
                elif self._parser.matched():
                    break

                # if there is only one possible next byte we can keep forcing
                elif next_byte_mask_sum == 1:

                    # look for valid children
                    next_byte = None
                    for byte in self._trie.keys():

                        # mark this self._trie node with an up-to-date match flag (may save work later)
                        node = self._trie.child(byte)
                        node.match_version = self._token_trie.match_version
                        # node.prob = 0.0 # reset when we reset the match_version
                        node.match = next_byte_mask[byte[0]]

                        # see if we found a match
                        if node.match:
                            next_byte = byte
                            break

                    # if we can't extend then this token is forced
                    if next_byte is None:
                        break

                    # otherwise since there is only one possible next byte we keep going
                    else:
                        commit_point = self._parser.consume_byte(
                            next_byte, log_prob=0.0
                        )

                        # if we are at a hidden commit point then we need to hide the bytes that match that node
                        if commit_point is not None and commit_point.node.hidden:

                            # This takes the item and commits to it as part of the parse and then shrinks it to zero width
                            # in other words this hides the item
                            self._parser.commit_and_collapse_item(commit_point)

                            # keep the bytes we still need to emit
                            if self._start_pos < commit_point.start:
                                self._parser.shadow_rewind(self._start_pos)

                            else:
                                # pop off any tokens that overlap the hidden bytes
                                i = len(self._token_byte_positions) - 1
                                while (
                                    i >= 0
                                    and self._token_byte_positions[i]
                                    - self._pre_parser_bytes
                                    > commit_point.start
                                ):
                                    self._token_ids.pop()
                                    self._token_byte_positions.pop()
                                    self._token_count -= 1
                                    i -= 1
                                # re-add any bytes we cut too far on
                                self._parser.shadow_rewind(
                                    self._token_byte_positions[-1]
                                    - self._pre_parser_bytes
                                )
                            is_new_token = False  # this restarts us at the top of the outer token gen loop
                            break

                        self._trie = self._trie.child(next_byte)

            self._forced_pos = self._parser.pos  # record how far the bytes are forced

            if is_new_token:
                # back up if we got forced up to a point that is not a valid token
                if next_byte_mask_sum <= 1:
                    while self._trie.value < 0 and self._trie.parent() is not None:
                        self._trie = self._trie.parent()
                        self._forced_pos -= 1
                    self._parser.pos = self._forced_pos

                # if we walked all the way to a forced token then we advance without computing the logits
                # we are forced if there are no more options and we are either in the middle of the grammar or at a trie leaf
                self._is_forced = next_byte_mask_sum <= 1 and (
                    len(self._trie) == 0
                    if self._parser.matched()
                    else self._trie != self._token_trie
                )
                if self._is_forced:
                    self._sampled_token_ind = self._trie.value
                    self._sampled_token = self.tokenizer.tokens[self._sampled_token_ind]
                    self._new_bytes_prob = 1.0
                    self._was_forced = True

                # we are at the end of the grammar
                elif next_byte_mask_sum == 0:

                    # mark the token we "sampled" if we have comsumed some bytes
                    if self._trie != self._token_trie:
                        self._sampled_token_ind = self._trie.value
                        self._sampled_token = self.tokenizer.tokens[
                            self._sampled_token_ind
                        ]
                        self._new_bytes_prob = 1.0

                # otherwise we need to compute the logits and sample a valid token
                else:

                    # if we were forced we might need to clean up the greedy tokenization to match the global tokenization behavior as seen in training
                    if self._was_forced:
                        self._token_ids, self._token_byte_positions = (
                            self._cleanup_tokens(
                                self._token_ids, self._token_byte_positions
                            )
                        )
                        self._was_forced = False

                    grammar_temp = self._parser.next_byte_temperature()
                    current_temp = grammar_temp if grammar_temp >= 0 else 0
                    logits_state = (
                        self._token_ids,
                        self._parser.bytes[self._start_pos : self._forced_pos],
                        current_temp,
                    )
                    break

        return is_done, logits_state, response_state

    def __call__(self, parser, grammar, ensure_bos_token=True):
        """Returns a new updated parser state executed through the grammar.

        Parameters
        ----------
        parser : str or Parser
            This is represents the current state of a guidance parser that will be extended
            using the passed grammar. If a string is given then we assume the previous parser
            state is just a fixed string prompt, if a full Parser is given then we extend that
            parser by appending the new grammar to the parser's current grammar and then
            inferencing the model. (TODO: implement full parser extension support)
        grammar: Grammar
            This is the grammar we are extending the parser with.
        """

        self.start(parser, grammar, ensure_bos_token)

        logits = None
        while True:
            is_done, logits_state, response_state = self.next(logits)
            logits = None

            if response_state is not None:
                (
                    response_new_bytes,
                    response_is_generated,
                    response_new_bytes_prob,
                    response_capture_groups,
                    response_capture_group_log_probs,
                    response_new_token_count,
                ) = response_state

                yield EngineCallResponse(
                    new_bytes=response_new_bytes,
                    is_generated=response_is_generated,
                    new_bytes_prob=response_new_bytes_prob,
                    capture_groups=response_capture_groups,
                    capture_group_log_probs=response_capture_group_log_probs,
                    new_token_count=response_new_token_count,
                )

            if logits_state is not None:
                token_ids, forced_bytes, current_temp = logits_state
                logits = self.get_logits(token_ids, forced_bytes, current_temp)

            if is_done:
                break

    def _tokenize_prefix(self, byte_string):
        """This is used to speed up the tokenization of long prompts without using the parser."""
        token_ids = []
        token_byte_positions = []

        # loop trying to decode a new token at each iteration
        pos = 0
        while True:

            # walk down the token trie looking for a unique token match
            trie = self._token_trie
            valid_pos = -1
            valid_value = -1
            while True:
                if pos >= len(byte_string):
                    if len(trie) > 0:
                        valid_pos = -1
                    break

                # check if we can keep going or are at a dead end
                if trie.has_child(byte_string[pos : pos + 1]):
                    trie = trie.child(byte_string[pos : pos + 1])
                    pos += 1

                    # record the last valid token down this path as we go
                    if trie.value >= 0:
                        valid_pos = pos
                        valid_value = trie.value
                else:
                    break  # we can't go any farther

            if valid_pos == -1:
                break
            else:
                token_ids.append(valid_value)
                token_byte_positions.append(valid_pos)
                pos = valid_pos

        return token_ids, token_byte_positions

    def _cleanup_tokens(self, token_ids, token_byte_positions):

        # compute a joint tokenization
        joint_token_ids = self.tokenizer.recode(token_ids)

        # see if we need to redo the tokenization
        redo = False
        if len(joint_token_ids) != len(token_ids):
            redo = True
        else:
            for i, id in enumerate(joint_token_ids):
                if token_ids[i] != id:
                    redo = True
                    break

        if redo:
            token_ids = joint_token_ids
            last_pos = token_byte_positions[-1]
            token_byte_positions = []
            pos = 0
            for i, id in enumerate(joint_token_ids):
                pos += len(self.tokenizer.tokens[id])
                token_byte_positions.append(pos)

            # ugly hack to deal with sentence piece craziness of space hiding after special tokens 
            # TODO: figure out how to make this more robust
            if (
                token_byte_positions[-1] == last_pos + 1
                and self.tokenizer.tokens[token_ids[0]] == b"<s>"
                and self.tokenizer.tokens[token_ids[1]][0:1] == b" "
            ):
                for i in range(1, len(token_byte_positions)):
                    token_byte_positions[i] -= 1
            
            # another ugly hack for tokenizers that are not stable on encode/decode cycles
            # currently only Phi-3, should generalize this method if we see more of these
            if token_byte_positions[-1] != last_pos:
                if not hasattr(self, "_disable_retokenize_check"):
                    msg = textwrap.dedent(
                        """Self-consistency check in _cleanup_tokens() failed.
                        
                        This is not a fatal issue, but if there are subsequent
                        generation problems, please include this warning in
                        your bug report."""
                    )
                    warnings.warn(msg)

        return token_ids, token_byte_positions

    def get_logits(self, token_ids, forced_bytes, current_temp):
        """A fake method designed to be overriden by subclasses."""

        # pretend to extend the KV cache and update the log probs
        return np.randn(len(self.tokenizer.tokens))

    def _report_failed_match(self, prompt):
        """Note that this can be overridden by subclasses that have more likely reasons than a bug in the token set (like remote models)."""
        return Exception(
            "We can't consume any more tokens, but we are not yet done! Perhaps your model's token set is incomplete? This happened after the prompt:"
            + str(prompt[-40:])
        )


class Model:
    """The base guidance model object, which represents a model in a given state.

    Model objects are immutable representations of model state, so whenever you change
    them you get a new Model object. However, these copies share the "expensive"
    parts of the underlying model like the parameters and KV-cache, through a shared
    Engine, so making copies of Model objects is cheap.

    .. automethod:: __add__
    """

    open_blocks: Dict["ContextBlock", None] = {}  # track what context blocks are open
    _grammar_only = 0  # a flag that tracks when we are forced to be executing only compiled grammars (like when we are inside a select)
    _throttle_refresh = 0  # a flag that tracks when we can throttle our display since we know future display calls are going to happen

    def __init__(self, engine, echo=True, **kwargs):
        """Build a new model object that represents a model in a given state.

        Note that this constructor is not meant to be used directly, since there

        Parameters
        ----------
        engine : Engine
            The inference engine to use for this model.
        echo : bool
            If true the final result of creating this model state will be displayed (as HTML in a notebook).
        """
        if isinstance(engine, str) and engine.startswith("http"):
            from ._remote import RemoteEngine

            engine = RemoteEngine(engine, **kwargs)

        # # auto-wrap the tokenizer in the standard guidance interface
        # if not isinstance(tokenizer, Tokenizer):
        #     tokenizer = Tokenizer(tokenizer)

        self.engine = engine
        self.chat_template = engine.get_chat_template() # TODO [HN]: Should this be a method or attr?
        self.echo = echo
        self.token_count = 0  # tracks how many tokens our byte state represents
        self.max_display_rate = 0.2  # this controls how frequently we are allowed to redraw the display (in seconds)
        self.opened_blocks = {}  # what context blocks have been opened but not closed
        # self.compute_log_probs = compute_log_probs

        # private attributes
        self._variables = {}  # these are the state variables stored with the model
        self._variables_log_probs = {}  # these are the state variables stored with the model
        self._cache_state = {}  # mutable caching state used to save computation
        self._state = ""  # the current bytes that represent the state of the model
        self._event_queue = None  # TODO: these are for streaming results in code, but that needs implemented
        self._event_parent = None
        self._last_display = 0  # used to track the last display call to enable throttling
        self._last_event_stream = 0  # used to track the last event streaming call to enable throttling

    @property
    def active_role_end(self):
        """The default end patterns we should use for `gen` calls.
        TODO: move this logic into the gen call...we can do with if we allow model_variables to run functions.

        These patterns are computed dynamically by the model object because they can depend on
        what the current open roles are, which is something
        """

        # add any active non-empty role ends. Ignore role ends that are spaces
        parts = []
        for _, role_end_str in self.opened_blocks.values():
            role_end_str = format_pattern.sub("", role_end_str)
            if len(role_end_str) > 0 and not re.fullmatch(r"\s+", role_end_str):
                parts.append(role_end_str)

        return select(parts)

    def _html(self):
        """Generate HTML that displays the model object."""
        display_out = self._state
        for context in reversed(self.opened_blocks):
            display_out += self.opened_blocks[context][1]
        display_out = html.escape(display_out)
        display_out = nodisp_pattern.sub("", display_out)
        display_out = html_pattern.sub(lambda x: html.unescape(x.group(1)), display_out)
        display_out = image_pattern.sub(
            lambda x: '<img src="data:image/png;base64,'
            + base64.b64encode(self[x.groups(1)[0]]).decode()
            + '" style="max-width: 400px; vertical-align: middle; margin: 4px;">',
            display_out,
        )
        display_out = (
            "<pre style='margin: 0px; padding: 0px; vertical-align: middle; padding-left: 8px; margin-left: -8px; border-radius: 0px; border-left: 1px solid rgba(127, 127, 127, 0.2); white-space: pre-wrap; font-family: ColfaxAI, Arial; font-size: 15px; line-height: 23px;'>"
            + display_out
            + "</pre>"
        )
        return display_out

    def _send_to_event_queue(self, value):
        """For streaming in code.

        TODO: Is this still needed?"""
        if self._event_queue is not None:
            self._event_queue.put(value)
        if self._event_parent is not None:
            self._event_parent._send_to_event_queue(value)

    def stream(self):
        return ModelStream(self)

    def copy(self):
        """Create a shallow copy of the model object."""

        # start with a shallow copy
        new_lm = copy.copy(self)

        # then copy a few things we need deeper copies of
        new_lm._variables = self._variables.copy()
        new_lm._variables_log_probs = self._variables_log_probs.copy()
        new_lm.opened_blocks = self.opened_blocks.copy()

        # create a new clean event queue
        new_lm._event_queue = None  # we start with no event queue because nobody is listening to us yet

        if self._event_queue is not None:
            # if the current lm has an event queue, we make it our parent
            new_lm._event_parent = self

        elif self._event_parent is not None:
            # otherwise if the current event que has an event parent then that is also our parent
            new_lm._event_parent = self._event_parent  

        return new_lm

    def _inplace_append(self, value, force_silent=False):
        """This is the base way to add content to the current LM object that is being constructed.

        All updates to the model state should eventually use this function.
        Note this should only be used after making a copy, otherwise immutability would be violated.

        Parameters
        ----------
        value : bytes
            The bytes we should append to our current state.
        """

        # update the byte state
        self._state += str(value)  # TODO: make _state to be bytes not a string

        # see if we should update the display
        if not force_silent:
            self._update_display()

        # this is for programmatic streaming among other things
        if Model._throttle_refresh > 0:
            curr_time = time.time()
            if curr_time - self._last_event_stream >= self.max_display_rate:
                self._last_event_stream = curr_time
                self._send_to_event_queue(self)
        else:
            self._send_to_event_queue(self)

    def _update_display(self, throttle=True):
        if self.echo:
            if Model._throttle_refresh > 0:
                curr_time = time.time()
                if throttle and curr_time - self._last_display < self.max_display_rate:
                    return  # we are throttling the update
                else:
                    self._last_display = curr_time

            if ipython_is_imported:
                clear_output(wait=True)
                display(HTML(self._html()))
            else:
                pprint(self._state)

    def reset(self, clear_variables=True):
        """This resets the state of the model object.

        Parameters
        ----------
        clear_variables : bool
            If we should clear all the model object's variables in addition to reseting the byte state.
        """
        self._state = self._state[:0]
        if clear_variables:
            self._variables = {}
            self._variables_log_probs = {}
        return self

    def _repr_html_(self):
        if ipython_is_imported:
            clear_output(wait=True)
        return self._html()

    def _current_prompt(self):
        """The current prompt in bytes (which is the state without the context close tags)."""
        return format_pattern.sub("", self._state)

    def __str__(self):
        """A string representation of the current model object (that includes context closers)."""
        out = self._current_prompt()
        for context in reversed(self.opened_blocks):
            out += format_pattern.sub("", self.opened_blocks[context][1])
        return out

    def __add__(self, value):
        """Adding is the primary mechanism for extending model state.

        Parameters
        ----------
        value : guidance grammar
            The grammar used to extend the current model.
        """

        # create the new lm object we will return
        # (we need to do this since Model objects are immutable)
        lm = self.copy()

        # inside this context we are free to drop display calls that come too close together
        with throttle_refresh():

            # find what new blocks need to be applied
            new_blocks = []
            for context in Model.open_blocks:
                if context not in lm.opened_blocks:
                    new_blocks.append(context)

                    # mark this so we don't re-add when computing the opener or closer (even though we don't know the close text yet)
                    lm.opened_blocks[context] = (0, "")

            # find what old blocks need to be removed
            old_blocks = []
            for context in list(reversed(lm.opened_blocks)):
                if context not in Model.open_blocks and context in lm.opened_blocks:
                    old_blocks.append((lm.opened_blocks[context], context))

                    # delete this so we don't re-close when computing the opener or closer
                    del lm.opened_blocks[context]

            # close any newly closed contexts
            for (pos, close_text), context in old_blocks:
                if context.name is not None:
                    lm._variables[context.name] = format_pattern.sub(
                        "", lm._state[pos:]
                    )
                lm += context.closer

            # apply any newly opened contexts (new from this object's perspective)
            for context in new_blocks:
                lm += context.opener
                with grammar_only():
                    tmp = lm + context.closer
                close_text = tmp._state[len(lm._state):]  # get the new state added by calling the closer
                lm.opened_blocks[context] = (len(lm._state), close_text)

                # clear out names that we override
                if context.name is not None:
                    if context.name in lm._variables:
                        del lm._variables[context.name]
                        if context.name in lm._variables_log_probs:
                            del lm._variables_log_probs[context.name]

            # wrap raw string values
            if isinstance(value, str):
                is_id = False
                parts = re.split(_tag_pattern, value)

                # we have no embedded objects
                if len(parts) == 1:
                    lm._inplace_append(value)
                    out = lm

                # if we have embedded objects we have to convert the string to a grammar tree
                else:
                    partial_grammar = _null_grammar
                    lm.suffix = ""
                    for i, part in enumerate(parts):
                        if i < len(parts) - 1:
                            lm.suffix = parts[i + 1]
                        if is_id:
                            call = _call_pool[part]
                            if isinstance(call, GrammarFunction):
                                partial_grammar += _call_pool[part]
                            else:
                                lm += partial_grammar
                                lm = _call_pool[part](lm)
                                partial_grammar = _null_grammar
                        elif part != "":
                            partial_grammar += string(part)
                        is_id = not is_id
                    out = lm + partial_grammar

            # if we find a null value we do nothing
            elif isinstance(value, Null):
                out = lm

            # run stateless functions (grammar nodes)
            elif isinstance(value, GrammarFunction):
                out = lm._run_stateless(value)

            # run stateful functions
            else:
                out = value(lm)
                if out is None:
                    raise Exception(
                        f"A guidance function returned `None`, not a model object! Did you forget to return the new lm at the end of your function?"
                    )
                if not isinstance(out, Model):
                    raise Exception(
                        f"A guidance function did not return a model object! Did you try to add a function to a model without calling the function? For example `model + guidance_function()` is correct, while `model + guidance_function` will cause this error."
                    )

        # this flushes the display
        out._inplace_append("")

        return out

    # def endswith(self, s):
    #     '''Checks if the current model state ends with the given value.'''
    #     return self._current_prompt().endswith(s)

    def __len__(self):
        """The string length of the current state.

        TODO: This should change to the byte length...
        """
        return len(str(self))

    def __setitem__(self, key, value):
        raise Exception(
            "Model objects are immutable so you can't use __setitem__! Consider using the .set(key, value) method instead to create a new updated model object."
        )

    def __getitem__(self, key):
        if key in self._variables:
            return self._variables[key]

        # look for named blocks that are still open with the given key as their name
        else:
            for context in list(reversed(self.opened_blocks)):
                if context.name == key:
                    return format_pattern.sub(
                        "", self._state[self.opened_blocks[context][0] :]
                    )

        raise KeyError(f"Model does not contain the variable '{key}'")

    def __contains__(self, item):
        return item in self._variables

    def get(self, key, default=None):
        """Return the value of a variable, or a default value if the variable is not present.

        Parameters
        ----------
        key : str
            The name of the variable.
        default : any
            The value to return if the variable is not current set.
        """
        return self._variables.get(key, default)

    def setattr(self, key, value):
        """Return a new model with the given model attribute set.

        Parameters
        ----------
        key : str
            The name of the attribute to be set.
        value : any
            The value to set the attribute to.
        """
        copy = self.copy()
        setattr(copy, key, value)
        return copy

    def delattr(self, key):
        """Return a new model with the given attribute deleted.

        Parameters
        ----------
        key : str
            The attribute name to remove.
        """
        copy = self.copy()
        delattr(copy, key)
        return copy

    def set(self, key, value):
        """Return a new model with the given variable value set.

        Parameters
        ----------
        key : str
            The name of the variable to be set.
        value : any
            The value to set the variable to.
        """
        copy = self.copy()
        copy._variables[key] = value
        copy._variables_log_probs[key] = 0.0
        return copy

    def remove(self, key):
        """Return a new model with the given variable deleted.

        Parameters
        ----------
        key : str
            The variable name to remove.
        """
        if key in self._variables:
            copy = self.copy()
            del copy._variables[key]
            if key in copy._variables_log_probs:
                del copy._variables_log_probs[key]
        else:
            copy = self
        return copy

    def log_prob(self, key, default=None):
        """Return the log prob of a variable, or a default value if the variable is not present.

        Parameters
        ----------
        key : str
            The name of the variable.
        default : any
            The value to return if the variable is not current set.
        """
        # TODO: support calling without a key to get the log prob of the whole model
        return self._variables_log_probs.get(key, default)

    # def get_cache(self):
    #     return self.engine.cache

    #     def tool_def(self, functions):

    #         self += """
    # # Tools

    # """
    #         if len(functions) > 0:
    #             self += '''## functions

    # namespace functions {

    # '''
    #         for function in functions:
    #             self += f"""// {function['description']}
    # type {function['name']} = (_: {{"""
    #             for prop_name,prop_data in function["parameters"]["properties"].items():
    #                 if "description" in prop_data:
    #                     self += f"\n// {prop_data['description']}\n"
    #                 self += prop_name
    #                 if prop_name not in function["parameters"]["required"]:
    #                     self += "?"
    #                 self += ": "
    #                 if "enum" in prop_data:
    #                     for enum in prop_data["enum"]:
    #                         self += f'"{enum}"'
    #                         if enum != prop_data["enum"][-1]:
    #                             self += " | "
    #                 else:
    #                     self += prop_data["type"]

    #                 if prop_name != list(function["parameters"]["properties"].keys())[-1]:
    #                     self += ",\n"
    #             self += """
    # }) => any;

    # """
    #             self[function['name']] = function
    #         self += "} // namespace functions\n"

    #         return self

    def _run_stateless(self, stateless_function, temperature=0.0, top_p=1.0, n=1):
        assert (
            Model._grammar_only == 0
        ), "We can't run grammar parsing while in context free mode! (for example inside a block closer)"

        logger.debug("start Model._run_stateless")

        # This needs to be here for streaming
        # if name is not None:
        #     self[name] = ""

        # replace ModelVariables with their actual values (note we save what we replaced so we can restore it later)
        replacements = replace_model_variables(stateless_function, self)

        # start the generation stream
        gen_obj = self.engine(self._current_prompt(), stateless_function)

        # we will return a new extended version of ourselves, which we track as `lm`
        lm = self

        # single generation
        if n == 1:
            generated_value = ""
            # logprobs_out = []

            delayed_bytes = b""
            # last_is_generated = False
            for chunk in gen_obj:

                # we make everything full probability if we are not computing uncertainty
                # if not self.engine.compute_log_probs:
                #     chunk.new_bytes_prob = 1.0

                # convert the bytes to a string (delaying if we don't yet have a valid unicode string)
                lm.token_count += chunk.new_token_count
                chunk.new_bytes = delayed_bytes + chunk.new_bytes
                try:
                    new_text = chunk.new_bytes.decode("utf8")
                except UnicodeDecodeError:
                    delayed_bytes = chunk.new_bytes
                    continue
                delayed_bytes = b""

                if len(chunk.new_bytes) > 0:
                    generated_value += new_text
                    if chunk.is_generated:
                        lm += f"<||_html:<span style='background-color: rgba({165*(1-chunk.new_bytes_prob) + 0}, {165*chunk.new_bytes_prob + 0}, 0, {0.15}); border-radius: 3px;' title='{chunk.new_bytes_prob}'>_||>"
                    lm += new_text
                    if chunk.is_generated:
                        lm += "<||_html:</span>_||>"

                # last_is_generated = chunk.is_generated

                if len(chunk.capture_groups) > 0:
                    for k in chunk.capture_groups:
                        v = chunk.capture_groups[k]

                        # see if we are in a list_append mode
                        if isinstance(v, list):
                            for i, inner_v in enumerate(v):
                                # convert to a string if possible
                                # TODO: will need to not just always do this once we support images etc.
                                try:
                                    inner_v = (
                                        inner_v.decode("utf8")
                                        if isinstance(inner_v, bytes)
                                        else inner_v
                                    )
                                except UnicodeDecodeError:
                                    pass

                                if k not in lm or not isinstance(lm._variables[k], list):
                                    lm._variables[k] = []
                                if k not in lm._variables_log_probs or not isinstance(lm._variables_log_probs[k], list):
                                    lm._variables_log_probs[k] = []
                                    
                                lm._variables[k].append(inner_v)
                                lm._variables_log_probs[k].append(
                                    chunk.capture_group_log_probs[k][i]
                                )

                        # ...or standard assignment mode
                        else:
                            # convert to a string if possible
                            # TODO: will need to not just always do this once we support images etc.
                            try:
                                v = v.decode("utf8") if isinstance(v, bytes) else v
                            except UnicodeDecodeError:
                                pass
                            lm._variables[k] = v
                            lm._variables_log_probs[k] = chunk.capture_group_log_probs[k]

            # if len(chunk.capture_groups) > 0:
            #     for k in chunk.capture_groups:
            #         v = chunk.capture_groups[k]
            #         lm[k] = v.decode("utf8") if isinstance(v, bytes) else v

        unreplace_model_variables(replacements)

        logger.debug("finish Model._run_stateless")

        return lm


class ModelStream:
    def __init__(self, model, grammar=None, timeout=5):
        """Create a model stream object that delays execution until it is iterated over."""
        if model.echo:
            model = model.copy()
            model.echo = False  # turn off display echoing
        self.model = model
        self.grammar = grammar
        self.timeout = timeout

    def __add__(self, grammar):
        """Extend this delayed chain of execution with another grammar append."""
        if self.grammar is None:
            return ModelStream(self.model, grammar)
        else:
            return ModelStream(self.model, self.grammar + grammar)

    def _inner_run(self, model):
        """This runs the model stream without iterating, and is only using internally by __iter__."""
        if isinstance(self.grammar, ModelStream):
            model = self.grammar._inner_run(model)
        elif self.grammar is None:
            model = self.model + ""
        else:
            model = self.model + self.grammar

    def __iter__(self):
        """Starts a thread to execute the model and grammar, yielding events as they occur."""

        # Create a thread-safe queue to hold events
        with CaptureEvents(self.model) as events:

            # Define the target function for the thread
            def target():
                try:
                    self._inner_run(self.model)
                    events.put(None)  # mark that we are done
                except BaseException as ex:
                    events.put(ex)

            # Start the thread
            thread = threading.Thread(target=target)
            thread.start()

            # Yield events from the queue as they become available
            while True:
                try:
                    # Wait for an event with a timeout to allow for thread termination
                    event = events.get(timeout=self.timeout)
                    if event is None:
                        break
                    elif isinstance(event, BaseException):
                        raise event
                    yield event
                except queue.Empty:
                    # Check if the thread is still alive
                    if not thread.is_alive():
                        break

            # Ensure the thread has completed
            thread.join()


class Chat(Model):
    """The base class for all chat-tuned models."""

    def get_role_start(self, role_name, **kwargs):
        """The starting grammar for a role.

        By default we follow the GPT role tag start conventions.

        Parameters
        ----------
        role_name : str
            The name of the role, like "user", or "assistant"
        kwargs : dict
            This kwargs are added to the role start as arguments.
        """
        return (
            "<|im_start|>"
            + role_name
            + "".join([f' {k}="{v}"' for k, v in kwargs.items()])
            + "\n"
        )

    def get_role_end(self, role_name=None):
        """The ending bytes for a role.

        Note that we cannot use a grammar in closers because they need to remain constant
        so we can append them whenever we need a representation before the final closing of the context.
        By default we follow the GPT role tag end conventions.

        Parameters
        ----------
        role_name : str
            The name of the role, like "user", or "assistant"
        """
        return "<|im_end|>"


class Instruct(Model):
    """The base class for all instruction-tuned models."""

    def get_role_start(self, role_name, **kwargs):
        raise Exception("Subclasses need to define what the role start should be!")

    def get_role_end(self, role_name=None):
        raise Exception("Subclasses need to define what the role end should be!")


class GrammarOnly:
    def __enter__(self):
        Model._grammar_only += 1

    def __exit__(self, exc_type, exc_value, traceback):
        Model._grammar_only -= 1


def grammar_only():
    """Returns a context manager that ensures only grammars are executed (not full python functions)."""
    return GrammarOnly()


class ThrottleRefresh:
    def __enter__(self):
        Model._throttle_refresh += 1

    def __exit__(self, exc_type, exc_value, traceback):
        Model._throttle_refresh -= 1


def throttle_refresh():
    """Returns a context manager that allows the print statement to drop display calls above the throttle rate."""
    return ThrottleRefresh()


class ConstraintException(Exception):
    def __init__(self, *args, **kwargs):
        self.prompt = kwargs.pop("prompt", None)
        self.data = kwargs.pop("data", None)
        super().__init__(*args, **kwargs)


# def _compute_probs(trie, probs, found):
#     '''Computes the log probabilities for each internal trie node.'''
#     if trie.value is not None:
#         found[trie.value] = 1
#         trie.prob += probs[trie.value]

#     if len(trie) > 0:
#         # child_probs = []
#         for b in trie.keys():
#             child = trie.child(b)
#             _compute_probs(child, probs, found)
#             trie.prob += child.prob
#         # trie.log_prob = np.logaddexp.reduce(child_log_probs)


def _check_dominated(node, parser, match_version, next_byte_mask):
    curr_pos = parser.pos
    for byte_num in next_byte_mask.nonzero()[0]:
        next_byte = bytes((byte_num,))
        if not node.has_child(next_byte):
            return False  # no possible exension this direction, so we are not dominated
        child = node.child(next_byte)
        if child.match_version < match_version:
            child.match_version = match_version
            child.match = next_byte_mask[next_byte[0]]

        if not child.match:
            return False  # this child does not dominate the node, so the node is not dominated
        elif child.value is None:  # this child might not dominate the node
            parser.consume_byte(next_byte, log_prob=0.0)
            child_dominate = _check_dominated(child, parser, match_version, parser.next_byte_mask())
            parser.pos = curr_pos
            if not child_dominate:
                return False
    return True