import copy
import logging
import queue
import re
import threading

import time
from typing import Iterator, Optional, TYPE_CHECKING


import numpy as np

from guidance.trace._trace import EndOfCellTextOutput
from guidance.visual._message import JupyterCellExecutionCompletedMessage

from ..trace import NodeAttr, StatelessGuidanceInput, StatefulGuidanceInput, LiteralInput, EmbeddedInput, \
    RoleOpenerInput, RoleCloserInput, TextOutput, CaptureOutput, TraceHandler
from ..visual import TraceMessage, AutoRenderer, trace_node_to_str, trace_node_to_html, GuidanceMessage, Renderer
from ..visual._message import JupyterCellExecutionCompletedOutputMessage, MockMetricMessage, JupyterCellExecutionCompletedMessage

try:
    from IPython.display import clear_output, display, HTML

    ipython_is_imported = True
except ImportError:
    ipython_is_imported = False

logger = logging.getLogger(__name__)

from .._schema import BaseGenToken, EngineCallResponse, EngineOutput, GenToken, GuidanceEngineMetrics, VisBytesChunk
from .._utils import softmax, CaptureEvents
from .._parser import TokenParser
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


# TODO(nopdive): Remove on implementation.
class MockMetricsGenerator:
    def __init__(self, renderer: Renderer):
        from ..visual._async import run_async_task

        self._renderer = renderer
        run_async_task(self._emit())

    async def _emit(self):
        import asyncio
        import time

        time_start = time.time()
        while True:
            await asyncio.sleep(5)
            time_end = time.time()
            time_elapsed = time_end - time_start

            logger.debug(f"mock:metric:{time_elapsed}")
            self._renderer.update(
                MockMetricMessage(name='time taken', value=time_elapsed)
            )


class Engine:
    """The engine owns the inference computation and is used/created by the Model class.

    Engine objects represent the expensive parts of inference. While Model objects are cheap and do not
    need to know about the tokenizer or the model parameters, Engine objects know about both. Many
    Model objects can reference a single Engine object. Engine objects can also be hidden behind a
    Server so a single server can serve many clients' model objects through a single Engine object.
    """

    def __init__(self, tokenizer: Tokenizer, compute_log_probs=False, **kwargs):
        self.tokenizer = tokenizer
        self.compute_log_probs = compute_log_probs
        self.metrics = GuidanceEngineMetrics()

        self.trace_handler = TraceHandler()
        self.renderer = AutoRenderer(self.trace_handler,  use_legacy_renderer=kwargs.get("use_legacy_renderer", True))
        self.renderer.subscribe(self._msg_recv)
        self.model_dict: dict[int, Model] = {}

        # TODO(nopdive): Remove
        # self.metrics_generator = MockMetricsGenerator(self.renderer)

    def _msg_recv(self, message: GuidanceMessage) -> None:
        # NOTE(nopdive): This is likely running on a secondary thread.
        logger.debug(f"ENGINE:{message}")
        if isinstance(message, JupyterCellExecutionCompletedMessage):
            logger.debug(f"ENGINE:cell executed")
            self.renderer.update(message)

        # model = self.model_dict.get(message.trace_id)
        # print(f"ENGINE:{message}", type(message))

        if isinstance(message, JupyterCellExecutionCompletedMessage):
            # print("last_state")
            last_model: "Model" = self.model_dict[message.last_trace_id]
            paths = []
            model = last_model
            while model is not None:
                paths.append(model)
                if model._parent_id is None:
                    break

                model: "Model" = self.model_dict[model._parent_id]

            paths.reverse()

            vis_chunks: list[VisBytesChunk] = [path.vis_chunk for path in paths if path.vis_chunk is not None]

            tokens = []
            gen_tokens: list[GenToken] = []
            for vis_chunk in vis_chunks:
                if vis_chunk.is_input:
                    for gen_token in vis_chunk.input_tokens:
                        tokens.append(gen_token.token)
                        gen_tokens.append(gen_token)
                else:
                    for gen_token in vis_chunk.generated_tokens:
                        tokens.append(gen_token.token)
                        gen_tokens.append(gen_token)

                    for gen_token in vis_chunk.force_forwarded_tokens:
                        tokens.append(gen_token.token)
                        gen_tokens.append(gen_token)

            probs = self.get_token_probs(tokens)

            for gen_token, prob in zip(gen_tokens, probs):
                gen_token.top_k = prob
                for _item in prob:
                    if _item.token == gen_token.token:
                        gen_token.prob = _item.prob
                        break

            self.renderer.update(JupyterCellExecutionCompletedOutputMessage(
                trace_id=message.last_trace_id,
                text=self.tokenizer.decode(tokens).decode("utf-8"),
                tokens=gen_tokens,
            ))

            # self.renderer.update(JupyterCellExecutionCompletedOutputMessage(
            #     trace_id=message.last_trace_id,
            #     text=self.tokenizer.decode(tokens).decode("utf-8"),
            #     tokens=tokens,
            #     probs=probs
            # ))

    def get_chat_template(self): # TODO [HN]: Add more logic here...should we instantiate class here? do we even need to?
        return self.tokenizer.chat_template() # Instantiate the class before returning to client for now
    
    def reset_metrics(self):
        self.metrics = GuidanceEngineMetrics()

    def start(self, prompt, grammar, ensure_bos_token=True) -> TokenParser:
        """Start processing parser state executed through the grammar.

        Parameters
        ----------
        prompt : str or Parser
            This is represents the current state of a guidance parser that will be extended
            using the passed grammar. If a string is given then we assume the previous parser
            state is just a fixed string prompt, if a full Parser is given then we extend that
            parser by appending the new grammar to the parser's current grammar and then
            inferencing the model. (TODO: implement full parser extension support)
        grammar: Grammar
            This is the grammar we are extending the prompt with.
        """
        # def __call__(self, grammar, max_tokens=1000000, n=1, top_p=1, temperature=0.0, ensure_bos_token=True):
        # assert n == 1, "Still need to add support for n > 1!"

        # TODO: re-enable this? llguidance currently doesn't support model variables
        # note we only support a fixed set of engine variables for the sake of security
        # self._replacements = replace_model_variables(
        #     grammar, self, allowed_vars=["eos_token", "bos_token"]
        # )

        # right now we only support a text/bytes prompt parser state, so we extract that
        if isinstance(prompt, bytes):
            prompt = prompt
        elif isinstance(prompt, str):
            prompt = bytes(prompt, encoding="utf8")
        elif isinstance(prompt, TokenParser):
            raise NotImplementedError(
                "Still need to implement support for extending a full Parser trace."
            )
        else:
            raise Exception("The passed prompt is of an unknown type!")

        return TokenParser(
            grammar=grammar,
            tokenizer=self.tokenizer,
            prompt=prompt,
            ensure_bos_token=ensure_bos_token
        )

    def __call__(self, prompt, grammar, ensure_bos_token=True) -> Iterator[EngineCallResponse]:
        """Main entry point for the inference-parser loop. Yields EngineCallResponse objects as
        the parser advances through the grammar.

        Parameters
        ----------
        prompt : str or Parser
            This is represents the current state of a guidance parser that will be extended
            using the passed grammar. If a string is given then we assume the previous parser
            state is just a fixed string prompt, if a full Parser is given then we extend that
            parser by appending the new grammar to the parser's current grammar and then
            inferencing the model. (TODO: implement full parser extension support)
        grammar: Grammar
            This is the grammar we are extending the prompt with.
        """
        parser = self.start(prompt, grammar, ensure_bos_token)

        engine_output = None
        while not parser.done():
            t0 = time.time()

            gen_data, response = parser.advance(engine_output)

            if gen_data is not None:
                is_in_accepting_state = (
                    parser.is_accepting() and self.tokenizer.eos_token_id is not None
                )

                mask = None
                if is_in_accepting_state:
                    # Whenever we are in an accepting state, we will allow the model to generate whatever it wants
                    # but we will treat any "illegal" tokens as EOS, allowing the model to finish gracefully.
                    # Hence, mask must be None
                    assert gen_data.mask[self.tokenizer.eos_token_id]
                else:
                    mask = gen_data.mask

                engine_output = self.get_next_top_k_tokens(
                    token_ids=gen_data.tokens,
                    mask=mask,
                    temperature=gen_data.temperature,
                )[0]

                if is_in_accepting_state and not gen_data.mask[engine_output.issued_token.token]:
                    engine_output.issued_token.token = self.tokenizer.eos_token_id
                    engine_output.issued_token.bytes = self.tokenizer.decode(
                        [engine_output.issued_token.token]
                    )
                    # TODO: Should we set the prob to 1.0 here?
                    engine_output.issued_token.prob = 1.0
            else:
                engine_output = None

            if response:
                response.latency_ms = (time.time() - t0) * 1000

            yield response

    def get_next_top_k_tokens(
        self,
        token_ids: list[int],
        mask: Optional[bytes],
        temperature: float,
        k: int = 5,
    ) -> list[EngineOutput]:
        t0 = time.time()
        new_tokens_logits = [self.get_logits(token_ids)]
        if new_tokens_logits is None:
            return []

        lat_ms = (time.time() - t0) * 1000

        def get_top_k(_probs: np.ndarray, _k: int = 5) -> list[GenToken]:
            top_k_indices = np.argsort(_probs)[::-1][:_k]
            top_k_probs = _probs[top_k_indices]

            return [
                GenToken(
                    token=token,
                    prob=prob,
                    text=self.tokenizer.decode([token]).decode("utf-8"),
                    latency_ms=lat_ms,
                    is_generated=True,
                )
                for token, prob in zip(top_k_indices, top_k_probs)
                if prob > 0
            ]

        # compute top-k without masking
        probs = (
            softmax(new_tokens_logits)
            if temperature < 0.0001
            else softmax(new_tokens_logits / temperature)
        )

        engine_list = []
        unseen_tokens = token_ids[-len(probs) :][1:]
        # for _probs, _logits in zip(probs, logits):

        # we're missing the very first token
        if len(token_ids) == len(probs):
            first_token = token_ids[0]
            engine_list.insert(
                0,
                EngineOutput(
                    issued_token=GenToken(
                        token=first_token,
                        prob=1.0,
                        text=self.tokenizer.decode([first_token]).decode("utf-8"),
                        latency_ms=0,
                        is_generated=False,
                    ),
                    top_k=[],
                    masked_top_k=[],
                    is_backtracked=False,
                ),
            )

        for i, _probs in enumerate(probs):
            top_k: list[GenToken] = get_top_k(_probs, k)
            _logits = new_tokens_logits[i]

            # compute top-k with masking
            masked_top_k: list[GenToken] = []
            if mask is not None:
                masked_logits = _logits * np.frombuffer(mask, dtype=np.uint8)
                masked_probs = (
                    softmax(masked_logits)
                    if temperature < 0.0001
                    else softmax(masked_logits / temperature)
                )
                masked_top_k = get_top_k(masked_probs, k)

            issued_token = masked_top_k[0] if len(masked_top_k) > 0 else top_k[0]
            if i < len(unseen_tokens):
                token = unseen_tokens[i]
                issued_token = GenToken(
                    token=token,
                    prob=_probs[token],
                    text=self.tokenizer.decode([token]).decode("utf-8"),
                    latency_ms=0,
                    is_generated=False,
                )

            engine_list.append(EngineOutput(
                issued_token=issued_token,
                top_k=top_k,
                masked_top_k=None if not masked_top_k else masked_top_k,
                is_backtracked=False,
            ))

        return engine_list

    def get_next_token(self, token_ids: list[int], mask: Optional[bytes], temperature: float) -> int:
        """Base implementation for getting the next token from the model which calls get_logits and sample_with_temperature.
        Subclasses may override this method, e.g. if they use external APIs that do not support getting logits directly.
        """
        logits = self.get_logits(token_ids)
        token = self.sample_with_temperature(logits, mask, temperature)
        return token

    def get_logits(self, token_ids: list[int]) -> np.ndarray:
        raise NotImplementedError
    
    def get_token_probs(self, token_ids: list[int], top_k: int = 5) -> list[list[BaseGenToken]]:
        raise NotImplementedError

    def sample_with_temperature(self, logits: np.ndarray, mask: Optional[bytes], temperature: float) -> int:
        if mask is not None:
            logits += np.frombuffer(mask, dtype=np.uint8)
        if temperature < 0.0001:
            return int(np.argmax(logits))
        # Get probabilities from softmax
        probabilities = softmax(logits/temperature)
        # Sample an index based on the probabilities
        sampled_index = np.random.choice(len(logits), p=probabilities)
        return sampled_index

    def _report_failed_match(self, prompt):
        """Note that this can be overridden by subclasses that have more likely reasons than a bug in the token set (like remote models)."""
        return Exception(
            "We can't consume any more tokens, but we are not yet done! Perhaps your model's token set is incomplete? This happened after the prompt:"
            + str(prompt[-40:])
        )


_id_counter = 0  # Counter for identifiers, this has to be outside the model to handle child classes properly.
class Model:
    """The base guidance model object, which represents a model in a given state.

    Model objects are immutable representations of model state, so whenever you change
    them you get a new Model object. However, these copies share the "expensive"
    parts of the underlying model like the parameters and KV-cache, through a shared
    Engine, so making copies of Model objects is cheap.

    .. automethod:: __add__
    """

    global_active_blocks: list["ContextBlock"] = []  # track what context blocks are globally active

    _grammar_only = 0  # a flag that tracks when we are forced to be executing only compiled grammars (like when we are inside a select)

    def __init__(self, engine, echo=True, parent_id=None, **kwargs):
        """Build a new model object that represents a model in a given state.

        Note that this constructor is not meant to be used directly, since there

        Parameters
        ----------
        engine : Engine
            The inference engine to use for this model.
        echo : bool
            If true the final result of creating this model state will be displayed (as HTML in a notebook).
        parent_id : int
            Parent model's identifier.
        """
        if isinstance(engine, str) and engine.startswith("http"):
            from ._remote import RemoteEngine

            engine = RemoteEngine(engine, **kwargs)

        # # auto-wrap the tokenizer in the standard guidance interface
        # if not isinstance(tokenizer, Tokenizer):
        #     tokenizer = Tokenizer(tokenizer)

        self.engine = engine
        self.chat_template = engine.get_chat_template() # TODO [HN]: Should this be a method or attr?
        # NOTE(nopdive): `echo` seems to be better on the engine, when is there an opportunity to turn echo off midway?
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
        self._trace_handler = engine.trace_handler  # builds state for models
        if self.echo:
            self._renderer = engine.renderer  # renderer for display
        else:
            self._renderer = None  # no renderer if echo is false
        self._event_queue = None  # TODO: these are for streaming results in code, but that needs implemented
        self._event_parent = None
        self._last_display = 0  # used to track the last display call to enable throttling
        self._last_event_stream = 0  # used to track the last event streaming call to enable throttling

        self._id = self.__class__.gen_id()  # model id needed for tracking state
        self._parent_id = parent_id
        self._parent: "Model" = None
        self._update_trace_node(self._id, self._parent_id, None)

        self.vis_chunk: VisBytesChunk = None
        self.engine.model_dict[self._id] = self

    @classmethod
    def gen_id(cls):
        global _id_counter

        _id = _id_counter
        _id_counter += 1
        return _id

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

        return trace_node_to_html(
            self._trace_handler.id_node_map[self._id],
            hasattr(self, "indent_roles")
        )

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

        new_lm._id = self.__class__.gen_id()
        new_lm._parent_id = self._id
        self._update_trace_node(new_lm._id, new_lm._parent_id, None)
        self.engine.model_dict[new_lm._id] = new_lm
        new_lm.vis_chunk = None
        new_lm._parent = self

        return new_lm

    def _inplace_append(self, value, force_silent=False):
        """This is the base way to add content to the current LM object that is being constructed.

        All updates to the model state should eventually use this function.
        Note this should only be used after making a copy, otherwise immutability would be violated.

        Parameters
        ----------
        value : bytes | str
            The bytes we should append to our current state.
        """

        # update the byte state
        v = value
        if not isinstance(v, str):
            v = str(value)
        self._state += v

        # this is for programmatic streaming among other things
        self._send_to_event_queue(self)


    def reset(self, clear_variables=True):
        """This resets the state of the model object.

        Parameters
        ----------
        clear_variables : bool
            If we should clear all the model object's variables in addition to reseting the byte state.
        """
        # TODO(nopdive): This violates the immutability assumption on model class for users. Remove on confirmation.

        self._state = self._state[:0]
        if clear_variables:
            self._variables = {}
            self._variables_log_probs = {}
        return self


    def role_opener(self, role_name, **kwargs):
        # TODO [HN]: Temporary change while I instrument chat_template in transformers only.
        # Eventually have all models use chat_template.
        if hasattr(self, "get_role_start"):
            return self.get_role_start(role_name, **kwargs)
        elif hasattr(self, "chat_template"):
            return self.chat_template.get_role_start(role_name)
        else:
            raise Exception(
                f"You need to use a chat model in order the use role blocks like `with {role_name}():`! Perhaps you meant to use the {type(lm).__name__}Chat class?"
            )


    def role_closer(self, role_name, **kwargs):
        # TODO [HN]: Temporary change while I instrument chat_template in transformers only.
        # Eventually have all models use chat_template.
        if hasattr(self, "get_role_end"):
            return self.get_role_end(role_name, **kwargs)
        elif hasattr(self, "chat_template"):
            return self.chat_template.get_role_end(role_name)
        else:
            raise Exception(
                f"You need to use a chat model in order the use role blocks like `with {role_name}():`! Perhaps you meant to use the {type(lm).__name__}Chat class?"
            )


    def _repr_html_(self):
        if ipython_is_imported:
            clear_output(wait=True)
        return self._html()

    def _current_prompt(self):
        """The current prompt in bytes (which is the state without the context close tags)."""
        return trace_node_to_str(self._trace_handler.id_node_map[self._id])

    def _update_trace_node(self, identifier: int, parent_id: Optional[int], node_attr: Optional[NodeAttr]):
        """Updates trace node that corresponds to this model."""

        self._trace_handler.update_node(identifier, parent_id, node_attr)
        if self._renderer is not None:
            self._renderer.update(
                TraceMessage(
                    trace_id=identifier,
                    parent_trace_id=parent_id,
                    node_attr=node_attr,
                )
            )


    def __str__(self):
        """A string representation of the current model object (that includes context closers)."""

        # TODO(nopdive): Ensure context closers or no?
        return trace_node_to_str(self._trace_handler.id_node_map[self._id])


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

        # find blocks that are now active, but haven't been opened by lm yet
        enter_blocks = []
        for context in Model.global_active_blocks:
            if context not in lm.opened_blocks:
                enter_blocks.append(context)
                lm.opened_blocks[context] = (0, "")

        # find opened blocks by lm, but are no longer active
        exit_blocks = []
        for context in list(reversed(lm.opened_blocks.keys())):
            if context not in Model.global_active_blocks:
                exit_blocks.append(context)

        # finish any exiting blocks
        for context in exit_blocks:
            pos, close_text = lm.opened_blocks[context]
            del lm.opened_blocks[context]

            # handle variables
            if context.name is not None:
                # TODO(nopdive): Replace with trace traversal.
                v = format_pattern.sub("", lm._state[pos:])
                lm._variables[context.name] = v
                self._update_trace_node(lm._id, lm._parent_id, CaptureOutput(name=context.name, value=v))

            # add closer
            # TODO(nopdive): Consider removing context closer/opener on confirmation.
            closer_text = self.role_closer(context.name)
            self._update_trace_node(lm._id, lm._parent_id, RoleCloserInput(name=context.name, text=closer_text))
            lm += context.closer
            lm = lm.copy()

        # start any entering blocks
        for context in enter_blocks:
            # add opener
            opener_text = self.role_opener(context.name)
            self._update_trace_node(lm._id, lm._parent_id, RoleOpenerInput(name=context.name, text=opener_text))
            lm += context.opener
            lm = lm.copy()

            # store closer for state extraction later
            close_text = self.role_closer(context.name)
            lm.opened_blocks[context] = (len(lm._state), close_text)

            # handle variables
            # NOTE(nopdive): No stack for variables, this process removes shadowed variables?
            if context.name is not None:
                if context.name in lm._variables:
                    del lm._variables[context.name]
                    if context.name in lm._variables_log_probs:
                        del lm._variables_log_probs[context.name]

        if isinstance(value, TextOutput):
            lm._inplace_append(value.value)
            out = lm
            self._update_trace_node(out._id, out._parent_id, value)
        elif isinstance(value, CaptureOutput):
            self._update_trace_node(lm._id, lm._parent_id, value)
            out = lm
        elif isinstance(value, str):
            # wrap raw string values

            is_id = False
            parts = re.split(_tag_pattern, value)

            # we have no embedded objects
            if len(parts) == 1:
                self._update_trace_node(lm._id, lm._parent_id, LiteralInput(value=value))

                lm._inplace_append(value)
                out = lm

                # generate VisBytesChunk
                _bytes = value.encode("utf-8")
                _tokens = out.engine.tokenizer.encode(_bytes)
                out.vis_chunk = VisBytesChunk(
                    bytes=_bytes,
                    is_input=True,
                    input_bytes=_bytes,
                    input_tokens=[
                        GenToken(
                            token=_token,
                            prob=1.0,
                            text=out.engine.tokenizer.decode([_token]).decode("utf-8"),
                            latency_ms=0,
                            is_generated=False,
                            is_force_forwarded=False,
                            is_input=True,
                        ) for _token in _tokens
                    ]
                )

                self._update_trace_node(out._id, out._parent_id, TextOutput(value=value, is_input=True, tokens=out.vis_chunk.input_tokens))

            # if we have embedded objects we have to convert the string to a grammar tree
            else:
                self._update_trace_node(lm._id, lm._parent_id, EmbeddedInput(value=value))

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
            self._update_trace_node(lm._id, lm._parent_id, StatelessGuidanceInput(value=value))
            out = lm._run_stateless(value)

        # run stateful functions
        else:
            self._update_trace_node(lm._id, lm._parent_id, StatefulGuidanceInput(value=value))
            out = value(lm)
            if out is None:
                raise Exception(
                    f"A guidance function returned `None`, not a model object! Did you forget to return the new lm at the end of your function?"
                )
            if not isinstance(out, Model):
                raise Exception(
                    f"A guidance function did not return a model object! Did you try to add a function to a model without calling the function? For example `model + guidance_function()` is correct, while `model + guidance_function` will cause this error."
                )

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

            new_lm_created = True
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

                while chunk.backtrack > 0:
                    parent = lm._parent
                    while parent is not None:
                        if parent.vis_chunk is not None:
                            break

                        parent = parent._parent

                    if parent.vis_chunk.input_tokens:
                        parent.vis_chunk.input_tokens.pop()
                        chunk.backtrack -= 1
                    elif parent.vis_chunk.generated_tokens:
                        parent.vis_chunk.generated_tokens.pop()
                        chunk.backtrack -= 1
                    elif parent.vis_chunk.force_forwarded_tokens:
                        parent.vis_chunk.force_forwarded_tokens.pop()
                        chunk.backtrack -= 1                    

                if len(chunk.new_bytes) > 0:
                    generated_value += new_text

                    # lm += TextOutput(
                    #     value=new_text,
                    #     is_generated=chunk.is_generated,
                    #     token_count=chunk.new_token_count,
                    #     prob=chunk.new_bytes_prob,
                    # )

                    if chunk.generated_bytes:
                        lm += TextOutput(
                            value=chunk.generated_bytes.decode("utf8"),
                            is_generated=True,
                            token_count=0,
                            prob=0.0,
                            tokens=chunk.generated_tokens,
                        )
                    
                    if chunk.force_forwarded_bytes:
                        lm += TextOutput(
                            value=chunk.force_forwarded_bytes.decode("utf8"),
                            is_force_forwarded=True,
                            token_count=0,
                            prob=0.0,
                            tokens=chunk.force_forwarded_tokens,
                        )
                    
                    new_lm_created = True
                else:
                    new_lm_created = False

                if not lm.vis_chunk or new_lm_created:
                    lm.vis_chunk = VisBytesChunk(
                        bytes=chunk.new_bytes,
                        is_input=False,
                        generated_bytes=chunk.generated_bytes,
                        generated_tokens=chunk.generated_tokens,
                        force_forwarded_bytes=chunk.force_forwarded_bytes,
                        force_forwarded_tokens=chunk.force_forwarded_tokens,
                        backtrack=chunk.backtrack,
                    )
                else:
                    # append to existing VisBytesChunk
                    lm.vis_chunk.bytes += chunk.new_bytes
                    lm.vis_chunk.generated_bytes += chunk.generated_bytes
                    lm.vis_chunk.force_forwarded_bytes += chunk.force_forwarded_bytes
                    lm.vis_chunk.backtrack += chunk.backtrack

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
                                    lm += CaptureOutput(name=k)
                                if k not in lm._variables_log_probs or not isinstance(lm._variables_log_probs[k], list):
                                    lm._variables_log_probs[k] = []

                                lm._variables[k].append(inner_v)
                                lm._variables_log_probs[k].append(
                                    chunk.capture_group_log_probs[k][i]
                                )
                                lm += CaptureOutput(
                                    name=k,
                                    value=inner_v,
                                    is_append=True,
                                    log_probs=lm._variables_log_probs[k][i],
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
                            lm += CaptureOutput(
                                name=k,
                                value=v,
                                log_probs=chunk.capture_group_log_probs[k],
                            )

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


class ConstraintException(Exception):
    def __init__(self, *args, **kwargs):
        self.prompt = kwargs.pop("prompt", None)
        self.data = kwargs.pop("data", None)
        super().__init__(*args, **kwargs)
