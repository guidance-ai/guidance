import operator
from itertools import takewhile
from typing import TYPE_CHECKING, Optional, Union, cast

from guidance._schema import SamplingParams

from ._base import Model
from ._engine import Engine, EngineInterpreter, LogitsOutput, Tokenizer
from ._transformers import TransformersTokenizer

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
import transformers as transformers_package

try:
    import onnxruntime_genai as og

    is_onnxrt_genai = True
except ModuleNotFoundError:
    is_onnxrt_genai = False


class OnnxRuntimeGenAITokenizer(Tokenizer):
    pass


class OnnxRuntimeGenAIEngine(Engine):
    """The core class that runs inference using onnxruntime-genai."""

    def __init__(
        self,
        model,
        tokenizer,
        chat_template=None,
        enable_backtrack=True,
        enable_ff_tokens=True,
        enable_monitoring=True,
        enable_token_probabilities=False,
        enable_top_k=False,
        top_k: int = 5,
        **kwargs,
    ):
        if not is_onnxrt_genai:
            raise Exception(
                "Please install onnxruntime-genai with `pip install --pre onnxruntime-genai` in order to use guidance.models.OnnxRuntimeGenAI!"
            )

        self.config = og.Config(model)
        self.model = og.Model(self.config)
        self.tokenizer = og.Tokenizer(self.model)

        self.search_options = {"batch_size": 1}
        self.params = og.GeneratorParams(self.model)
        self.params.set_search_options(**self.search_options)

        self.hf_tokenizer = TransformersTokenizer(
            hf_tokenizer=tokenizer,
            chat_template=chat_template,
        )

        self._cached_token_ids = []
        self._cached_logits = None

        self.generator = None

        super().__init__(
            self.hf_tokenizer,
            enable_backtrack=enable_backtrack,
            enable_ff_tokens=enable_ff_tokens,
            enable_monitoring=enable_monitoring,
            enable_token_probabilities=enable_token_probabilities,
            enable_top_k=enable_top_k,
            top_k=top_k,
        )

    def get_logits(self, token_ids: list[int], include_all_uncached_tokens: bool = False) -> LogitsOutput:
        """Computes the logits for the given token state.

        This overrides a method from the LocalEngine class that is used to get
        inference results from the model.
        """

        assert not include_all_uncached_tokens, "include_all_uncached_tokens is not supported in OnnxRuntime-GenAi"

        if len(token_ids) == 0:
            raise ValueError("token_ids must contain some tokens.")

        new_token_ids = []
        num_cached = sum(takewhile(operator.truth, map(operator.eq, token_ids, self._cached_token_ids)))

        if num_cached == 0:
            self.generator = og.Generator(self.model, self.params)
            self._cached_token_ids.clear()
            new_token_ids.extend(token_ids)
        elif num_cached == len(token_ids) and num_cached == len(self._cached_token_ids):
            # last token input is the same as the last cached token, so return the last cached logits
            return {
                "logits": self._cached_logits,
                "n_tokens": len(token_ids),
                "n_cached": num_cached,
            }
        else:
            if num_cached == len(token_ids):
                # we need to pass at least one new token
                num_cached = num_cached - 1

            self.generator.rewind_to(num_cached - 1)
            self._cached_token_ids = self._cached_token_ids[:num_cached]  # truncate cached token ids
            extra_token_ids = len(token_ids) - num_cached
            new_token_ids.extend(token_ids[-extra_token_ids:])

        self.generator.append_tokens(new_token_ids)
        logits = self.generator.get_logits()[0]
        self._cached_logits = logits
        self._cached_token_ids.extend(new_token_ids)

        return {
            "logits": logits,
            "n_tokens": len(token_ids),
            "n_cached": num_cached,
        }


class OnnxRuntimeGenAI(Model):
    def __init__(
        self,
        model: str,
        hf_tokenizer: Union[
            "PreTrainedTokenizer",
            "PreTrainedTokenizerFast",
        ],
        interpreter_cls: Optional[type[EngineInterpreter]] = None,
        echo=True,
        chat_template=None,
        enable_backtrack=True,
        enable_ff_tokens=True,
        enable_monitoring=True,
        sampling_params: Optional[SamplingParams] = None,
        **kwargs,
    ):
        if interpreter_cls is None:
            interpreter_cls = EngineInterpreter

        engine = OnnxRuntimeGenAIEngine(
            model=model,
            tokenizer=hf_tokenizer,
            chat_template=chat_template,
            enable_backtrack=enable_backtrack,
            enable_ff_tokens=enable_ff_tokens,
            enable_monitoring=enable_monitoring,
            sampling_params=sampling_params,
            **kwargs,
        )
        client = interpreter_cls(engine)
        super().__init__(
            interpreter=client,
            sampling_params=SamplingParams() if sampling_params is None else sampling_params,
            echo=echo,
        )
