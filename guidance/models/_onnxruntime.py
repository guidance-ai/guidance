import operator
from itertools import takewhile
from typing import TYPE_CHECKING, Union

try:
    from ._transformers import TransformersTokenizer

    has_transformers = True
except ModuleNotFoundError:
    has_transformers = False

from guidance._schema import SamplingParams

from ._base import Model
from ._engine import Engine, EngineInterpreter, LogitsOutput

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

try:
    import onnxruntime_genai as og

    has_onnxrt_genai = True
except ModuleNotFoundError:
    has_onnxrt_genai = False


class OnnxRuntimeGenAIEngine(Engine):
    """The core class that runs inference using onnxruntime-genai."""

    def __init__(
        self,
        model: str,
        tokenizer: Union["PreTrainedTokenizer", "PreTrainedTokenizerFast", None] = None,
        chat_template=None,
        enable_backtrack=True,
        enable_ff_tokens=True,
        enable_monitoring=True,
        enable_token_probabilities=False,
        enable_top_k=False,
        top_k: int = 5,
        **kwargs,
    ):
        if not has_onnxrt_genai:
            raise Exception(
                "Please install onnxruntime-genai with `pip install --pre onnxruntime-genai` in order to use guidance.models.OnnxRuntimeGenAI!"
            )

        if not has_transformers:
            raise Exception(
                "Please install transformers with `pip install transformers` in order to use guidance.models.Transformers!"
            )

        self.config = og.Config(model)
        self.config.clear_providers()
        if "execution_provider" in kwargs:
            self.config.append_provider(kwargs["execution_provider"])

        self.model = og.Model(self.config)
        self.tokenizer = og.Tokenizer(self.model)

        self.search_options = {"batch_size": 1}
        self.params = og.GeneratorParams(self.model)
        self.params.set_search_options(**self.search_options)

        transformers_tokenizer = (
            TransformersTokenizer(
                hf_tokenizer=tokenizer,
                chat_template=chat_template,
            )
            if tokenizer is not None
            else TransformersTokenizer.from_pretrained(model, chat_template)
        )

        self._cached_token_ids = []
        self._cached_logits = None

        self.generator = None

        super().__init__(
            transformers_tokenizer,
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

        if self.generator is None:
            self.generator = og.Generator(self.model, self.params)

        if num_cached == 0:
            self._cached_token_ids.clear()
            self.generator.rewind_to(0)
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
                num_cached -= 1
                self.generator.rewind_to(num_cached)
                new_token_ids.append(token_ids[-1])
            else:
                extra_token_ids = len(token_ids) - num_cached
                new_token_ids.extend(token_ids[-extra_token_ids:])
                self.generator.rewind_to(num_cached)

            # truncate cache
            self._cached_token_ids = self._cached_token_ids[:num_cached]

        if len(new_token_ids) > 0:
            self.generator.append_tokens(new_token_ids)

        logits = self.generator.get_logits()[0]

        # Need to add special truncating logic here for weird models that have a different output size than tokenizer vocab
        logits = logits[:, : self.tokenizer._vocab_size]

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
        transformers_tokenizer: Union["PreTrainedTokenizer", "PreTrainedTokenizerFast", None] = None,
        echo=True,
        chat_template=None,
        enable_backtrack=True,
        enable_ff_tokens=True,
        enable_monitoring=True,
        sampling_params: SamplingParams | None = None,
        **kwargs,
    ):
        engine = OnnxRuntimeGenAIEngine(
            model=model,
            tokenizer=transformers_tokenizer,
            chat_template=chat_template,
            enable_backtrack=enable_backtrack,
            enable_ff_tokens=enable_ff_tokens,
            enable_monitoring=enable_monitoring,
            sampling_params=sampling_params,
            **kwargs,
        )
        super().__init__(
            interpreter=EngineInterpreter(engine),
            sampling_params=SamplingParams() if sampling_params is None else sampling_params,
            echo=echo,
        )
