from typing import TYPE_CHECKING, Any, ContextManager, Iterator, Optional

from guidance._schema import SamplingParams

from ..._ast import GrammarNode, JsonNode, RegexNode, RuleNode
from ...trace import OutputAttr, TextOutput
from .._base import Model
from .._openai_base import BaseOpenAIClientWrapper, BaseOpenAIInterpreter

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletionChunk
import contextlib


class LiteLLMOpenAIClientWrapper(BaseOpenAIClientWrapper):
    def __init__(self, router):
        self.router = router

    @contextlib.contextmanager
    def _wrapped_completion(
        self,
        model: str,
        messages: list[dict[str, Any]],
        logprobs: bool,
        **kwargs,
    ) -> Iterator["ChatCompletionChunk"]:
        """Wrapped completion call within a context manager."""
        kwargs["stream"] = True  # Ensure we are streaming here
        stream_wrapper = self.router.completion(
            model=model,
            messages=messages,
            logprobs=logprobs,
            **kwargs,
        )

        try:
            yield stream_wrapper
        finally:
            # stream_wrapper.completion_stream is the underlying stream, e.g. openai.Stream
            if hasattr(stream_wrapper.completion_stream, "close"):
                # Close the stream if it has a close method
                stream_wrapper.completion_stream.close()

    def streaming_chat_completions(
        self,
        model: str,
        messages: list[dict[str, Any]],
        logprobs: bool,
        **kwargs,
    ) -> ContextManager[Iterator["ChatCompletionChunk"]]:
        """Streaming chat completions."""
        return self._wrapped_completion(
            model=model,
            messages=messages,
            logprobs=logprobs,
            **kwargs,
        )  # type: ignore[return-value]


class LiteLLMInterpreter(BaseOpenAIInterpreter):
    SUPPORTED_ENDPOINT_TYPES = ["openai", "azure_ai", "azure", "gemini", "anthropic", "xai", "hosted_vllm", "groq"]

    def __init__(self, model_description: dict, **kwargs):
        try:
            import litellm
        except ImportError as ie:
            raise Exception(
                "Please install the litellm package version >= 1.71.0 using `pip install litellm -U` in order to use guidance.models.LiteLLM!"
            ) from ie

        self.ep_type = self._check_model(model_description)
        # set default model to the first one in the list
        self.model = model_description["model_name"]
        self.router = litellm.Router(model_list=[model_description])
        self.client = LiteLLMOpenAIClientWrapper(self.router)

        # Disable log_probs for any remote endpoints by default.
        # Otherwise, generation will fail for some endpoints.
        self.logprobs = False

        super().__init__(model=self.model, client=self.client, **kwargs)

    def _check_model(self, model_desc: dict) -> str:
        """Check if the model description is valid."""
        for ep_type in self.SUPPORTED_ENDPOINT_TYPES:
            if model_desc["litellm_params"]["model"].startswith(ep_type):
                return ep_type

        raise Exception(
            "Cannot parse endpoint type. "
            "Please use this 'model' format in 'litellm_params': endpoint_type/model_name (e.g., openai/gpt-3.5-turbo). "
            "Supported endpoints are: " + ", ".join(self.SUPPORTED_ENDPOINT_TYPES)
        )

    def rule(self, node: RuleNode, **kwargs) -> Iterator[OutputAttr]:
        kwargs = self._process_kwargs(**kwargs)

        # Disable this check for now as all the supported endpoints have 'stop' support.
        # if node.stop and self.ep_type not in ["hosted_vllm", "azure", "azure_ai", "gemini", "openai", "xai", "anthropic"]:
        #     raise ValueError(f"stop condition not yet supported for {self.ep_type} endpoint")
        if node.suffix:
            raise ValueError(f"suffix not yet supported for {self.ep_type} endpoint")
        if node.stop_capture:
            raise ValueError(f"stop_capture not yet supported for {self.ep_type} endpoint")

        kwargs = kwargs.copy()
        if node.temperature:
            kwargs["temperature"] = node.temperature
        if node.max_tokens:
            kwargs["max_tokens"] = node.max_tokens
        if node.stop:
            if self.ep_type in ["xai"] and isinstance(node.stop.regex, str):
                kwargs["stop"] = [node.stop.regex]
            else:
                kwargs["stop"] = node.stop.regex

        chunks = self.run(node.value, **kwargs)
        if node.capture:
            buffered_text = ""
            for chunk in chunks:
                # TODO: this isinstance check is pretty darn fragile.
                # ~there must be a better way~
                if isinstance(chunk, TextOutput):
                    buffered_text += chunk.value
                yield chunk
            yield self.state.apply_capture(
                name=node.capture,
                value=buffered_text,
                log_prob=1,  # TODO
                is_append=node.list_append,
            )
        else:
            yield from chunks

    def regex(self, node: RegexNode, **kwargs) -> Iterator[OutputAttr]:
        kwargs = self._process_kwargs(**kwargs)
        if node.regex is not None and self.ep_type not in ["hosted_vllm"]:
            raise ValueError(f"Regex is not yet supported for ep {self.ep_type}")

        if self.ep_type == "hosted_vllm":
            return self._regex_vllm(node, **kwargs)

        # We're in unconstrained mode now.
        return self._run(**kwargs)

    def _regex_vllm(self, node: RegexNode, **kwargs):
        """Run the regex node using vLLM."""
        if "extra_body" not in kwargs:
            kwargs["extra_body"] = {}

        kwargs["extra_body"].update(dict(guided_decoding_backend="guidance", guided_regex=node.regex))

        buffer: str = ""
        for attr in self._run(**kwargs):
            if isinstance(attr, TextOutput):
                buffer += attr.value
            yield attr

    def json(self, node: JsonNode, **kwargs) -> Iterator[OutputAttr]:
        kwargs = self._process_kwargs(**kwargs)
        if (
            self.ep_type in ["openai"]
            and (self.model in ["gpt-3.5-turbo"] or self.model.startswith("gpt-4-"))
            and node.schema is not None
        ):
            raise ValueError(f"json_schema is not supported for ep {self.ep_type}/{self.model}")

        if self.ep_type in ["azure_ai"]:
            raise ValueError(f"json_object/json_schema are not supported for ep {self.ep_type}")

        if node.schema is None:
            response_format = {"type": "json_object"}
        else:
            # set additionalProperties to False but allow it to be overridden
            node.schema["additionalProperties"] = node.schema.get("additionalProperties", False)
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": "json_schema",
                    "schema": node.schema,
                    "strict": True,
                },
            }

        return self._run(
            response_format=response_format,
            **kwargs,
        )

    def grammar(self, node: GrammarNode, **kwargs) -> Iterator[OutputAttr]:
        kwargs = self._process_kwargs(**kwargs)
        if self.ep_type == "hosted_vllm":
            return self._grammar_vllm(node, **kwargs)

        raise ValueError(f"Grammar is not yet supported for ep {self.ep_type}")

    def _grammar_vllm(self, node: GrammarNode, **kwargs) -> Iterator[OutputAttr]:
        """Run the grammar node using vLLM."""
        if "extra_body" not in kwargs:
            kwargs["extra_body"] = {}

        kwargs["extra_body"].update(dict(guided_decoding_backend="guidance", guided_grammar=node.ll_grammar()))

        buffer: str = ""
        for attr in self._run(**kwargs):
            if isinstance(attr, TextOutput):
                buffer += attr.value
            yield attr
        matches = node.match(
            buffer,
            raise_exceptions=False,
            # Turn of max_tokens since we don't have access to the tokenizer
            enforce_max_tokens=False,
        )
        if matches is None:
            # TODO: should probably raise...
            # raise ValueError("vLLM failed to constrain the grammar")
            pass
        else:
            for name, value in matches.captures.items():
                log_probs = matches.log_probs[name]
                if isinstance(value, list):
                    assert isinstance(log_probs, list)
                    assert len(value) == len(log_probs)
                    for v, l in zip(value, log_probs):
                        yield self.state.apply_capture(name=name, value=v, log_prob=l, is_append=True)
                else:
                    yield self.state.apply_capture(name=name, value=value, log_prob=log_probs, is_append=False)

    def _process_kwargs(self, **kwargs):
        sampling_params = kwargs.pop("sampling_params", None)
        if sampling_params is None:
            return kwargs

        kwargs["top_p"] = sampling_params.pop("top_p", None)
        kwargs["top_k"] = sampling_params.pop("top_k", None)
        kwargs["min_p"] = sampling_params.pop("min_p", None)
        kwargs["repetition_penalty"] = sampling_params.pop("repetition_penalty", None)

        if self.ep_type == "groq":
            # Groq does not support top_k, min_p, or repetition_penalty
            kwargs.pop("top_k", None)
            kwargs.pop("min_p", None)
            kwargs.pop("repetition_penalty", None)

        return kwargs


class LiteLLM(Model):
    def __init__(
        self, model_description: dict, sampling_params: Optional[SamplingParams] = None, echo: bool = True, **kwargs
    ):
        interpreter = LiteLLMInterpreter(model_description=model_description, **kwargs)
        super().__init__(
            interpreter=interpreter,
            sampling_params=SamplingParams() if sampling_params is None else sampling_params,
            echo=echo,
        )
