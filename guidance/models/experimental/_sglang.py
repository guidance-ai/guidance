from typing import Iterator, Optional

from guidance.types import SamplingParams

from ..._ast import GrammarNode, JsonNode, RegexNode, RuleNode
from ...trace import OutputAttr, TextOutput
from .._base import Model
from .._openai_base import BaseOpenAIInterpreter, OpenAIClientWrapper


class SglangInterpreter(BaseOpenAIInterpreter):
    def __init__(
        self,
        model: str,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs,
    ):
        try:
            import openai
        except ImportError as ie:
            raise Exception(
                "Please install the openai package version >= 1 using `pip install openai -U` in order to use guidance.models.OpenAI!"
            ) from ie

        client = openai.OpenAI(base_url=base_url, api_key=api_key, **kwargs)
        super().__init__(model=model, client=OpenAIClientWrapper(client), **kwargs)

    def rule(self, node: RuleNode, **kwargs) -> Iterator[OutputAttr]:
        kwargs = self._process_kwargs(**kwargs)

        # Disable this check for now as all the supported endpoints have 'stop' support.
        if node.suffix:
            raise ValueError("suffix not yet supported for sglang endpoint")
        if node.stop_capture:
            raise ValueError("stop_capture not yet supported for sglang endpoint")

        kwargs = kwargs.copy()
        if node.temperature:
            kwargs["temperature"] = node.temperature
        if node.max_tokens:
            kwargs["max_tokens"] = node.max_tokens
        if node.stop:
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

        if "extra_body" not in kwargs:
            kwargs["extra_body"] = {}

        kwargs["extra_body"].update(dict(regex=node.regex))

        buffer: str = ""
        for attr in self._run(**kwargs):
            if isinstance(attr, TextOutput):
                buffer += attr.value
            yield attr

    def json(self, node: JsonNode, **kwargs) -> Iterator[OutputAttr]:
        kwargs = self._process_kwargs(**kwargs)

        if node.schema is not None:
            # set additionalProperties to False but allow it to be overridden
            node.schema["additionalProperties"] = node.schema.get("additionalProperties", False)

        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "json_schema",
                "schema": node.schema if node.schema is not None else {"type": "object"},
                "strict": True,
            },
        }

        return self._run(
            response_format=response_format,
            **kwargs,
        )

    def grammar(self, node: GrammarNode, **kwargs) -> Iterator[OutputAttr]:
        buffer: str = ""

        kwargs = self._process_kwargs(**kwargs)
        extra_body = {"ebnf": node.ll_grammar()}
        kwargs["extra_body"].update(extra_body)

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
        if "extra_body" not in kwargs:
            kwargs["extra_body"] = {}

        sampling_params = kwargs.pop("sampling_params", None)
        if sampling_params is None:
            return kwargs

        if "top_p" in sampling_params:
            kwargs["top_p"] = sampling_params["top_p"]

        # top_k must be put in extra_body
        top_k = sampling_params.pop("top_k", None)
        if top_k is not None:
            kwargs["extra_body"]["top_k"] = top_k

        min_p = sampling_params.pop("min_p", None)
        if min_p is not None:
            kwargs["extra_body"]["min_p"] = min_p

        repetition_penalty = sampling_params.pop("repetition_penalty", None)
        if repetition_penalty is not None:
            kwargs["extra_body"]["repetition_penalty"] = repetition_penalty

        return kwargs


class SglangModel(Model):
    def __init__(self, model: str, sampling_params: Optional[SamplingParams] = None, echo: bool = True, **kwargs):
        super().__init__(
            interpreter=SglangInterpreter(model=model, **kwargs),
            sampling_params=SamplingParams() if sampling_params is None else sampling_params,
            echo=echo,
        )
