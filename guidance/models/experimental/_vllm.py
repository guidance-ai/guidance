from typing import Iterator, Optional

from guidance._schema import SamplingParams

from ..._ast import GrammarNode
from ...trace import OutputAttr, TextOutput
from .._openai_base import (
    BaseOpenAIInterpreter,
    OpenAIClientWrapper
)
from .._base import Model


class VLLMInterpreter(BaseOpenAIInterpreter):
    def __init__(
        self,
        model: str,
        default_sampling_params: Optional[SamplingParams],
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs,
    ):
        try:
            import openai
        except ImportError:
            raise Exception(
                "Please install the openai package version >= 1 using `pip install openai -U` in order to use guidance.models.OpenAI!"
            )
        
        client = openai.OpenAI(base_url=base_url, api_key=api_key, **kwargs)
        super().__init__(model=model, client=OpenAIClientWrapper(client), default_sampling_params=default_sampling_params, **kwargs)

    def grammar(self, node: GrammarNode, **kwargs) -> Iterator[OutputAttr]:
        buffer: str = ""
        
        kwargs = self._process_kwargs(**kwargs)
        extra_body = {
            "guided_decoding_backend" : "guidance",
            "guided_grammar" : node.ll_grammar(),
        }
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
                        yield self.state.apply_capture(
                            name=name, value=v, log_prob=l, is_append=True
                        )
                else:
                    yield self.state.apply_capture(
                        name=name, value=value, log_prob=log_probs, is_append=False
                    )
                    
    def _process_kwargs(self, **kwargs):
        if "extra_body" not in kwargs:
            kwargs["extra_body"] = {}
            
        # top_k must be put in extra_body
        top_k = kwargs.pop("top_k", None)
        if top_k is None:
            top_k = self.default_sampling_params.get("top_k", None)
        if top_k is not None:
            kwargs["extra_body"]["top_k"] = top_k
            
        return kwargs


class VLLMModel(Model):
    def __init__(self, model: str, default_sampling_params: Optional[SamplingParams], echo=True, **kwargs):
        super().__init__(
            interpreter=VLLMInterpreter(model=model, default_sampling_params=default_sampling_params, **kwargs),
            echo=echo,
        )
