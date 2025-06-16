from typing import Iterator, Optional

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
        super().__init__(model=model, client=OpenAIClientWrapper(client))

    def grammar(self, node: GrammarNode, **kwargs) -> Iterator[OutputAttr]:
        buffer: str = ""
        for attr in self._run(
            extra_body=dict(
                guided_decoding_backend="guidance",
                guided_grammar=node.ll_grammar(),
            )
        ):
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


class VLLMModel(Model):
    def __init__(self, model: str, echo=True, **kwargs):
        super().__init__(
            interpreter=VLLMInterpreter(model=model, **kwargs),
            echo=echo,
        )
