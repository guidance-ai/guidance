from typing import Iterator
from ..._ast import GrammarNode
from ...trace import OutputAttr, TextOutput
from .._openai import BaseOpenAIInterpreter
from .._base import Model

class VLLMInterpreter(BaseOpenAIInterpreter):
    def grammar(self, node: GrammarNode, **kwargs) -> Iterator[OutputAttr]:
        buffer: str = ""
        for attr in self._run(
            extra_body = dict(
                guided_decoding_backend="guidance",
                guided_grammar=node.ll_grammar(),
            )
        ):
            if isinstance(attr, TextOutput):
                buffer += attr.value
            yield attr
        # TODO: apply captures by re-applying parser

class VLLMModel(Model):
    def __init__(self, model: str, echo=True, **kwargs):
        super().__init__(
            interpreter=VLLMInterpreter(model=model, **kwargs),
            echo=echo,
        )
