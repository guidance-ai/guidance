import re
from typing import Optional, TYPE_CHECKING, Union

from .._base import Model
from .._engine import EngineClient, EngineState, Llama3VisionClient, Phi3VisionClient
from ._engine import TransformersEngine


if TYPE_CHECKING:
     from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast

class Transformers(Model):
    def __init__(
        self,
        model: Union[str, "PreTrainedModel"],
        tokenizer: Union[
            "PreTrainedTokenizer",
            "PreTrainedTokenizerFast",
            None,
        ] = None,
        client_cls: Optional[type[EngineClient]] = None,
        echo=True,
        compute_log_probs=False,
        chat_template=None,
        enable_backtrack=True,
        enable_ff_tokens=True,
        enable_monitoring=True,
        **kwargs,
    ):
        """Build a new Transformers model object that represents a model in a given state."""
        if client_cls is None and isinstance(model, str):
            if re.search("Llama-3.*-Vision", model):
                client_cls = Llama3VisionClient
            elif re.search("Phi-3-vision", model):
                client_cls = Phi3VisionClient
        if client_cls is None:
                client_cls = EngineClient

        client = client_cls(
            TransformersEngine(
                model,
                tokenizer,
                compute_log_probs,
                chat_template=chat_template,
                enable_backtrack=enable_backtrack,
                enable_ff_tokens=enable_ff_tokens,
                enable_monitoring=enable_monitoring,
                **kwargs,
            )
        )
        super().__init__(
            client=client,
            state=EngineState(),
            echo=echo,
        )
